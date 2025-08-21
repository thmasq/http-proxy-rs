use anyhow::{Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use clap::Parser;
use hex;
use http::{HeaderValue, Method, StatusCode, Uri, header::HeaderName};
use pingora::prelude::*;
use pingora_cache::{
    CacheKey,
    eviction::{EvictionManager, lru::Manager as LruManager},
};
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{ProxyHttp, Session};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use url::Url;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:8080")]
    listen: String,

    #[arg(short, long, default_value = "config.json")]
    config: String,

    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheConfig {
    /// Enable caching
    enabled: bool,

    /// Maximum memory usage for cache in MB
    max_memory_mb: usize,

    /// Default TTL for cached responses in seconds
    default_ttl_seconds: u64,

    /// Maximum size of individual cached items in MB
    max_item_size_mb: usize,

    /// Cache only responses with these status codes
    cacheable_status_codes: Vec<u16>,

    /// Cache responses with these content types (empty = cache all)
    cacheable_content_types: Vec<String>,

    /// Don't cache responses from these domains
    no_cache_domains: Vec<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_mb: 256,
            default_ttl_seconds: 3600, // 1 hour
            max_item_size_mb: 10,
            cacheable_status_codes: vec![200, 203, 300, 301, 302, 404, 410],
            cacheable_content_types: vec![
                "text/html".to_string(),
                "text/css".to_string(),
                "text/javascript".to_string(),
                "application/javascript".to_string(),
                "application/json".to_string(),
                "image/jpeg".to_string(),
                "image/png".to_string(),
                "image/gif".to_string(),
                "image/webp".to_string(),
                "image/svg+xml".to_string(),
            ],
            no_cache_domains: vec![
                "api.twitter.com".to_string(),
                "graph.facebook.com".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Allowed domains for embedding (empty = allow all)
    allowed_domains: Vec<String>,

    /// Blocked domains
    blocked_domains: Vec<String>,

    /// Rate limiting: requests per minute per IP
    rate_limit_per_minute: u32,

    /// Custom headers to add to responses
    custom_headers: HashMap<String, String>,

    /// Whether to strip all tracking headers
    strip_tracking_headers: bool,

    /// User agent to use for upstream requests
    user_agent: String,

    /// Timeout for upstream requests in seconds
    upstream_timeout: u64,

    /// Cache configuration
    cache: CacheConfig,
}

impl Default for Config {
    fn default() -> Self {
        let mut custom_headers = HashMap::new();
        custom_headers.insert("X-Proxy-By".to_string(), "EmbedProxy/1.0".to_string());

        Self {
            allowed_domains: vec![],
            blocked_domains: vec![
                "malware.example.com".to_string(),
                "phishing.example.com".to_string(),
            ],
            rate_limit_per_minute: 60,
            custom_headers,
            strip_tracking_headers: true,
            user_agent: "Mozilla/5.0 (compatible; EmbedProxy/1.0)".to_string(),
            upstream_timeout: 30,
            cache: CacheConfig::default(),
        }
    }
}

#[derive(Debug)]
struct RateLimiter {
    requests: RwLock<HashMap<String, Vec<Instant>>>,
    limit: u32,
}

impl RateLimiter {
    fn new(limit: u32) -> Self {
        Self {
            requests: RwLock::new(HashMap::new()),
            limit,
        }
    }

    async fn check_rate_limit(&self, ip: &str) -> bool {
        let mut requests = self.requests.write().await;
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        let ip_requests = requests.entry(ip.to_string()).or_insert_with(Vec::new);

        // Remove old requests
        ip_requests.retain(|&time| time > minute_ago);

        if ip_requests.len() >= self.limit as usize {
            return false;
        }

        ip_requests.push(now);
        true
    }
}

#[derive(Debug)]
struct CacheEntry {
    data: Bytes,
    headers: ResponseHeader,
    created_at: SystemTime,
    ttl: Duration,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        if let Ok(elapsed) = self.created_at.elapsed() {
            elapsed > self.ttl
        } else {
            true // If we can't determine elapsed time, consider it expired
        }
    }
}

// Remove Debug derive and implement manually to avoid LruManager Debug requirement
struct InMemoryCache {
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    eviction_manager: Arc<LruManager<8>>, // 8 shards for good performance
    max_memory_bytes: usize,
    current_memory_bytes: Arc<RwLock<usize>>,
}

impl std::fmt::Debug for InMemoryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryCache")
            .field("max_memory_bytes", &self.max_memory_bytes)
            .finish()
    }
}

impl InMemoryCache {
    fn new(max_memory_mb: usize) -> Self {
        let max_memory_bytes = max_memory_mb * 1024 * 1024;
        // Use reasonable defaults for LRU capacity
        let estimated_items = max_memory_bytes / 1024; // Estimate ~1KB per item on average
        let capacity_per_shard = estimated_items / 8;

        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            eviction_manager: Arc::new(LruManager::with_capacity(
                estimated_items,
                capacity_per_shard,
            )),
            max_memory_bytes,
            current_memory_bytes: Arc::new(RwLock::new(0)),
        }
    }

    async fn get(&self, key: &str) -> Option<(ResponseHeader, Bytes)> {
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(key) {
            if entry.is_expired() {
                drop(entries);
                self.remove(key).await;
                return None;
            }

            // Update access time in eviction manager
            let cache_key = CacheKey::new("embed-proxy", key, "");
            let compact_key = cache_key.to_compact();
            self.eviction_manager
                .access(&compact_key, 1, SystemTime::now());

            Some((entry.headers.clone(), entry.data.clone()))
        } else {
            None
        }
    }

    async fn put(
        &self,
        key: String,
        headers: ResponseHeader,
        data: Bytes,
        ttl: Duration,
    ) -> Result<()> {
        let entry_size = data.len() + key.len() + 256; // Rough estimate including headers

        // Check if this item would exceed memory limits
        if entry_size > self.max_memory_bytes {
            return Ok(()); // Don't cache items larger than total cache size
        }

        // Evict items if needed to make space
        self.ensure_space(entry_size).await;

        let entry = CacheEntry {
            data,
            headers,
            created_at: SystemTime::now(),
            ttl,
        };

        // Add to eviction manager
        let cache_key = CacheKey::new("embed-proxy", &*key, "");
        let compact_key = cache_key.to_compact();
        self.eviction_manager
            .admit(compact_key, entry_size, SystemTime::now());

        // Add to cache
        let mut entries = self.entries.write().await;
        entries.insert(key, entry);

        // Update memory usage
        let mut current_memory = self.current_memory_bytes.write().await;
        *current_memory += entry_size;

        Ok(())
    }

    async fn remove(&self, key: &str) {
        let mut entries = self.entries.write().await;
        if let Some(entry) = entries.remove(key) {
            let entry_size = entry.data.len() + key.len() + 256;

            // Remove from eviction manager
            let cache_key = CacheKey::new("embed-proxy", key, "");
            let compact_key = cache_key.to_compact();
            self.eviction_manager.remove(&compact_key);

            // Update memory usage
            let mut current_memory = self.current_memory_bytes.write().await;
            *current_memory = current_memory.saturating_sub(entry_size);
        }
    }

    async fn ensure_space(&self, needed_space: usize) {
        let current_memory = *self.current_memory_bytes.read().await;

        if current_memory + needed_space <= self.max_memory_bytes {
            return; // Enough space available
        }

        // Need to evict items
        let target_memory = (self.max_memory_bytes * 80) / 100; // Target 80% usage after eviction
        let mut to_evict = Vec::new();

        // Get entries to evict from LRU manager
        // Note: In a real implementation, you'd need to integrate more tightly with
        // the eviction manager's eviction algorithm. For now, we'll use a simple approach.
        {
            let entries = self.entries.read().await;
            let mut total_size = current_memory;

            // Find oldest entries by creation time (simple LRU approximation)
            let mut entries_by_age: Vec<_> = entries.iter().collect();
            entries_by_age.sort_by_key(|(_, entry)| entry.created_at);

            for (key, entry) in entries_by_age {
                if total_size <= target_memory {
                    break;
                }

                let entry_size = entry.data.len() + key.len() + 256;
                to_evict.push(key.clone());
                total_size = total_size.saturating_sub(entry_size);
            }
        }

        // Remove evicted entries
        for key in to_evict {
            self.remove(&key).await;
        }
    }
}

pub struct EmbedProxy {
    config: Arc<Config>,
    rate_limiter: Arc<RateLimiter>,
    tracking_headers: Arc<HashSet<String>>,
    frame_headers: Arc<HashSet<String>>,
    csp_regex: Arc<Regex>,
    cache: Option<Arc<InMemoryCache>>,
}

impl EmbedProxy {
    pub fn new(config: Config) -> Self {
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limit_per_minute));

        // Headers that commonly interfere with embedding
        let frame_headers = Arc::new(
            [
                "x-frame-options",
                "x-frame-options-allowall",
                "frame-options",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Common tracking headers that trigger protection
        let tracking_headers = Arc::new(
            [
                "x-fb-debug",
                "x-fb-rev",
                "x-served-by",
                "x-cache",
                "x-cache-hits",
                "x-timer",
                "x-varnish",
                "x-fastly-request-id",
                "x-github-request-id",
                "x-request-id",
                "x-correlation-id",
                "x-trace-id",
                "server-timing",
                "x-runtime",
                "x-powered-by",
                "x-aspnet-version",
                "x-aspnetmvc-version",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Regex to find and remove frame-ancestors from CSP
        let csp_regex = Arc::new(Regex::new(r"frame-ancestors[^;]*;?").unwrap());

        // Initialize cache if enabled
        let cache = if config.cache.enabled {
            let cache = Arc::new(InMemoryCache::new(config.cache.max_memory_mb));
            info!(
                "Initialized cache with {}MB memory limit",
                config.cache.max_memory_mb
            );
            Some(cache)
        } else {
            info!("Cache disabled in configuration");
            None
        };

        Self {
            config: Arc::new(config),
            rate_limiter,
            tracking_headers,
            frame_headers,
            csp_regex,
            cache,
        }
    }

    fn generate_cache_key(&self, url: &Url, request_headers: &RequestHeader) -> String {
        let mut hasher = Sha256::new();

        // Include URL in cache key
        hasher.update(url.as_str().as_bytes());

        // Include relevant headers that might affect response
        if let Some(accept) = request_headers.headers.get("accept") {
            if let Ok(accept_str) = accept.to_str() {
                hasher.update(b"accept:");
                hasher.update(accept_str.as_bytes());
            }
        }

        if let Some(accept_encoding) = request_headers.headers.get("accept-encoding") {
            if let Ok(encoding_str) = accept_encoding.to_str() {
                hasher.update(b"accept-encoding:");
                hasher.update(encoding_str.as_bytes());
            }
        }

        let hash = hasher.finalize();
        hex::encode(hash)
    }

    fn should_cache_response(&self, url: &Url, response: &ResponseHeader) -> bool {
        if !self.config.cache.enabled {
            return false;
        }

        // Check if domain is in no-cache list
        if let Some(domain) = url.host_str() {
            if self
                .config
                .cache
                .no_cache_domains
                .iter()
                .any(|d| domain.contains(d))
            {
                return false;
            }
        }

        // Check status code
        if !self
            .config
            .cache
            .cacheable_status_codes
            .contains(&response.status.as_u16())
        {
            return false;
        }

        // Check content type if configured
        if !self.config.cache.cacheable_content_types.is_empty() {
            if let Some(content_type) = response.headers.get("content-type") {
                if let Ok(ct_str) = content_type.to_str() {
                    let ct_main = ct_str.split(';').next().unwrap_or("").trim();
                    if !self
                        .config
                        .cache
                        .cacheable_content_types
                        .contains(&ct_main.to_string())
                    {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check content length if available
        if let Some(content_length) = response.headers.get("content-length") {
            if let Ok(length_str) = content_length.to_str() {
                if let Ok(length) = length_str.parse::<usize>() {
                    let max_size = self.config.cache.max_item_size_mb * 1024 * 1024;
                    if length > max_size {
                        debug!(
                            "Response too large to cache: {} bytes > {} bytes",
                            length, max_size
                        );
                        return false;
                    }
                }
            }
        }

        // Don't cache responses with no-cache directives
        if let Some(cache_control) = response.headers.get("cache-control") {
            if let Ok(cc_str) = cache_control.to_str() {
                if cc_str.contains("no-cache")
                    || cc_str.contains("no-store")
                    || cc_str.contains("private")
                {
                    return false;
                }
            }
        }

        true
    }

    fn get_cache_ttl(&self, response: &ResponseHeader) -> Duration {
        // Try to extract TTL from Cache-Control header
        if let Some(cache_control) = response.headers.get("cache-control") {
            if let Ok(cc_str) = cache_control.to_str() {
                for directive in cc_str.split(',') {
                    let directive = directive.trim();
                    if directive.starts_with("max-age=") {
                        if let Ok(seconds) = directive[8..].parse::<u64>() {
                            return Duration::from_secs(seconds);
                        }
                    }
                }
            }
        }

        // Try to extract TTL from Expires header
        if let Some(expires) = response.headers.get("expires") {
            if let Ok(_expires_str) = expires.to_str() {
                // This is a simplified parsing - in production you might want to use a proper HTTP date parser
                // For now, just use default TTL
            }
        }

        // Use default TTL
        Duration::from_secs(self.config.cache.default_ttl_seconds)
    }

    fn is_domain_allowed(&self, domain: &str) -> bool {
        if self.config.blocked_domains.contains(&domain.to_string()) {
            return false;
        }

        if self.config.allowed_domains.is_empty() {
            return true;
        }

        self.config
            .allowed_domains
            .iter()
            .any(|allowed| domain == allowed || domain.ends_with(&format!(".{}", allowed)))
    }

    fn extract_target_url(&self, uri: &Uri) -> Result<Url> {
        let path = uri.path();

        // Expected format: /proxy/https://example.com/path
        if !path.starts_with("/proxy/") {
            return Err(anyhow::anyhow!("Invalid proxy path format"));
        }

        let url_part = &path[7..]; // Remove "/proxy/"
        let query = uri.query().unwrap_or("");

        let full_url = if query.is_empty() {
            url_part.to_string()
        } else {
            format!("{}?{}", url_part, query)
        };

        Url::parse(&full_url).context("Failed to parse target URL")
    }

    fn modify_response_headers(&self, headers: &mut ResponseHeader) {
        // Remove frame-related headers
        for header in self.frame_headers.iter() {
            headers.remove_header(header);
        }

        // Handle Content-Security-Policy - fix borrow checker issue
        let csp_header_value = headers
            .headers
            .get("content-security-policy")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        if let Some(csp_str) = csp_header_value {
            let replaced = self.csp_regex.replace_all(&csp_str, "");
            let modified_csp = replaced.trim();

            headers.remove_header("content-security-policy");

            if !modified_csp.is_empty() {
                if let Ok(new_value) = HeaderValue::from_str(modified_csp) {
                    headers
                        .append_header("content-security-policy", new_value)
                        .ok();
                }
            }
        }

        // Remove tracking headers if configured
        if self.config.strip_tracking_headers {
            for header in self.tracking_headers.iter() {
                headers.remove_header(header);
            }
        }

        // Add CORS headers for embedding
        headers
            .insert_header("Access-Control-Allow-Origin", "*")
            .ok();
        headers
            .insert_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            .ok();
        headers
            .insert_header("Access-Control-Allow-Headers", "*")
            .ok();

        // Add custom headers
        for (name, value) in &self.config.custom_headers {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                headers.insert_header(header_name, header_value).ok();
            }
        }

        // Add cache headers for better performance
        headers
            .insert_header("Cache-Control", "public, max-age=3600")
            .ok();
    }
}

#[derive(Debug)]
pub struct ProxyContext {
    cache_key: Option<String>,
    target_url: Option<Url>,
    cache_hit: bool,
    response_body: Vec<Bytes>,
}

#[async_trait]
impl ProxyHttp for EmbedProxy {
    type CTX = ProxyContext;

    fn new_ctx(&self) -> Self::CTX {
        ProxyContext {
            cache_key: None,
            target_url: None,
            cache_hit: false,
            response_body: Vec::new(),
        }
    }

    async fn request_filter(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<bool, Box<pingora::Error>> {
        // Rate limiting
        let client_ip = session
            .client_addr()
            .map(|addr| addr.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        if !self.rate_limiter.check_rate_limit(&client_ip).await {
            let mut response =
                ResponseHeader::build(StatusCode::TOO_MANY_REQUESTS, None).map_err(Box::from)?;
            response
                .insert_header("Retry-After", "60")
                .map_err(Box::from)?;
            session.set_keepalive(None);
            session
                .write_response_header(Box::new(response), true)
                .await
                .map_err(Box::from)?;
            return Ok(true); // Request handled
        }

        // Handle OPTIONS requests for CORS
        if session.req_header().method == Method::OPTIONS {
            let mut response = ResponseHeader::build(StatusCode::OK, None).map_err(Box::from)?;
            response
                .insert_header("Access-Control-Allow-Origin", "*")
                .map_err(Box::from)?;
            response
                .insert_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                .map_err(Box::from)?;
            response
                .insert_header("Access-Control-Allow-Headers", "*")
                .map_err(Box::from)?;
            response
                .insert_header("Access-Control-Max-Age", "86400")
                .map_err(Box::from)?;
            session
                .write_response_header(Box::new(response), true)
                .await
                .map_err(Box::from)?;
            return Ok(true); // Request handled
        }

        // Extract target URL for caching
        if let Ok(target_url) = self.extract_target_url(&session.req_header().uri) {
            ctx.target_url = Some(target_url.clone());

            // Check cache if enabled and this is a GET request
            if self.cache.is_some() && session.req_header().method == Method::GET {
                let cache_key = self.generate_cache_key(&target_url, session.req_header());
                ctx.cache_key = Some(cache_key.clone());

                if let Some(cache) = &self.cache {
                    // Try to get from cache
                    if let Some((mut headers, body)) = cache.get(&cache_key).await {
                        debug!("Cache HIT for {}", target_url);
                        ctx.cache_hit = true;

                        // Apply embedding modifications to cached headers
                        self.modify_response_headers(&mut headers);
                        headers.remove_header("X-Cache-Status");
                        headers
                            .insert_header("X-Cache-Status", "HIT")
                            .map_err(Box::from)?;

                        // Send cached response
                        session
                            .write_response_header(Box::new(headers), false)
                            .await
                            .map_err(Box::from)?;
                        session
                            .write_response_body(Some(body), true)
                            .await
                            .map_err(Box::from)?;

                        return Ok(true); // Request handled from cache
                    } else {
                        debug!("Cache MISS for {}", target_url);
                    }
                }
            }
        }

        Ok(false) // Continue processing
    }

    async fn upstream_peer(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>, Box<pingora::Error>> {
        let target_url = if let Some(url) = &ctx.target_url {
            url.clone()
        } else {
            self.extract_target_url(&session.req_header().uri)
                .map_err(|_| Box::new(pingora::Error::new_str("Failed to extract target URL")))
                .expect("Failed toe extract target URL")
        };

        // Check if domain is allowed
        if let Some(domain) = target_url.host_str() {
            if !self.is_domain_allowed(domain) {
                return Err(Box::new(*pingora::Error::new_str("Domain not allowed")));
            }
        }

        let port = target_url
            .port()
            .unwrap_or(if target_url.scheme() == "https" {
                443
            } else {
                80
            });

        let peer = Box::new(HttpPeer::new(
            format!("{}:{}", target_url.host_str().unwrap_or(""), port),
            target_url.scheme() == "https",
            target_url.host_str().unwrap_or("").to_string(),
        ));

        Ok(peer)
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> Result<(), Box<pingora::Error>> {
        let target_url = if let Some(url) = &ctx.target_url {
            url.clone()
        } else {
            self.extract_target_url(&upstream_request.uri)
                .map_err(|_| pingora::Error::new_str("Failed to extract target URL"))?
        };

        // Reconstruct the path and query for the upstream request
        let path = target_url.path();
        let query = target_url.query().unwrap_or("");
        let new_uri = if query.is_empty() {
            path.to_string()
        } else {
            format!("{}?{}", path, query)
        };

        // Parse and set the new URI
        let parsed_uri: Uri = new_uri
            .parse()
            .map_err(|_| pingora::Error::new_str("Failed to parse URI"))?;
        upstream_request.set_uri(parsed_uri);

        // Set proper headers
        upstream_request
            .insert_header("User-Agent", &self.config.user_agent)
            .map_err(Box::from)?;
        upstream_request
            .insert_header("Accept", "*/*")
            .map_err(Box::from)?;
        upstream_request
            .insert_header("Accept-Encoding", "gzip, deflate")
            .map_err(Box::from)?;

        // Remove proxy-specific headers
        upstream_request.remove_header("x-forwarded-for");
        upstream_request.remove_header("x-forwarded-proto");
        upstream_request.remove_header("x-real-ip");

        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        _ctx: &mut Self::CTX,
    ) -> Result<(), Box<pingora::Error>> {
        self.modify_response_headers(upstream_response);
        Ok(())
    }

    fn response_body_filter(
        &self,
        session: &mut Session,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<Option<Duration>, Box<pingora::Error>>
    where
        Self::CTX: Send + Sync,
    {
        // Collect response body for caching
        if let Some(body_bytes) = body {
            ctx.response_body.push(body_bytes.clone());
        }

        // Note: Caching logic would need to be moved to an async context
        // For now, we'll just return Ok(None) to indicate no delay
        Ok(None)
    }

    async fn logging(
        &self,
        session: &mut Session,
        _e: Option<&pingora::Error>,
        ctx: &mut Self::CTX,
    ) {
        let status = session
            .response_written()
            .map(|resp| resp.status.as_u16())
            .unwrap_or(0);

        let client_ip = session
            .client_addr()
            .map(|addr| addr.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let cache_status = if ctx.cache_hit { "HIT" } else { "MISS" };

        // Perform caching if needed
        if !ctx.cache_hit && !ctx.response_body.is_empty() {
            if let (Some(cache), Some(cache_key), Some(target_url)) =
                (&self.cache, &ctx.cache_key, &ctx.target_url)
            {
                if let Some(response_header) = session.response_written() {
                    if self.should_cache_response(target_url, response_header) {
                        // Combine all body chunks
                        let full_body = if ctx.response_body.len() == 1 {
                            ctx.response_body[0].clone()
                        } else {
                            let total_len: usize = ctx.response_body.iter().map(|b| b.len()).sum();
                            let mut combined = Vec::with_capacity(total_len);
                            for chunk in &ctx.response_body {
                                combined.extend_from_slice(chunk);
                            }
                            Bytes::from(combined)
                        };

                        let ttl = self.get_cache_ttl(response_header);

                        // Store in cache
                        match cache
                            .put(cache_key.clone(), response_header.clone(), full_body, ttl)
                            .await
                        {
                            Ok(_) => {
                                debug!("Cached response for {} (TTL: {:?})", target_url, ttl);
                            }
                            Err(e) => {
                                warn!("Failed to cache response for {}: {}", target_url, e);
                            }
                        }
                    }
                }
            }
        }

        info!(
            "client_ip={} method={} uri={} status={} cache={}",
            client_ip,
            session.req_header().method,
            session.req_header().uri,
            status,
            cache_status
        );
    }
}

async fn load_config(path: &str) -> Result<Config> {
    match tokio::fs::read_to_string(path).await {
        Ok(content) => serde_json::from_str(&content).context("Failed to parse config file"),
        Err(_) => {
            warn!("Config file not found, creating default config at {}", path);
            let default_config = Config::default();
            let content = serde_json::to_string_pretty(&default_config)?;
            tokio::fs::write(path, content)
                .await
                .context("Failed to write default config")?;
            Ok(default_config)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("embed_proxy={},pingora={}", log_level, log_level))
        .init();

    info!("Starting Embed Proxy Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = load_config(&args.config).await?;
    info!("Loaded configuration from {}", args.config);

    // Create server
    let mut server = Server::new(None)?;
    server.bootstrap();

    // Create proxy service
    let proxy = EmbedProxy::new(config);
    let mut proxy_service = pingora_proxy::http_proxy_service(&server.configuration, proxy);

    // Configure listening address
    let addr: SocketAddr = args.listen.parse().context("Invalid listen address")?;
    proxy_service.add_tcp(&addr.to_string());

    info!("Proxy server listening on {}", addr);
    info!("Usage: GET http://{}/proxy/https://example.com/path", addr);

    // Add service to server
    server.add_service(proxy_service);

    // Run server
    server.run_forever();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_allowed() {
        let config = Config {
            allowed_domains: vec!["example.com".to_string(), "test.org".to_string()],
            blocked_domains: vec!["blocked.com".to_string()],
            ..Default::default()
        };

        let proxy = EmbedProxy::new(config);

        assert!(proxy.is_domain_allowed("example.com"));
        assert!(proxy.is_domain_allowed("sub.example.com"));
        assert!(proxy.is_domain_allowed("test.org"));
        assert!(!proxy.is_domain_allowed("blocked.com"));
        assert!(!proxy.is_domain_allowed("other.com"));
    }

    #[test]
    fn test_extract_target_url() {
        let config = Config::default();
        let proxy = EmbedProxy::new(config);

        let uri: Uri = "/proxy/https://example.com/path?param=value"
            .parse()
            .unwrap();
        let url = proxy.extract_target_url(&uri).unwrap();

        assert_eq!(url.as_str(), "https://example.com/path?param=value");
    }
}
