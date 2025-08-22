use anyhow::{Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use clap::Parser;
use hex;
use http::{header::HeaderName, HeaderValue, Method, StatusCode, Uri};
use pingora::prelude::*;
use pingora::protocols::TcpKeepalive;
use pingora_cache::cache_control::CacheControl;
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_memory_cache::{CacheStatus, MemoryCache};
use pingora_proxy::{ProxyHttp, Session};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Once;
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr as StdSocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
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

    /// Maximum number of items in cache
    max_items: usize,

    /// Default TTL for cached responses in seconds
    default_ttl_seconds: u64,

    /// Maximum TTL for cached responses in seconds
    max_ttl_seconds: u64,

    /// Maximum size of individual cached items in MB
    max_item_size_mb: usize,

    /// Cache only responses with these status codes
    cacheable_status_codes: Vec<u16>,

    /// Cache responses with these content types (empty = cache all)
    cacheable_content_types: Vec<String>,

    /// Don't cache responses from these domains
    no_cache_domains: Vec<String>,

    /// Enable cache compression
    enable_compression: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_items: 10000,
            default_ttl_seconds: 3600, // 1 hour
            max_ttl_seconds: 86400,    // 24 hours
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
            enable_compression: true,
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

    /// Connection timeout in seconds (separate from request timeout)
    connection_timeout: u64,

    /// Read timeout in seconds
    read_timeout: u64,

    /// Write timeout in seconds  
    write_timeout: u64,

    /// Cache configuration
    cache: CacheConfig,

    /// Enable request/response metrics
    enable_metrics: bool,
}

impl Default for Config {
    fn default() -> Self {
        let custom_headers = HashMap::new();

        Self {
            allowed_domains: vec![],
            blocked_domains: vec![
                "malware.example.com".to_string(),
                "phishing.example.com".to_string(),
            ],
            rate_limit_per_minute: 60,
            custom_headers,
            strip_tracking_headers: true,
            user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36".to_string(),
            upstream_timeout: 30,
            connection_timeout: 10,
            read_timeout: 30,
            write_timeout: 30,
            cache: CacheConfig::default(),
            enable_metrics: true,
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

    async fn cleanup_old_entries(&self) {
        let mut requests = self.requests.write().await;
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        requests.retain(|_, times| {
            times.retain(|&time| time > minute_ago);
            !times.is_empty()
        });
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Bytes,
    headers: ResponseHeader,
    created_at: SystemTime,
    ttl: Duration,
    size: usize,
}

impl CacheEntry {
    fn new(data: Bytes, headers: ResponseHeader, ttl: Duration) -> Self {
        let size = data.len()
            + headers
                .headers
                .iter()
                .map(|(k, v)| k.as_str().len() + v.len())
                .sum::<usize>()
            + 128; // Rough estimate for metadata

        Self {
            data,
            headers,
            created_at: SystemTime::now(),
            ttl,
            size,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at
            .elapsed()
            .map(|elapsed| elapsed > self.ttl)
            .unwrap_or(true)
    }
}

#[derive(Debug)]
struct CacheMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    entries: AtomicU64,
}

#[allow(dead_code)]
impl CacheMetrics {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            entries: AtomicU64::new(0),
        }
    }

    fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    fn add_entry(&self) {
        self.entries.fetch_add(1, Ordering::Relaxed);
    }

    fn remove_entry(&self) {
        self.entries.fetch_sub(1, Ordering::Relaxed);
    }

    fn get_metrics(&self) -> (u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.entries.load(Ordering::Relaxed),
        )
    }
}

pub struct EmbedProxy {
    config: Arc<Config>,
    rate_limiter: Arc<RateLimiter>,
    tracking_headers: Arc<HashSet<String>>,
    frame_headers: Arc<HashSet<String>>,
    csp_regex: Arc<Regex>,
    cache: Option<Arc<MemoryCache<String, CacheEntry>>>,
    cache_metrics: Arc<CacheMetrics>,
    background_tasks_started: Once,
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
        let csp_regex = Arc::new(
            Regex::new(r"frame-ancestors[^;]*;?\s*").expect("Failed to compile CSP regex"),
        );

        // Initialize cache if enabled using pingora-memory-cache
        let cache = if config.cache.enabled {
            let cache = Arc::new(MemoryCache::new(config.cache.max_items));
            info!(
                "Initialized cache with {} max items",
                config.cache.max_items
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
            cache_metrics: Arc::new(CacheMetrics::new()),
            background_tasks_started: Once::new(),
        }
    }

    fn generate_cache_key(&self, url: &Url, request_headers: &RequestHeader) -> String {
        let mut hasher = Sha256::new();

        // Include URL in cache key
        hasher.update(url.as_str().as_bytes());

        // Include relevant headers that might affect response
        let relevant_headers = ["accept", "accept-encoding", "accept-language"];
        for header_name in &relevant_headers {
            if let Some(header_value) = request_headers.headers.get(*header_name) {
                if let Ok(header_str) = header_value.to_str() {
                    hasher.update(header_name.as_bytes());
                    hasher.update(b":");
                    hasher.update(header_str.as_bytes());
                }
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
                    let ct_main = ct_str.split(';').next().unwrap_or("").trim().to_lowercase();
                    if !self
                        .config
                        .cache
                        .cacheable_content_types
                        .iter()
                        .any(|allowed| ct_main.starts_with(&allowed.to_lowercase()))
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
                let cc_lower = cc_str.to_lowercase();
                if cc_lower.contains("no-cache")
                    || cc_lower.contains("no-store")
                    || cc_lower.contains("private")
                {
                    return false;
                }
            }
        }

        true
    }

    fn get_cache_ttl(&self, response: &ResponseHeader) -> Duration {
        // Try to extract TTL from Cache-Control header using pingora's parser
        let cc = CacheControl::from_resp_headers(response).unwrap();
        if let Some(max_age) = cc.max_age().expect("Could not get max age") {
            // Convert u32 to u64 for comparison and Duration::from_secs
            let max_age_u64 = u64::from(max_age);
            let ttl = Duration::from_secs(max_age_u64.min(self.config.cache.max_ttl_seconds));
            debug!("Using max-age TTL: {:?}", ttl);
            return ttl;
        }
        // Try to parse Expires header
        if let Some(expires) = response.headers.get("expires") {
            if let Ok(expires_str) = expires.to_str() {
                // Try to parse HTTP date format
                if let Ok(expires_time) = DateTime::parse_from_rfc2822(expires_str)
                    .or_else(|_| DateTime::parse_from_rfc3339(expires_str))
                {
                    let now = Utc::now();
                    if let Ok(duration) = (expires_time.with_timezone(&Utc) - now).to_std() {
                        let ttl =
                            duration.min(Duration::from_secs(self.config.cache.max_ttl_seconds));
                        debug!("Using Expires TTL: {:?}", ttl);
                        return ttl;
                    }
                }
            }
        }
        // Use default TTL
        let default_ttl = Duration::from_secs(self.config.cache.default_ttl_seconds);
        debug!("Using default TTL: {:?}", default_ttl);
        default_ttl
    }

    fn is_domain_allowed(&self, domain: &str) -> bool {
        if self
            .config
            .blocked_domains
            .iter()
            .any(|blocked| domain == blocked || domain.ends_with(&format!(".{}", blocked)))
        {
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
            return Err(anyhow::anyhow!(
                "Invalid proxy path format. Expected: /proxy/https://example.com/path"
            ));
        }

        let url_part = &path[7..]; // Remove "/proxy/"

        if url_part.is_empty() {
            return Err(anyhow::anyhow!("No target URL provided"));
        }

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

        // Handle Content-Security-Policy
        if let Some(csp_value) = headers
            .headers
            .get("content-security-policy")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
        {
            let modified_csp = self.csp_regex.replace_all(&csp_value, "");
            let cleaned_csp = modified_csp.trim();

            headers.remove_header("content-security-policy");

            if !cleaned_csp.is_empty() {
                if let Ok(new_value) = HeaderValue::from_str(cleaned_csp) {
                    let _ = headers.append_header("content-security-policy", new_value);
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
        let _ = headers.insert_header("Access-Control-Allow-Origin", "*");
        let _ = headers.insert_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        let _ = headers.insert_header("Access-Control-Allow-Headers", "*");

        // Add custom headers
        for (name, value) in &self.config.custom_headers {
            if let (Ok(header_name), Ok(header_value)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                let _ = headers.insert_header(header_name, header_value);
            }
        }

        // Add cache headers for better performance
        let _ = headers.insert_header("Cache-Control", "public, max-age=3600");
    }

    async fn lookup_cache(&self, cache_key: &str) -> Option<(ResponseHeader, Bytes)> {
        if let Some(cache) = &self.cache {
            debug!("Looking up cache key: {}", cache_key);
            let (entry_opt, status) = cache.get(cache_key);

            debug!("Cache lookup status: {:?}", status);

            match status {
                CacheStatus::Hit => {
                    if let Some(entry) = entry_opt {
                        if !entry.is_expired() {
                            self.cache_metrics.hit();
                            info!("Cache HIT for key: {}", cache_key);
                            return Some((entry.headers.clone(), entry.data.clone()));
                        } else {
                            // Entry expired, remove it
                            cache.remove(cache_key);
                            info!("Cache entry expired for key: {}", cache_key);
                            self.cache_metrics.miss();
                        }
                    } else {
                        warn!(
                            "Cache status was Hit but no entry found for key: {}",
                            cache_key
                        );
                        self.cache_metrics.miss();
                    }
                }
                CacheStatus::Expired => {
                    info!("Cache entry expired for key: {}", cache_key);
                    self.cache_metrics.miss();
                }
                CacheStatus::Miss => {
                    debug!("Cache MISS for key: {}", cache_key);
                    self.cache_metrics.miss();
                }
                CacheStatus::Stale(_duration) => {
                    info!("Cache entry stale for key: {}", cache_key);
                    self.cache_metrics.miss();
                }
                CacheStatus::LockHit => {
                    debug!("Lock hit for {}", cache_key);
                    self.cache_metrics.miss();
                }
            }
        } else {
            debug!("No cache configured");
        }

        None
    }

    async fn store_cache(
        &self,
        cache_key: String,
        response: &ResponseHeader,
        body: Bytes,
        ttl: Duration,
    ) -> Result<()> {
        if let Some(cache) = &self.cache {
            let entry = CacheEntry::new(body, response.clone(), ttl);

            // Check size limits
            if entry.size > self.config.cache.max_item_size_mb * 1024 * 1024 {
                debug!("Entry too large to cache: {} bytes", entry.size);
                return Ok(());
            }

            // Store with TTL - pingora-memory-cache handles eviction automatically
            cache.put(&cache_key, entry, Some(ttl));
            self.cache_metrics.add_entry();
            debug!("Successfully cached response with TTL: {:?}", ttl);
        }
        Ok(())
    }

    fn ensure_background_tasks_started(&self) {
        self.background_tasks_started.call_once(|| {
            let rate_limiter = self.rate_limiter.clone();
            let cache_metrics = self.cache_metrics.clone();
            let cache = self.cache.clone();

            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
                loop {
                    interval.tick().await;
                    rate_limiter.cleanup_old_entries().await;

                    // Log cache stats
                    let (hits, misses, evictions, entries) = cache_metrics.get_metrics();
                    let hit_rate = if hits + misses > 0 {
                        (hits as f64 / (hits + misses) as f64) * 100.0
                    } else {
                        0.0
                    };

                    if cache.is_some() {
                        info!(
                            "Cache stats: entries={}, hits={}, misses={}, evictions={}, hit_rate={:.2}%",
                            entries, hits, misses, evictions, hit_rate
                        );
                    }
                }
            });

            info!("Background tasks started");
        });
    }
}

#[derive(Debug)]
pub struct ProxyContext {
    cache_key: Option<String>,
    target_url: Option<Url>,
    cache_hit: bool,
    response_body: Vec<Bytes>,
    start_time: Instant,
}

#[async_trait]
impl ProxyHttp for EmbedProxy {
    type CTX = ProxyContext;

    fn new_ctx(&self) -> Self::CTX {
        self.ensure_background_tasks_started();

        ProxyContext {
            cache_key: None,
            target_url: None,
            cache_hit: false,
            response_body: Vec::new(),
            start_time: Instant::now(),
        }
    }

    async fn request_filter(
        &self,
        session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<bool, Box<pingora::Error>> {
        ctx.start_time = Instant::now();

        // Get client IP for rate limiting
        let client_ip = session
            .client_addr()
            .map(|addr| addr.to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Rate limiting
        if !self.rate_limiter.check_rate_limit(&client_ip).await {
            warn!("Rate limit exceeded for client: {}", client_ip);
            let mut response =
                ResponseHeader::build(StatusCode::TOO_MANY_REQUESTS, None).map_err(Box::from)?;
            let _ = response.insert_header("Retry-After", "60");
            session.set_keepalive(None);
            session
                .write_response_header(Box::new(response), true)
                .await
                .map_err(Box::from)?;
            return Ok(true);
        }

        // Handle OPTIONS requests for CORS
        if session.req_header().method == Method::OPTIONS {
            let mut response = ResponseHeader::build(StatusCode::OK, None).map_err(Box::from)?;
            let _ = response.insert_header("Access-Control-Allow-Origin", "*");
            let _ = response.insert_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            let _ = response.insert_header("Access-Control-Allow-Headers", "*");
            let _ = response.insert_header("Access-Control-Max-Age", "86400");

            session
                .write_response_header(Box::new(response), true)
                .await
                .map_err(Box::from)?;
            return Ok(true);
        }

        // Extract and validate target URL
        match self.extract_target_url(&session.req_header().uri) {
            Ok(target_url) => {
                debug!("Extracted target URL: {}", target_url);

                // Check if domain is allowed
                if let Some(domain) = target_url.host_str() {
                    if !self.is_domain_allowed(domain) {
                        warn!("Domain not allowed: {}", domain);
                        let mut response = ResponseHeader::build(StatusCode::FORBIDDEN, None)
                            .map_err(Box::from)?;
                        let _ = response.insert_header("Content-Type", "text/plain");

                        session
                            .write_response_header(Box::new(response), false)
                            .await
                            .map_err(Box::from)?;
                        session
                            .write_response_body(Some(Bytes::from("Domain not allowed")), true)
                            .await
                            .map_err(Box::from)?;
                        return Ok(true);
                    }
                }

                ctx.target_url = Some(target_url.clone());

                // CRITICAL: Only check cache for GET requests
                if session.req_header().method == Method::GET && self.cache.is_some() {
                    let cache_key = self.generate_cache_key(&target_url, session.req_header());
                    debug!("Generated cache key: {}", cache_key);
                    ctx.cache_key = Some(cache_key.clone());

                    // Try cache lookup
                    if let Some((mut headers, body)) = self.lookup_cache(&cache_key).await {
                        ctx.cache_hit = true;
                        info!("Cache HIT for {}", target_url);

                        // Get length before moving `body`
                        let body_len = body.len();

                        // Modify headers before sending
                        self.modify_response_headers(&mut headers);
                        let _ = headers.insert_header("X-Cache-Status", "HIT");

                        session
                            .write_response_header(Box::new(headers), false)
                            .await
                            .map_err(Box::from)?;
                        session
                            .write_response_body(Some(body), true)
                            .await
                            .map_err(Box::from)?;

                        info!("Served from cache: {} bytes", body_len);
                        return Ok(true);
                    } else {
                        info!("Cache MISS for {}", target_url);
                    }
                } else {
                    debug!(
                        "Skipping cache: method={}, cache_enabled={}",
                        session.req_header().method,
                        self.cache.is_some()
                    );
                }
            }
            Err(e) => {
                warn!("Failed to extract target URL: {}", e);
                let mut response =
                    ResponseHeader::build(StatusCode::BAD_REQUEST, None).map_err(Box::from)?;
                let _ = response.insert_header("Content-Type", "text/plain");

                session
                    .write_response_header(Box::new(response), false)
                    .await
                    .map_err(Box::from)?;
                session
                    .write_response_body(Some(Bytes::from(format!("Invalid request: {}", e))), true)
                    .await
                    .map_err(Box::from)?;
                return Ok(true);
            }
        }

        Ok(false) // Continue processing
    }

    async fn upstream_peer(
        &self,
        _session: &mut Session,
        ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>, Box<pingora::Error>> {
        let target_url = ctx
            .target_url
            .as_ref()
            .ok_or_else(|| Box::new(*pingora::Error::new_str("No target URL available")))?;

        let port = target_url
            .port()
            .unwrap_or(if target_url.scheme() == "https" {
                443
            } else {
                80
            });
        let host = target_url
            .host_str()
            .ok_or_else(|| Box::new(*pingora::Error::new_str("No host in target URL")))?;

        let mut peer = HttpPeer::new(
            format!("{}:{}", host, port),
            target_url.scheme() == "https",
            host.to_string(),
        );

        // CRITICAL: Configure timeouts and connection options
        peer.options.connection_timeout = Some(Duration::from_secs(self.config.connection_timeout));
        peer.options.read_timeout = Some(Duration::from_secs(self.config.read_timeout));
        peer.options.write_timeout = Some(Duration::from_secs(self.config.write_timeout));

        // Configure TLS options for better compatibility
        if target_url.scheme() == "https" {
            peer.options.verify_cert = true;
            peer.options.verify_hostname = true;
        }

        // Enable connection reuse for better performance
        peer.options.tcp_keepalive = Some(TcpKeepalive {
            idle: Duration::from_secs(60), // Start probing after 60s of inactivity
            interval: Duration::from_secs(10), // Probe every 10 seconds
            count: 6,                      // Send up to 6 probes before giving up
            #[cfg(target_os = "linux")]
            user_timeout: Duration::from_secs(180), // Total timeout for unacknowledged data
        });

        debug!(
            "Created peer with connection_timeout: {}s, read_timeout: {}s, write_timeout: {}s",
            self.config.connection_timeout, self.config.read_timeout, self.config.write_timeout
        );

        Ok(Box::new(peer))
    }

    async fn upstream_request_filter(
        &self,
        _session: &mut Session,
        upstream_request: &mut RequestHeader,
        ctx: &mut Self::CTX,
    ) -> Result<(), Box<pingora::Error>> {
        let target_url = ctx
            .target_url
            .as_ref()
            .ok_or_else(|| pingora::Error::new_str("No target URL available"))?;

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
            .map_err(|_| pingora::Error::new_str("Failed to parse upstream URI"))?;
        upstream_request.set_uri(parsed_uri);

        // Set the Host header to the target domain (this is crucial)
        if let Some(host) = target_url.host_str() {
            let _ = upstream_request.insert_header("Host", host);
        }

        // CRITICAL: Set the configured User-Agent instead of keeping curl's
        let _ = upstream_request.insert_header("User-Agent", &self.config.user_agent);

        // Add Accept-Encoding for better compatibility
        if upstream_request.headers.get("accept-encoding").is_none() {
            let _ = upstream_request.insert_header("Accept-Encoding", "gzip, deflate, br");
        }

        // Ensure we have an Accept header
        if upstream_request.headers.get("accept").is_none() {
            let _ = upstream_request.insert_header("Accept", "*/*");
        }

        // Remove problematic proxy headers
        upstream_request.remove_header("x-forwarded-for");
        upstream_request.remove_header("x-forwarded-proto");
        upstream_request.remove_header("x-real-ip");
        upstream_request.remove_header("x-proxy-by");
        upstream_request.remove_header("via");
        upstream_request.remove_header("forwarded");
        upstream_request.remove_header("referer");

        // Remove any internal proxy headers that might cause issues
        upstream_request.remove_header("x-cache-status");

        debug!(
            "Proxying to: {} with User-Agent: {}",
            target_url, self.config.user_agent
        );

        Ok(())
    }

    async fn response_filter(
        &self,
        _session: &mut Session,
        upstream_response: &mut ResponseHeader,
        ctx: &mut Self::CTX,
    ) -> Result<(), Box<pingora::Error>> {
        self.modify_response_headers(upstream_response);

        // Add cache status header
        if !ctx.cache_hit {
            let _ = upstream_response.insert_header("X-Cache-Status", "MISS");
        }

        Ok(())
    }

    fn response_body_filter(
        &self,
        _session: &mut Session,
        body: &mut Option<Bytes>,
        _end_of_stream: bool,
        ctx: &mut Self::CTX,
    ) -> Result<Option<Duration>, Box<pingora::Error>>
    where
        Self::CTX: Send + Sync,
    {
        // Collect response body for caching
        if let Some(body_bytes) = body {
            ctx.response_body.push(body_bytes.clone());
        }

        Ok(None)
    }

    async fn logging(
        &self,
        session: &mut Session,
        e: Option<&pingora::Error>,
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
        let duration = ctx.start_time.elapsed();

        // Log error if present
        if let Some(error) = e {
            error!("Request error: client_ip={} error={}", client_ip, error);
        }

        // CRITICAL: Only attempt caching for successful GET requests
        if !ctx.cache_hit
            && !ctx.response_body.is_empty()
            && session.req_header().method == Method::GET
            && status == 200
            && self.cache.is_some()
        {
            if let (Some(cache_key), Some(target_url)) = (&ctx.cache_key, &ctx.target_url) {
                if let Some(response_header) = session.response_written() {
                    debug!("Checking if response should be cached for {}", target_url);

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

                        info!(
                            "Caching response: key={}, size={} bytes, TTL={:?}",
                            cache_key,
                            full_body.len(),
                            ttl
                        );

                        // Store in cache
                        match self
                            .store_cache(cache_key.clone(), response_header, full_body, ttl)
                            .await
                        {
                            Ok(_) => {
                                info!(
                                    "Successfully cached response for {} (TTL: {:?})",
                                    target_url, ttl
                                );
                            }
                            Err(e) => {
                                warn!("Failed to cache response for {}: {}", target_url, e);
                            }
                        }
                    } else {
                        debug!("Response not cacheable for {}", target_url);
                    }
                } else {
                    debug!("No response header available for caching");
                }
            } else {
                debug!("Missing cache key or target URL for caching");
            }
        } else {
            debug!("Skipping cache storage: hit={}, body_empty={}, method={}, status={}, cache_enabled={}", 
               ctx.cache_hit, ctx.response_body.is_empty(), session.req_header().method, status, self.cache.is_some());
        }

        // Log request
        if self.config.enable_metrics {
            info!(
            "client_ip={} method={} uri={} status={} cache={} duration={:.2}ms target={} body_size={}",
            client_ip,
            session.req_header().method,
            session.req_header().uri,
            status,
            cache_status,
            duration.as_secs_f64() * 1000.0,
            ctx.target_url
                .as_ref()
                .map(|u| u.as_str())
                .unwrap_or("unknown"),
            ctx.response_body.iter().map(|b| b.len()).sum::<usize>()
        );
        }
    }
}

async fn load_config(path: &str) -> Result<Config> {
    match tokio::fs::read_to_string(path).await {
        Ok(content) => serde_json::from_str(&content).context("Failed to parse config file"),
        Err(_) => {
            warn!("Config file not found, creating default config at {}", path);
            let default_config = Config::default();
            let content = serde_json::to_string_pretty(&default_config)
                .context("Failed to serialize default config")?;
            tokio::fs::write(path, content)
                .await
                .context("Failed to write default config")?;
            Ok(default_config)
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    let log_level = if args.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("embed_proxy={},pingora={}", log_level, log_level))
        .init();

    info!("Starting Embed Proxy Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(load_config(&args.config))?;
    info!("Loaded configuration from {}", args.config);

    // Create server - start with None for default, then we'll configure the proxy service
    let mut server = Server::new(None)?;
    server.bootstrap();

    // Create proxy service
    let proxy = EmbedProxy::new(config);

    let mut proxy_service = pingora_proxy::http_proxy_service(&server.configuration, proxy);

    // Configure listening address
    let addr: StdSocketAddr = args.listen.parse().context("Invalid listen address")?;
    proxy_service.add_tcp(&addr.to_string());

    info!("Proxy server listening on {}", addr);
    info!("Usage: GET http://{}/proxy/https://example.com/path", addr);
    info!(
        "Health check: GET http://{}/proxy/https://httpbin.org/status/200",
        addr
    );

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
        assert!(!proxy.is_domain_allowed("sub.blocked.com"));
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

        let invalid_uri: Uri = "/invalid/path".parse().unwrap();
        assert!(proxy.extract_target_url(&invalid_uri).is_err());

        let empty_uri: Uri = "/proxy/".parse().unwrap();
        assert!(proxy.extract_target_url(&empty_uri).is_err());
    }

    #[test]
    fn test_cache_ttl_parsing() {
        let config = Config::default();
        let proxy = EmbedProxy::new(config);

        // Test max-age directive
        let mut headers = ResponseHeader::build(StatusCode::OK, None).unwrap();
        headers
            .insert_header("cache-control", "public, max-age=7200")
            .unwrap();
        let ttl = proxy.get_cache_ttl(&headers);
        assert_eq!(ttl, Duration::from_secs(7200));

        // Test default TTL
        let headers2 = ResponseHeader::build(StatusCode::OK, None).unwrap();
        let default_ttl = proxy.get_cache_ttl(&headers2);
        assert_eq!(default_ttl, Duration::from_secs(3600)); // Default from config
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(2);

        assert!(limiter.check_rate_limit("127.0.0.1").await);
        assert!(limiter.check_rate_limit("127.0.0.1").await);
        assert!(!limiter.check_rate_limit("127.0.0.1").await); // Should be blocked

        // Different IP should be allowed
        assert!(limiter.check_rate_limit("192.168.1.1").await);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let config = Config::default();
        let proxy = EmbedProxy::new(config);

        let cache_key = "test_key".to_string();
        let headers = ResponseHeader::build(StatusCode::OK, None).unwrap();
        let data = Bytes::from("test data");
        let ttl = Duration::from_secs(60);

        // Test cache miss
        let result = proxy.lookup_cache(&cache_key).await;
        assert!(result.is_none());

        // Test cache store and hit
        proxy
            .store_cache(cache_key.clone(), &headers, data.clone(), ttl)
            .await
            .unwrap();
        let result = proxy.lookup_cache(&cache_key).await;
        assert!(result.is_some());

        let (retrieved_headers, retrieved_data) = result.unwrap();
        assert_eq!(retrieved_data, data);
        assert_eq!(retrieved_headers.status, headers.status);
    }
}
