FROM rust:1.89-slim-bullseye AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./

RUN mkdir src && echo "fn main() {}" > src/main.rs

RUN cargo build --release && rm -rf src target/release/deps/http_proxy_rs*

COPY src ./src

RUN cargo build --release

RUN ls -la target/release/ && \
    file target/release/http-proxy-rs && \
    ldd target/release/http-proxy-rs

FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

WORKDIR /app

COPY --from=builder /app/target/release/http-proxy-rs /usr/local/bin/http-proxy-rs

COPY config.json ./config.json

RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app && \
    chmod +x /usr/local/bin/http-proxy-rs

USER appuser

EXPOSE 8080

ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

CMD ["http-proxy-rs", "--listen", "0.0.0.0:8080", "--config", "config.json"]
