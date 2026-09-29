# syntax=docker/dockerfile:1

# ---- Stage 1: build ---------------------------------------------------
# rust:1.90-slim, not 1.82-slim as originally specified: our resolved
# Cargo.lock (generated with a current toolchain) pulls in transitive
# dependencies whose MSRV exceeds 1.82 (clap_lex needs Cargo's
# edition2024 support, stabilized in 1.85; icu_provider/icu_normalizer
# etc. need rustc 1.88). See the README's Docker section for the verbatim
# build errors at each version tried and the full judgment call.
FROM rust:1.90-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache the dependency-compilation layer. Copy only the manifests/lockfile
# and a stub for every workspace crate's declared targets (lib.rs for each
# of compute/agent/server/store, plus compute's `experiment` bin — cargo
# needs the source file for every target declared in a member's Cargo.toml
# to exist even when building only `-p server`, and `store` is a workspace
# member too since it's a path dependency of `compute`), then do a
# throwaway build. As long as no Cargo.toml/Cargo.lock changed, Docker
# reuses this layer — and cargo reuses its compiled *dependency* artifacts
# in target/ — on later builds where only application source under src/
# changed.
COPY Cargo.toml Cargo.lock ./
COPY crates/compute/Cargo.toml crates/compute/Cargo.toml
COPY crates/agent/Cargo.toml crates/agent/Cargo.toml
COPY crates/server/Cargo.toml crates/server/Cargo.toml
COPY crates/store/Cargo.toml crates/store/Cargo.toml

RUN mkdir -p crates/compute/src/bin crates/agent/src crates/server/src crates/store/src \
    && echo "pub fn _dummy() {}" > crates/compute/src/lib.rs \
    && echo "fn main() {}" > crates/compute/src/bin/experiment.rs \
    && echo "pub fn _dummy() {}" > crates/agent/src/lib.rs \
    && echo "fn main() {}" > crates/server/src/main.rs \
    && echo "pub fn _dummy() {}" > crates/store/src/lib.rs \
    && cargo build --release -p server \
    && rm -rf crates/compute/src crates/agent/src crates/server/src crates/store/src

# Now bring in the real source and rebuild. Only the crates whose source
# actually changed get recompiled; external dependencies (nalgebra,
# reqwest, axum, tokio, ...) reuse the cache from the throwaway build above.
#
# The `touch` before rebuilding is load-bearing, not decorative: BuildKit's
# COPY can leave mtimes that don't postdate cargo's fingerprint records from
# the dummy build, so cargo's fast mtime pre-check can conclude nothing
# changed and silently keep linking the *dummy* stub binary even though the
# real source is already in place. Verified by building without the touch:
# `cargo build` finished in 0.08s (compiling nothing) and the resulting
# image's `/server` exited immediately with no output — the dummy
# `fn main() {}`, not the real server.
COPY crates/compute/src crates/compute/src
COPY crates/compute/build.rs crates/compute/build.rs
COPY crates/agent/src crates/agent/src
COPY crates/server/src crates/server/src
COPY crates/server/static crates/server/static
COPY crates/store/src crates/store/src

RUN find crates/compute/src crates/agent/src crates/server/src crates/store/src -type f -exec touch {} + \
    && cargo build --release -p server

# distroless has no shell, so `RUN mkdir` isn't available in the run stage
# below -- create the SQLite data directory here, in the build stage (which
# does have a shell), and copy the empty directory across instead.
RUN mkdir -p /data

# ---- Stage 2: run -------------------------------------------------------
# distroless/cc-debian12 (glibc + libgcc/libstdc++, no shell, no package
# manager) rather than a static-musl build — see the README's "Docker" /
# static linking judgment call for why.
FROM gcr.io/distroless/cc-debian12

COPY --from=build /app/target/release/server /server
COPY --from=build /data /data

EXPOSE 8080
CMD ["/server"]
