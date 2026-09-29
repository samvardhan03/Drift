//! Embeds `VERGEN_GIT_SHA` (the build's git commit) into the environment
//! `env!`/`option_env!` can read at compile time, so `trace::engine_commit`
//! can surface it in every `EvidenceTrace`. Never fails the build --
//! `.emit()`'s `Result` is deliberately discarded, per spec: a source
//! tarball with no `.git` directory (or any other git-unavailable
//! environment) should still build, just with `engine_commit` falling back
//! to `"unknown"`.
fn main() {
    let _ = vergen::EmitBuilder::builder().git_sha(false).emit();
}
