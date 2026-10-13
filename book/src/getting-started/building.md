# Building

Photoka is a macOS app and uses the Rust workspace at the repository root. Install the stable Rust toolchain and the Xcode command-line tools, clone the repository, and run commands from that root.

```sh
cargo build --workspace
cargo run --release -p photoka -- image.psd
cargo run -p photoka-cli -- --help
```

The standard repository gates are:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo xtask layers
```

UI work should also be checked using the offscreen snapshot example described in the [development guide](https://github.com/cforcalvin/photoka/blob/main/docs/development.md).

## Build the documentation

Install mdBook, then build from the repository root:

```sh
cargo install mdbook --locked
mdbook build book
```

The generated site is written to `book/output/` and is not source documentation.
