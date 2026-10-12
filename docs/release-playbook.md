# Release playbook

How Photoka builds and publishes release binaries is described in [releasing.md](releasing.md)
(`.github/workflows/release.yml`, `packaging/`, `cargo xtask version`).

The release workflow calls the shared, public workflows in
[storytold/craft-common-actions](https://github.com/storytold/craft-common-actions), pinned to a
commit. No signing secrets are configured, so packages are unsigned until the project has its own
Apple and Windows signing certificates.
