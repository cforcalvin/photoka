# Installation

Photoka is currently an early-alpha project for macOS 11 or later. Release artifacts (a universal DMG and a command-line tool zip), when available, are published through the repository's GitHub Releases page. Verify the release notes and checksums before running downloaded artifacts.

For source builds, continue to [Building](building.md). The desktop application and the headless CLI are separate workspace applications:

- `apps/photoka`: native desktop application
- `apps/photoka-cli`: headless CLI, JSON-lines server, and MCP server

Packaging and signing details are maintained in the existing [release documentation](https://github.com/cforcalvin/photoka/blob/main/docs/releasing.md). Signing is conditional on the configured release secrets; the existence of an artifact alone is not proof that it was signed.

## Trusting input files

Treat PSD, PSB, `.photoka`, raster images, profiles, LUTs, patterns, and embedded document content as potentially malicious. Review [Secure file handling](../security/secure-file-handling.md) before using unknown files in privileged or sensitive environments.
