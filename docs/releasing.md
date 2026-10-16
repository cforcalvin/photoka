# Releasing Photoka

Photoka ships for macOS only. Every push to the `release` branch runs
`.github/workflows/release.yml`, which builds the universal macOS app (DMG) and command-line tool
(zip) on `macos-15`, then creates or updates a **draft** GitHub Release named `Photoka <version>`
with those files and `SHA256SUMS.txt`. Nobody sees a draft until a maintainer publishes it.

User-facing names say **Photoka**. Files, binaries and ids stay lowercase
(`photoka-<version>-macos-universal.dmg`, `io.github.cforcalvin.photoka`).

**Signing:** local and CI builds can be Developer ID signed and notarized when credentials are
present (see Secrets below). Without them, `package.sh` ad-hoc signs and skips notarization.
Locally, a notarytool keychain profile (`MACOS_NOTARY_PROFILE`, e.g. `Photoka`) is enough; CI
uses the certificate and Apple ID secrets.

## Cutting a release

1. **Bump the version** on `main`. The only place it lives is `[workspace.package] version`
   in the root `Cargo.toml`:

   ```sh
   cargo xtask version                 # prints the current version, e.g. 0.1.0
   cargo xtask version set 0.2.0       # or 0.2.0-rc.1; updates Cargo.toml and Cargo.lock
   ```

   Commit the change (`Cargo.toml` + `Cargo.lock`) through the normal review flow.
2. **Merge `main` into `release`** (or fast-forward it) and push. The workflow starts by itself.
3. **Wait for the draft.** The Releases page gets a draft `Photoka 0.2.0`, tagged `v0.2.0` on the
   pushed commit, with both artifacts and `SHA256SUMS.txt`. The notes are generated from the
   merged PRs.
4. **Check it.** Download the DMG, open the app, and run the CLI's `--version`.
5. **Add the scorecard deltas** to the draft's notes: what moved in
   [`docs/scorecard.md`](scorecard.md) since the previous release
   (`git diff v<previous> v<this> -- docs/scorecard.md`).
6. **Publish** the draft in the GitHub UI. Versions with a pre-release suffix (`-rc.1`) are
   marked as pre-releases.

Pushing to `release` again before you publish replaces the draft's assets.

**Test runs:** *Actions → Release → Run workflow* runs the whole pipeline by hand. The optional
`version` input (such as `0.2.0-rc.1`) overrides `Cargo.toml` for that run only; the job applies
it with `cargo xtask version set` before building, so the binaries report it too.

## What gets built

| Platform | Artifacts | Built on |
|---|---|---|
| macOS 11+ (universal: Apple silicon + Intel) | `photoka-<v>-macos-universal.dmg`, `photoka-cli-<v>-macos-universal.zip` | `macos-15` |

Every binary reports its version, the commit and the build date: `photoka --version`,
`photoka-cli --version`, and *Help › About Photoka*. `packaging/env.sh` sets `PHOTOKA_BUILD_SHA`
and `PHOTOKA_BUILD_DATE`, and `crates/engine/src/build_info.rs` reads them at compile time. A
plain `cargo build` doesn't set them and reports `0.2.0 (dev build)`.

The release job checks out [craft-fonts](https://github.com/storytold/craft-fonts) at the commit
pinned in `.github/workflows/release.yml` and builds with `CRAFT_FONTS_DIR` and
`CRAFT_FONTS_REQUIRED=1`, so releases embed its Japanese fonts and fail rather than ship without
them. The app carries each font's licence as `OFL-<family>.txt` in `Contents/Resources/Licenses`
(`copy_font_licences` in `packaging/env.sh`). Bump the pin deliberately, together with the one in
`ci.yml`. Build option: `docs/development.md` › Fonts.

### macOS

`packaging/macos/package.sh` builds `aarch64-apple-darwin` and `x86_64-apple-darwin` with
`MACOSX_DEPLOYMENT_TARGET=11.0`, joins them with `lipo`, and assembles `Photoka.app`:

- `Info.plist` is generated from `Info.plist.in`. The bundle id is `io.github.cforcalvin.photoka`.
  The plist sets `LSMinimumSystemVersion` 11.0, `NSHighResolutionCapable`, and document types:
  `.photoka` (Owner), plus PSD/PSB and the image formats Photoka reads (Alternate, so it never
  takes over Preview's defaults). The icon is `assets/app-icon/photoka.icns`.
- **Signing** goes inside-out with the hardened runtime and a secure timestamp. The executable
  is signed first, then the bundle. There's no `--deep` on the final signature. The
  entitlements (`entitlements.plist`) are deliberately empty.
- **Notarization:** the app is zipped and sent with `xcrun notarytool submit --wait`, then the
  ticket is stapled to the app. The app goes on a DMG (`hdiutil`, with an `Applications` link
  to drag onto). The DMG is signed, notarized and stapled too. The script checks the results
  with `codesign --verify --strict`, `stapler validate` and `spctl -a -vvv`.
  Its Finder window (background, icon size and positions) comes from
  [`packaging/macos/dmg/`](../packaging/macos/dmg/README.md), and its volume is named `Photoka`
  without the version, which the window's background needs; the DMG file name keeps the version.
- **CLI:** the universal `photoka-cli` is signed with the same Developer ID, the hardened
  runtime and a secure timestamp (identifier `io.github.cforcalvin.photoka-cli`), zipped, and the zip
  is sent to `notarytool`. Only `.app`, `.dmg` and `.pkg` can hold a stapled ticket, not a bare
  Mach-O, so Gatekeeper looks the CLI's ticket up online the first time a downloaded
  (quarantined) copy runs. Offline, that first run can be refused until the Mac is online again.
- **Verification:** `packaging/macos/verify.sh` checks the artifacts as users download them. It
  unpacks the CLI zip and requires, for the binary inside, `codesign --verify --strict`, both
  architectures, the hardened runtime flag, a `Developer ID Application` authority from
  `APPLE_TEAM_ID`, a timestamp, and `spctl --assess --type install` reporting
  `source=Notarized Developer ID` (the same online lookup Gatekeeper does). For the DMG it runs
  `codesign --verify`, `stapler validate` and `spctl --type open`. Without signing secrets it only
  checks signature integrity and warns. Run it locally after a build:
  `packaging/macos/verify.sh --arch aarch64`.

Locally, without certificates, the script signs ad-hoc (`codesign -s -`) and skips notarization.
That's enough to check the bundle and the DMG on your own Mac:

```sh
packaging/macos/package.sh                    # universal; needs both rustup targets
packaging/macos/package.sh --arch aarch64     # quicker, host-only
open dist/release/photoka-*-macos-*.dmg
```

## Secrets

**Local notarization:** set `MACOS_SIGN_IDENTITY` to your Developer ID Application identity and
`MACOS_NOTARY_PROFILE` to a profile from `xcrun notarytool store-credentials` (e.g. `Photoka`),
then run `packaging/macos/package.sh`.

**CI:** add these to a `release` environment (*Settings → Environments → release*, deployment
branches restricted to `release`), give the `macos` job `environment: release`, and add a step
before packaging that runs `packaging/macos/import-cert.sh` with the certificate secrets and
passes the Apple ID secrets to `package.sh`:

| Secret | Used for |
|---|---|
| `APPLE_CERTIFICATE` | base64 `.p12` with a "Developer ID Application" certificate |
| `APPLE_CERTIFICATE_PASSWORD` | password for that `.p12` |
| `KEYCHAIN_PASSWORD` | password for the temporary CI keychain (random if unset) |
| `APPLE_ID` | Apple ID used by `notarytool` |
| `APPLE_PASSWORD` | app-specific password for that Apple ID |
| `APPLE_TEAM_ID` | the certificate's team id |

`packaging/macos/import-cert.sh` decodes the `.p12` into a temporary keychain
(`security create-keychain`, then `security import`, then `set-key-partition-list` so codesign
can use the key without a prompt). It exports the identity's SHA-1 as `MACOS_SIGN_IDENTITY`.
Delete the keychain at the end of the job.

`GITHUB_TOKEN` creates the release.

## Icons

`assets/app-icon/photoka.svg` is the canonical icon: the maintainer's artwork of a white sprite
on a dandelion, embedded as a raster in a 512-unit rounded tile (MIT OR Apache-2.0, see
`LICENSE.txt` there; geometry in `README.md`). `packaging/icons.sh` regenerates the 1024 px PNG, the `.icns` and the
hicolor PNGs (in-app branding) from it. It needs `resvg` and `iconutil`. The outputs are
committed, so packaging never needs those tools.

## Checks

`.github/workflows/packaging-lint.yml` runs in seconds on any change to `packaging/` or the
workflows: actionlint (which also shellchecks every `run:` block), shellcheck on the packaging
scripts, and a parse of the macOS plists.
