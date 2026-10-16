#!/usr/bin/env bash
# Build, sign and (optionally) notarize the macOS release artifacts:
#
#   $DIST/photoka-<version>-macos-<arch>.dmg          Photoka.app on a drag-to-Applications DMG
#   $DIST/photoka-cli-<version>-macos-<arch>.zip      the headless CLI
#
# Usage: packaging/macos/package.sh [--arch universal|aarch64|x86_64] [--skip-build]
#
# Signing (env):
#   MACOS_SIGN_IDENTITY   codesign identity (name or SHA-1). Default "-" = ad-hoc (local testing;
#                         Gatekeeper will reject the result on other Macs). In CI, import-cert.sh sets it.
#   MACOS_KEYCHAIN        keychain holding the identity (optional)
# Notarization (env; a real identity, plus either a keychain profile or the three Apple ID vars):
#   MACOS_NOTARY_PROFILE  name from `xcrun notarytool store-credentials` (preferred locally)
#   APPLE_ID, APPLE_PASSWORD (app-specific password), APPLE_TEAM_ID
set -euo pipefail
# On exFAT (and other non-HFS volumes) macOS writes AppleDouble `._*` sidecars next to every
# copied file. codesign treats those as nested Mach-O and fails; stop creating them, and strip
# any that already exist before signing.
export COPYFILE_DISABLE=1
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$ROOT/packaging/macos"

ARCH=universal
SKIP_BUILD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h | --help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ARCH" in
  universal) TARGETS=(aarch64-apple-darwin x86_64-apple-darwin) ;;
  aarch64) TARGETS=(aarch64-apple-darwin) ;;
  x86_64) TARGETS=(x86_64-apple-darwin) ;;
  *) echo "unknown --arch $ARCH" >&2; exit 2 ;;
esac

# Keep in sync with LSMinimumSystemVersion in Info.plist.in.
export MACOSX_DEPLOYMENT_TARGET=11.0
IDENTITY="${MACOS_SIGN_IDENTITY:--}"
SHORT_VERSION="${VERSION%%-*}"
WORK="$CARGO_TARGET_DIR/macos-package"
# Assemble and sign on the boot volume. CARGO_TARGET_DIR may be on exFAT (no xattrs), and
# codesign then spills AppleDouble `._*` files that break `codesign --verify --strict`.
STAGE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/photoka-macos.XXXXXX")"
trap 'rm -rf "$STAGE_ROOT"' EXIT
APP="$STAGE_ROOT/Photoka.app"
DMG="$DIST/photoka-$VERSION-macos-$ARCH.dmg"
CLI_ZIP="$DIST/photoka-cli-$VERSION-macos-$ARCH.zip"

NOTARIZE=0
if [ "$IDENTITY" = "-" ]; then
  warn "macOS: ad-hoc signing (no MACOS_SIGN_IDENTITY); artifacts are not notarized"
elif [ -n "${MACOS_NOTARY_PROFILE:-}" ]; then
  NOTARIZE=1
elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_PASSWORD:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
  NOTARIZE=1
else
  warn "macOS: set MACOS_NOTARY_PROFILE, or APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID; signed but not notarized"
fi

echo "==> Photoka $VERSION for macOS ($ARCH), identity: $IDENTITY, notarize: $NOTARIZE${MACOS_NOTARY_PROFILE:+ (profile $MACOS_NOTARY_PROFILE)}"

# ---- build -------------------------------------------------------------------------------------
if [ "$SKIP_BUILD" = 0 ]; then
  args=()
  for t in "${TARGETS[@]}"; do args+=(--target "$t"); done
  (cd "$ROOT" && cargo build --release --locked -p photoka -p photoka-cli --features heif "${args[@]}")
fi

rm -rf "$WORK"
mkdir -p "$WORK/bin"
for bin in photoka photoka-cli; do
  inputs=()
  for t in "${TARGETS[@]}"; do inputs+=("$CARGO_TARGET_DIR/$t/release/$bin"); done
  lipo -create -output "$WORK/bin/$bin" "${inputs[@]}"
  lipo -info "$WORK/bin/$bin"
done

# ---- signing helpers ---------------------------------------------------------------------------
sign() {
  # Hardened runtime + secure timestamp for a real identity; ad-hoc can't be timestamped.
  local ts=(--timestamp)
  [ "$IDENTITY" = "-" ] && ts=(--timestamp=none)
  local kc=()
  [ -n "${MACOS_KEYCHAIN:-}" ] && kc=(--keychain "$MACOS_KEYCHAIN")
  codesign --force --sign "$IDENTITY" ${kc[@]+"${kc[@]}"} "${ts[@]}" "$@"
}

notarize() {
  local file="$1" out id status
  local creds=()
  if [ -n "${MACOS_NOTARY_PROFILE:-}" ]; then
    creds=(--keychain-profile "$MACOS_NOTARY_PROFILE")
  else
    creds=(--apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID")
  fi
  echo "==> notarizing $(basename "$file") (this can take a few minutes)"
  out="$(xcrun notarytool submit "$file" "${creds[@]}" --wait --timeout 1h --output-format json)" || true
  echo "$out"
  id="$(printf '%s' "$out" | plutil -extract id raw -o - - 2>/dev/null || true)"
  status="$(printf '%s' "$out" | plutil -extract status raw -o - - 2>/dev/null || true)"
  if [ "$status" != "Accepted" ]; then
    if [ -n "$id" ]; then
      xcrun notarytool log "$id" "${creds[@]}" || true
    fi
    echo "error: notarization of $(basename "$file") failed (status: ${status:-unknown})" >&2
    exit 1
  fi
}

# ---- Photoka.app ----------------------------------------------------------------------------
echo "==> assembling $APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
# Executable and icon carry the display name (CFBundleExecutable / CFBundleIconFile).
cp "$WORK/bin/photoka" "$APP/Contents/MacOS/Photoka"
cp "$ROOT/assets/app-icon/photoka.icns" "$APP/Contents/Resources/Photoka.icns"
# Licences of the embedded craft-fonts fonts (only when built with CRAFT_FONTS_DIR).
if [ -n "${CRAFT_FONTS_DIR:-}" ]; then
  mkdir -p "$APP/Contents/Resources/Licenses"
  copy_font_licences "$APP/Contents/Resources/Licenses"
fi
sed -e "s/@VERSION@/$VERSION/g" -e "s/@SHORT_VERSION@/$SHORT_VERSION/g" \
  -e "s/@BUILD_SHA@/${PHOTOKA_BUILD_SHA:-unknown}/g" \
  "$HERE/Info.plist.in" >"$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist"
printf 'APPL????' >"$APP/Contents/PkgInfo"

# Sign inside-out: nested code first, then the bundle itself (no --deep on the final signature).
# Today the only nested code is the main executable; frameworks/helpers would be signed here too.
sign --options runtime --entitlements "$HERE/entitlements.plist" "$APP/Contents/MacOS/Photoka"
sign --options runtime --entitlements "$HERE/entitlements.plist" "$APP"
codesign --verify --strict --deep --verbose=2 "$APP"

if [ "$NOTARIZE" = 1 ]; then
  ditto -c -k --keepParent "$APP" "$WORK/Photoka-notarize.zip"
  notarize "$WORK/Photoka-notarize.zip"
  xcrun stapler staple "$APP"
  xcrun stapler validate "$APP"
  spctl --assess --type execute -vvv "$APP"
fi

# ---- DMG ---------------------------------------------------------------------------------------
echo "==> building $DMG"
STAGE="$STAGE_ROOT/dmg"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/Photoka.app"
ln -s /Applications "$STAGE/Applications"
# Finder window layout: background, icon size and positions (packaging/macos/dmg/README.md).
mkdir -p "$STAGE/.background"
cp "$HERE/dmg/background.tiff" "$STAGE/.background/background.tiff"
cp "$HERE/dmg/dmg-layout.DS_Store" "$STAGE/.DS_Store"
rm -f "$DMG" "$STAGE_ROOT/raw.dmg"
# The volume name has no version: .DS_Store finds the background through an alias that includes it.
# `create -srcfolder` copies files as they are. `makehybrid -hfs` stamps com.apple.FinderInfo on
# every file, which survives the drag to /Applications and makes the installed app fail
# `codesign --verify --strict` and Gatekeeper; it is only the fallback for runners where
# `create -srcfolder` fails ("Resource busy", sandboxed sessions).
if ! hdiutil create -volname "Photoka" -srcfolder "$STAGE" -fs HFS+ -format UDZO \
  -imagekey zlib-level=9 -o "$DMG"; then
  echo "warning: hdiutil create failed; falling back to makehybrid (installed app will carry FinderInfo)" >&2
  rm -f "$DMG"
  hdiutil makehybrid -hfs -hfs-volume-name "Photoka" -hfs-openfolder "$STAGE" -o "$STAGE_ROOT/raw.dmg" "$STAGE"
  hdiutil convert "$STAGE_ROOT/raw.dmg" -format UDZO -imagekey zlib-level=9 -o "$DMG"
  rm -f "$STAGE_ROOT/raw.dmg"
fi
sign "$DMG"
codesign --verify --strict --verbose=2 "$DMG"
if [ "$NOTARIZE" = 1 ]; then
  notarize "$DMG"
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG"
  spctl --assess --type open --context context:primary-signature -vvv "$DMG"
fi

# ---- CLI ---------------------------------------------------------------------------------------
echo "==> building $CLI_ZIP"
CLI_DIR="$STAGE_ROOT/photoka-cli-$VERSION-macos-$ARCH"
mkdir -p "$CLI_DIR"
cp "$WORK/bin/photoka-cli" "$CLI_DIR/"
copy_docs "$CLI_DIR"
# Same Developer ID and hardened runtime as the app, with an explicit reverse-DNS identifier
# (codesign would otherwise use the bare file name).
sign --options runtime --identifier io.github.cforcalvin.photoka-cli "$CLI_DIR/photoka-cli"
codesign --verify --strict --verbose=2 "$CLI_DIR/photoka-cli"
rm -f "$CLI_ZIP"
ditto -c -k --keepParent "$CLI_DIR" "$CLI_ZIP"
# notarytool accepts a zip of the bare binary. A bare Mach-O can't carry a stapled ticket, so
# Gatekeeper looks the notarization up online when a quarantined copy first runs.
if [ "$NOTARIZE" = 1 ]; then notarize "$CLI_ZIP"; fi

"$WORK/bin/photoka-cli" --version
echo "==> done (check the artifacts as shipped with packaging/macos/verify.sh --arch $ARCH)"
ls -lh "$DMG" "$CLI_ZIP"
