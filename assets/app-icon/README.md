# Photoka app icon

A **round white sprite sitting in a dandelion's leaves**: a soft, kawaii character with dot eyes,
a small smile and blushed cheeks, a wisp curling from its head, on a green stem against a flat
grey field.

## Geometry

A 512-unit tile (`viewBox="0 0 512 512"`), rounded square with `rx=112`, no border. The artwork is
a 1024 px raster embedded in the SVG (JPEG, data URI) and clipped to the tile, over a `#919191`
fill that matches its background. The macOS renders pad it onto Apple's 824/1024 icon grid; the
Windows and Linux renders crop 22 units off each side so the figure reads at 16–48 px.

## Provenance

Original artwork supplied by the Photoka maintainer. No third-party material. License: see
`LICENSE.txt`.

## Files

- `photoka.svg`: canonical master; also the hicolor scalable icon, the README logo and the DMG
  window's artwork.
- `photoka-1024.png`: 1024 px render on the macOS grid.
- `photoka.icns`: macOS bundle icon (`CFBundleIconFile`).
- `hicolor/<size>/apps/io.github.cforcalvin.photoka.png` (16–512) and `hicolor/scalable/...svg`:
  Linux icon theme, and the in-app title-bar mark (`crates/ui-egui/src/brand.rs`, 128 px).

The app also sets the window icon and Wayland app ID at runtime (`apps/photoka/src/app_icon.rs`,
`main.rs`).

## Regenerate

Replace `photoka.svg` (keep the 512-unit `viewBox`), then run `packaging/icons.sh`. It needs
`resvg`, plus `iconutil` on macOS. The DMG background links `photoka.svg`; regenerate it with
`packaging/macos/dmg/generate.py`.
