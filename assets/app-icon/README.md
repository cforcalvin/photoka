# Photoka app icon

A **nine-tailed kitsune with a brow diamond**: an engraving portrait, seated, looking back at the
viewer, in the Crafting Apps' owl-template framing (full-bleed field, head and body filling the
tile, tails running off the edges).

## Palette

Exactly three colours:

| Colour | Hex | Used for |
|---|---|---|
| Ink | `#0b0b0c` | line work and contours |
| Paper | `#efe9dc` | the figure |
| Photoka blue (app colour) | `#2f7bf5` | the full-bleed field |

## Geometry

A 512-unit tile (`viewBox="0 0 512 512"`), rounded square with `rx=112`, no border. The macOS
renders pad it onto Apple's 824/1024 icon grid; the Windows and Linux renders crop 22 units off
each side so the figure reads at 16–48 px.

## Provenance

The upstream PhotoCraft owner's original drawing, made in ArtCraft (2880 px, engraving style,
keyed to the palette) and vectorised. The source PNG is not part of this repository. License: see
`LICENSE.txt`.

## Files

- `photoka.svg`: canonical master (traced at 2048 px).
- `photoka-small.svg`: lighter trace (traced at 1024 px); also the hicolor scalable icon.
- `photoka-1024.png`: 1024 px render on the macOS grid.
- `photoka.icns`: macOS bundle icon (`CFBundleIconFile`).
- `photoka.ico`: Windows icon, 16–256 px, embedded in the `.exe` by `apps/photoka/build.rs`.
- `hicolor/<size>/apps/io.github.cforcalvin.photoka.png` (16–512) and `hicolor/scalable/...svg`:
  Linux icon theme.

The app also sets the window icon and Wayland app ID at runtime (`apps/photoka/src/app_icon.rs`,
`main.rs`). On Windows the taskbar shows the icon of the Start Menu shortcut that launches the
exe, so the MSI's `<Icon Id>` keeps the `.exe` extension (ICE50); `app_icon.rs` documents the
details and tests the `.ico` sizes and the WiX icon references.

## Regenerate

Replace `photoka.svg` (and `photoka-small.svg`), then run `packaging/icons.sh`. It needs
`resvg`, plus `iconutil` on macOS; the `.ico` is packed by `cargo xtask ico`.
