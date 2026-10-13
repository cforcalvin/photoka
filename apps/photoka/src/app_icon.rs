//! The app icon (#248). The bundle icon is `assets/app-icon/photoka.icns` (packaging/macos);
//! [`window_icon`] sets the Dock icon when the app runs unbundled (`cargo run`).

/// Window, taskbar and (when running unbundled) Dock icon. macOS gets the padded 1024 px render
/// on Apple's icon grid; elsewhere the tighter 256 px hicolor render reads better at small sizes.
pub fn window_icon() -> egui::IconData {
    match eframe::icon_data::from_png_bytes(PNG) {
        Ok(icon) => icon,
        Err(e) => {
            log::warn!("app icon: {e}");
            egui::IconData::default()
        }
    }
}

#[cfg(target_os = "macos")]
const PNG: &[u8] = include_bytes!("../../../assets/app-icon/photoka-1024.png");
#[cfg(not(target_os = "macos"))]
const PNG: &[u8] = include_bytes!("../../../assets/app-icon/hicolor/256x256/apps/io.github.cforcalvin.photoka.png");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_icon_decodes() {
        let icon = window_icon();
        assert!(icon.width >= 256 && icon.width == icon.height, "{}x{}", icon.width, icon.height);
        assert_eq!(icon.rgba.len(), icon.width as usize * icon.height as usize * 4);
        assert!(icon.rgba.chunks(4).any(|p| p[3] > 0), "the icon isn't blank");
    }
}
