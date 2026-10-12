//! Windows only: embed the app icon and version info (VERSIONINFO) into `photoka-cli.exe`.
//!
//! On every other target this does nothing, even when `PHOTOKA_REQUIRE_WINRES` is set. A
//! missing resource compiler is a warning, so a cross-compile from macOS or Linux still links,
//! unless `PHOTOKA_REQUIRE_WINRES=1` (set by the release packaging script) turns it into a
//! build error.

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../assets/app-icon/photoka.ico");
    println!("cargo:rerun-if-env-changed=PHOTOKA_REQUIRE_WINRES");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return Ok(());
    }
    let mut res = winresource::WindowsResource::new();
    // FileVersion / ProductVersion strings and the numeric FILEVERSION / PRODUCTVERSION are left
    // to `WindowsResource::new`, which reads the Cargo package version. A pre-release such as
    // `1.2.3-rc.1` stays in the strings; the numeric fields are `1.2.3.0`.
    res.set_icon("../../assets/app-icon/photoka.ico")
        .set("ProductName", "Photoka")
        .set("FileDescription", "Photoka command-line interface")
        .set("CompanyName", "Learning Machines LLC")
        .set("LegalCopyright", "Copyright (c) the Photoka authors. MIT OR Apache-2.0.")
        .set("OriginalFilename", "photoka-cli.exe")
        .set("InternalName", "photoka-cli");
    if let Err(err) = res.compile() {
        if std::env::var_os("PHOTOKA_REQUIRE_WINRES").is_some() {
            return Err(std::io::Error::other(format!(
                "embedding Windows resources into photoka-cli.exe failed: {err}. \
                 Install the Windows SDK resource compiler (rc.exe) or mingw-w64 windres and rebuild."
            )));
        }
        println!("cargo:warning=photoka-cli.exe built without icon/version resources: {err}");
    }
    Ok(())
}
