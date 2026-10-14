//! The Civitai API key lives in the macOS Keychain, never in the preferences file.

use crate::{MlError, Result};

const SERVICE: &str = "io.github.cforcalvin.photoka";
const CIVITAI: &str = "civitai-api-key";

/// The stored Civitai API key, if any.
pub fn civitai_key() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        security_framework::passwords::get_generic_password(SERVICE, CIVITAI).ok().and_then(|b| String::from_utf8(b).ok()).filter(|k| !k.trim().is_empty())
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Store (or, with an empty key, remove) the Civitai API key.
pub fn set_civitai_key(key: &str) -> Result<()> {
    let key = key.trim();
    if key.len() > 512 || key.chars().any(char::is_control) {
        return Err(MlError::BadParams("that doesn't look like a Civitai API key".into()));
    }
    #[cfg(target_os = "macos")]
    {
        if key.is_empty() {
            // Not stored is already the goal.
            let _ = security_framework::passwords::delete_generic_password(SERVICE, CIVITAI);
            return Ok(());
        }
        security_framework::passwords::set_generic_password(SERVICE, CIVITAI, key.as_bytes())
            .map_err(|e| MlError::Other(format!("could not save the key in the Keychain: {e}")))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = key;
        Err(MlError::Unsupported("the Keychain is only available on macOS".into()))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rejects_malformed_keys_without_touching_the_keychain() {
        assert!(super::set_civitai_key(&"x".repeat(600)).is_err());
        assert!(super::set_civitai_key("a\nb").is_err());
    }
}
