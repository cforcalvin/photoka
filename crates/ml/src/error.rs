use thiserror::Error;

/// Everything that can go wrong in model management and generation. Each message is meant to be
/// shown to the user (or an agent) as is.
#[derive(Debug, Error)]
pub enum MlError {
    #[error("{0}")]
    Io(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("{file}: the download doesn't match the catalog (expected sha256 {expected}, got {actual}); it was deleted, try again")]
    HashMismatch { file: String, expected: String, actual: String },
    #[error("{0}")]
    BadModel(String),
    #[error("{0}")]
    BadParams(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("not enough memory: {0}")]
    OutOfMemory(String),
    #[error("not enough disk space: {0}")]
    DiskFull(String),
    #[error("ComfyUI: {0}")]
    Comfy(String),
    #[error("cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

impl MlError {
    pub(crate) fn io(context: impl std::fmt::Display, e: std::io::Error) -> Self {
        if e.raw_os_error() == Some(28) {
            return MlError::DiskFull(format!("{context}: {e}"));
        }
        MlError::Io(format!("{context}: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, MlError>;
