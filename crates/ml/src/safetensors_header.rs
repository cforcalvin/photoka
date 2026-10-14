//! Read just the header of a `.safetensors` file (tensor names, dtypes, shapes) without loading
//! the weights: model detection and import checks use it.
//!
//! Format: an 8-byte little-endian header length, then that many bytes of JSON mapping tensor
//! names to `{dtype, shape, data_offsets}`, plus an optional `__metadata__` object.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use crate::{MlError, Result};

/// Real headers are a few MB at most; anything bigger is not a model we can use.
const MAX_HEADER: u64 = 100 << 20;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Byte range in the data section (after the header).
    pub offsets: [u64; 2],
}

impl TensorInfo {
    /// Bytes per element of `dtype`, for the float and integer types safetensors files use.
    pub fn element_size(&self) -> Option<u64> {
        Some(match self.dtype.as_str() {
            "F64" | "I64" | "U64" => 8,
            "F32" | "I32" | "U32" => 4,
            "F16" | "BF16" | "I16" | "U16" => 2,
            "U8" | "I8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
            _ => return None,
        })
    }

    pub fn elements(&self) -> Option<u64> {
        self.shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Header {
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: BTreeMap<String, String>,
    /// File offset of the data section.
    pub data_start: u64,
    /// File length when read.
    pub file_len: u64,
}

impl Header {
    pub fn read(path: &Path) -> Result<Header> {
        let mut f = std::fs::File::open(path).map_err(|e| MlError::io(path.display(), e))?;
        let file_len = f.metadata().map_err(|e| MlError::io(path.display(), e))?.len();
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|_| MlError::BadModel(format!("{}: not a safetensors file (too short)", path.display())))?;
        let n = u64::from_le_bytes(len);
        if n == 0 || n > MAX_HEADER || n.saturating_add(8) > file_len {
            return Err(MlError::BadModel(format!("{}: not a safetensors file (bad header length)", path.display())));
        }
        let mut buf = vec![0u8; n as usize];
        f.read_exact(&mut buf).map_err(|e| MlError::io(path.display(), e))?;
        let mut h = Self::parse(&buf).map_err(|e| MlError::BadModel(format!("{}: {e}", path.display())))?;
        h.data_start = 8 + n;
        h.file_len = file_len;
        Ok(h)
    }

    /// The raw little-endian bytes of tensor `name`, checked against its dtype, shape and the
    /// file's length. `f` is the open file this header was read from.
    pub fn read_bytes(&self, f: &mut std::fs::File, name: &str) -> Result<Vec<u8>> {
        use std::io::{Seek, SeekFrom};
        let t = self.tensors.get(name).ok_or_else(|| MlError::BadModel(format!("tensor {name} is missing")))?;
        let [a, b] = t.offsets;
        let len = b.checked_sub(a).ok_or_else(|| MlError::BadModel(format!("tensor {name} has a bad byte range")))?;
        let expected = t.elements().zip(t.element_size()).and_then(|(n, s)| n.checked_mul(s));
        if expected != Some(len) {
            return Err(MlError::BadModel(format!("tensor {name}: {len} bytes don't match its {} shape {:?}", t.dtype, t.shape)));
        }
        let start = self.data_start.checked_add(a).filter(|s| s.saturating_add(len) <= self.file_len);
        let start = start.ok_or_else(|| MlError::BadModel(format!("tensor {name} lies outside the file (truncated download?)")))?;
        let len = usize::try_from(len).map_err(|_| MlError::BadModel(format!("tensor {name} is too large")))?;
        f.seek(SeekFrom::Start(start)).map_err(|e| MlError::io(name, e))?;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf).map_err(|e| MlError::io(name, e))?;
        Ok(buf)
    }

    pub fn parse(json: &[u8]) -> std::result::Result<Header, String> {
        let v: serde_json::Value = serde_json::from_slice(json).map_err(|e| format!("not a safetensors file ({e})"))?;
        let obj = v.as_object().ok_or("not a safetensors file (header is not an object)")?;
        let mut h = Header::default();
        for (name, t) in obj {
            if name == "__metadata__" {
                if let Some(m) = t.as_object() {
                    for (k, v) in m {
                        if let Some(s) = v.as_str() {
                            h.metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = t.get("dtype").and_then(|d| d.as_str()).ok_or_else(|| format!("tensor {name} has no dtype"))?;
            let shape = t
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| format!("tensor {name} has no shape"))?
                .iter()
                .map(|d| d.as_u64().ok_or_else(|| format!("tensor {name} has a bad shape")))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let offsets = match t.get("data_offsets").and_then(|o| o.as_array()).map(|o| o.iter().map(|v| v.as_u64()).collect::<Vec<_>>()) {
                Some(o) if o.len() == 2 => [o[0].unwrap_or(0), o[1].unwrap_or(0)],
                _ => return Err(format!("tensor {name} has no data_offsets")),
            };
            h.tensors.insert(name.clone(), TensorInfo { dtype: dtype.to_string(), shape, offsets });
        }
        Ok(h)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[u64]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    pub fn any_key(&self, f: impl Fn(&str) -> bool) -> bool {
        self.tensors.keys().any(|k| f(k))
    }
}

#[cfg(test)]
pub(crate) fn write_test_file(path: &Path, tensors: &[(&str, &[u64])]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, shape) in tensors {
        let n: u64 = shape.iter().product::<u64>() * 2;
        header.insert((*name).into(), serde_json::json!({"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + n]}));
        offset += n;
    }
    let json = serde_json::to_vec(&header).unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    bytes.resize(bytes.len() + offset as usize, 0);
    std::fs::write(path, bytes).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_header_and_rejects_garbage() {
        let dir = std::env::temp_dir().join(format!("photoka-ml-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.safetensors");
        write_test_file(&p, &[("x.weight", &[2, 3]), ("y", &[4])]);
        let h = Header::read(&p).unwrap();
        assert_eq!(h.shape("x.weight"), Some(&[2u64, 3][..]));
        assert!(h.has("y"));
        let mut f = std::fs::File::open(&p).unwrap();
        assert_eq!(h.read_bytes(&mut f, "x.weight").unwrap().len(), 12);
        assert!(h.read_bytes(&mut f, "nope").is_err());
        let mut lying = h.clone();
        lying.tensors.get_mut("y").unwrap().offsets = [0, 1 << 40];
        assert!(lying.read_bytes(&mut f, "y").is_err());
        let mut short = h.clone();
        short.file_len = 20;
        assert!(short.read_bytes(&mut f, "y").is_err(), "truncated file");
        std::fs::write(&p, b"short").unwrap();
        assert!(Header::read(&p).is_err());
        std::fs::write(&p, [255u8; 16]).unwrap();
        assert!(Header::read(&p).is_err());
        let mut bad = 5u64.to_le_bytes().to_vec();
        bad.extend_from_slice(b"[1,2]");
        std::fs::write(&p, bad).unwrap();
        assert!(Header::read(&p).is_err());
        assert!(Header::parse(br#"{"t":{"dtype":"F16","shape":[-1]}}"#).is_err());
        assert!(Header::parse(br#"{"t":{"dtype":"F16","shape":[1]}}"#).is_err(), "no offsets");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
