//! Resumable, sha256-verified downloads into the models folder.
//!
//! Bytes go to `<file>.part`; a cancelled or dropped download resumes from there with an HTTP
//! `Range` request. The file is renamed into place only once its size and sha256 match the
//! catalog. An API key is sent only to the first host: redirects to a CDN never see it.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::{MlError, Progress, Result};

const USER_AGENT: &str = "Photoka-dev";
/// Keep this much free on the disk after a download.
const DISK_MARGIN: u64 = 512 << 20;
const ATTEMPTS: u32 = 4;
const CHUNK: usize = 1 << 20;

/// One file to fetch.
#[derive(Clone, Debug)]
pub struct DownloadRequest {
    pub url: String,
    pub dest: PathBuf,
    pub size: u64,
    pub sha256: String,
    /// Sent as `Authorization: Bearer …` to the first host only.
    pub bearer: Option<String>,
}

/// Fetch `req.url` to `req.dest`, resuming a partial download. Cancellation keeps the partial
/// file, so the next call continues where this one stopped.
pub fn download(req: &DownloadRequest, progress: &dyn Progress) -> Result<()> {
    let name = req.dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    if req.dest.is_file() && std::fs::metadata(&req.dest).map(|m| m.len()).ok() == Some(req.size) {
        return Ok(());
    }
    let dir = req.dest.parent().ok_or_else(|| MlError::BadParams("download destination has no folder".into()))?;
    std::fs::create_dir_all(dir).map_err(|e| MlError::io(dir.display(), e))?;
    let part = part_path(&req.dest);
    let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if have > req.size {
        let _ = std::fs::remove_file(&part);
        have = 0;
    }
    let needed = req.size.saturating_sub(have).saturating_add(DISK_MARGIN);
    if let Some(free) = crate::system::available_space(dir)
        && free < needed
    {
        return Err(MlError::DiskFull(format!("{name} needs {} free, {} is available", human(needed), human(free))));
    }

    let mut hasher = Sha256::new();
    if have > 0 {
        progress.report(0.0, &format!("Checking the partial download of {name}"));
        have = hash_existing(&part, have, &mut hasher, progress)?;
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .user_agent(USER_AGENT)
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into();

    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        progress.check()?;
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(u64::from(attempt) * 500));
        }
        match fetch(&agent, req, &part, &mut have, &mut hasher, &name, progress) {
            Ok(()) => {
                last_err = None;
                break;
            }
            Err(e @ (MlError::Network(_) | MlError::Io(_))) if have < req.size => {
                log::warn!("download of {name} interrupted ({e}); retrying");
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    if let Some(e) = last_err {
        return Err(MlError::Network(format!("{name}: {e}; download again to resume")));
    }
    if have != req.size {
        return Err(MlError::Network(format!("{name}: the server sent {} of {}; download again to resume", human(have), human(req.size))));
    }
    let actual = hex(&hasher.finalize());
    if actual != req.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(MlError::HashMismatch { file: name, expected: req.sha256.clone(), actual });
    }
    std::fs::rename(&part, &req.dest).map_err(|e| MlError::io(req.dest.display(), e))?;
    progress.report(1.0, &format!("Downloaded {name}"));
    Ok(())
}

/// `<dest>.part`.
pub fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Bytes already downloaded for `dest`, if a partial download exists.
pub fn partial_len(dest: &Path) -> Option<u64> {
    std::fs::metadata(part_path(dest)).ok().map(|m| m.len()).filter(|&n| n > 0)
}

fn hash_existing(part: &Path, len: u64, hasher: &mut Sha256, progress: &dyn Progress) -> Result<u64> {
    let mut f = std::fs::File::open(part).map_err(|e| MlError::io(part.display(), e))?;
    let mut buf = vec![0u8; CHUNK];
    let mut done = 0u64;
    while done < len {
        progress.check()?;
        let n = f.read(&mut buf).map_err(|e| MlError::io(part.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(buf.get(..n).unwrap_or_default());
        done += n as u64;
    }
    Ok(done)
}

fn fetch(agent: &ureq::Agent, req: &DownloadRequest, part: &Path, have: &mut u64, hasher: &mut Sha256, name: &str, progress: &dyn Progress) -> Result<()> {
    let mut r = agent.get(&req.url);
    if *have > 0 {
        r = r.header("Range", format!("bytes={have}-"));
    }
    if let Some(key) = &req.bearer {
        r = r.header("Authorization", format!("Bearer {key}"));
    }
    let resp = r.call().map_err(|e| MlError::Network(e.to_string()))?;
    let status = resp.status().as_u16();
    match status {
        206 => {}
        200 => {
            if *have > 0 {
                // The server ignored the range: start over.
                *have = 0;
                *hasher = Sha256::new();
            }
        }
        416 if *have == req.size => return Ok(()),
        416 => {
            let _ = std::fs::remove_file(part);
            *have = 0;
            *hasher = Sha256::new();
            return Err(MlError::Network("the partial download didn't match the server's file; starting over".into()));
        }
        401 | 403 if req.bearer.is_some() => return Err(MlError::Network(format!("{name}: the server rejected the API key; check it in the Model Manager"))),
        401 | 403 => return Err(MlError::Network(format!("{name}: this download needs an API key; add one in the Model Manager"))),
        404 | 410 => return Err(MlError::NotFound(format!("{name} is no longer at {}", req.url))),
        429 => return Err(MlError::Network("the server is rate limiting downloads; try again later".into())),
        s => return Err(MlError::Network(format!("{name}: server answered HTTP {s}"))),
    }
    let html = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.starts_with("text/html"));
    if html {
        return Err(MlError::Network(format!("{name}: the server sent a web page instead of the file (a sign-in page?); check the API key")));
    }
    let mut f = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(part).map_err(|e| MlError::io(part.display(), e))?;
    f.set_len(*have).map_err(|e| MlError::io(part.display(), e))?;
    f.seek(SeekFrom::Start(*have)).map_err(|e| MlError::io(part.display(), e))?;
    let mut body = resp.into_body().into_with_config().limit(u64::MAX).reader();
    let mut buf = vec![0u8; CHUNK];
    let mut last = Instant::now();
    let mut speed_mark = (Instant::now(), *have);
    let mut rate = 0.0f64;
    loop {
        if progress.cancelled() {
            let _ = f.flush();
            return Err(MlError::Cancelled);
        }
        let n = body.read(&mut buf).map_err(|e| MlError::Network(e.to_string()))?;
        if n == 0 {
            break;
        }
        let chunk = buf.get(..n).unwrap_or_default();
        if have.saturating_add(n as u64) > req.size {
            drop(f);
            let _ = std::fs::remove_file(part);
            return Err(MlError::HashMismatch { file: name.to_string(), expected: format!("{} bytes", req.size), actual: "a larger file".into() });
        }
        f.write_all(chunk).map_err(|e| MlError::io(part.display(), e))?;
        hasher.update(chunk);
        *have += n as u64;
        if last.elapsed() >= Duration::from_millis(250) {
            let dt = speed_mark.0.elapsed().as_secs_f64();
            if dt >= 1.0 {
                rate = (*have - speed_mark.1) as f64 / dt;
                speed_mark = (Instant::now(), *have);
            }
            let frac = (*have as f64 / req.size.max(1) as f64) as f32;
            let speed = if rate > 0.0 { format!(", {}/s", human(rate as u64)) } else { String::new() };
            progress.report(frac, &format!("Downloading {name}: {} of {}{speed}", human(*have), human(req.size)));
            last = Instant::now();
        }
    }
    f.sync_all().map_err(|e| MlError::io(part.display(), e))?;
    Ok(())
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `1.2 GB`-style size.
pub fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} KB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A tiny HTTP server honouring `Range`, that can cut the first response short.
    fn serve(data: Arc<Vec<u8>>, cut_first_at: Option<usize>, auth: Option<&'static str>) -> (String, Arc<AtomicUsize>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.safetensors", l.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let h2 = hits.clone();
        std::thread::spawn(move || {
            for stream in l.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut req = Vec::new();
                let mut b = [0u8; 1];
                while !req.ends_with(b"\r\n\r\n") && s.read(&mut b).unwrap_or(0) == 1 {
                    req.push(b[0]);
                }
                let text = String::from_utf8_lossy(&req).to_lowercase();
                let n = h2.fetch_add(1, Ordering::SeqCst);
                if let Some(a) = auth
                    && !text.contains(&format!("authorization: bearer {a}"))
                {
                    let _ = s.write_all(b"HTTP/1.1 401 Unauthorized\r\ncontent-length: 0\r\n\r\n");
                    continue;
                }
                let start: usize = text.split("range: bytes=").nth(1).and_then(|r| r.split('-').next()).and_then(|v| v.trim().parse().ok()).unwrap_or(0);
                let body = &data[start.min(data.len())..];
                let status = if start > 0 { "206 Partial Content" } else { "200 OK" };
                let _ = s.write_all(format!("HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", body.len()).as_bytes());
                let send = match cut_first_at {
                    Some(c) if n == 0 => &body[..c.min(body.len())],
                    _ => body,
                };
                let _ = s.write_all(send);
            }
        });
        (url, hits)
    }

    struct Flag(AtomicBool);
    impl Progress for Flag {
        fn report(&self, _: f32, _: &str) {}
        fn cancelled(&self) -> bool {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("photoka-dl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn payload() -> Arc<Vec<u8>> {
        Arc::new((0..3_000_000u32).map(|i| (i * 7 % 251) as u8).collect())
    }

    #[test]
    fn downloads_resumes_and_verifies() {
        let data = payload();
        let sha = hex(&Sha256::digest(data.as_slice()));
        let (url, hits) = serve(data.clone(), Some(1_000_000), None);
        let d = dir("ok");
        let req = DownloadRequest { url, dest: d.join("m.safetensors"), size: data.len() as u64, sha256: sha, bearer: None };
        download(&req, &crate::NoProgress).unwrap();
        assert_eq!(std::fs::read(&req.dest).unwrap(), *data);
        assert!(hits.load(Ordering::SeqCst) >= 2, "the cut-off first response was resumed");
        assert!(!part_path(&req.dest).exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hash_mismatch_deletes_the_partial_file() {
        let data = payload();
        let (url, _) = serve(data.clone(), None, None);
        let d = dir("bad");
        let req = DownloadRequest { url, dest: d.join("m.safetensors"), size: data.len() as u64, sha256: "0".repeat(64), bearer: None };
        assert!(matches!(download(&req, &crate::NoProgress), Err(MlError::HashMismatch { .. })));
        assert!(!req.dest.exists() && !part_path(&req.dest).exists());
        let short = DownloadRequest { size: 10, ..req.clone() };
        assert!(download(&short, &crate::NoProgress).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn api_key_and_cancel() {
        let data = payload();
        let sha = hex(&Sha256::digest(data.as_slice()));
        let (url, _) = serve(data.clone(), None, Some("secret"));
        let d = dir("key");
        let mut req = DownloadRequest { url, dest: d.join("m.safetensors"), size: data.len() as u64, sha256: sha, bearer: None };
        assert!(download(&req, &crate::NoProgress).unwrap_err().to_string().contains("API key"));
        req.bearer = Some("wrong".into());
        assert!(download(&req, &crate::NoProgress).unwrap_err().to_string().contains("rejected"));
        req.bearer = Some("secret".into());
        assert!(matches!(download(&req, &Flag(AtomicBool::new(true))), Err(MlError::Cancelled)));
        download(&req, &crate::NoProgress).unwrap();
        assert_eq!(std::fs::metadata(&req.dest).unwrap().len(), data.len() as u64);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn unreachable_server_is_an_error() {
        let d = dir("down");
        let req = DownloadRequest { url: "http://127.0.0.1:9/x".into(), dest: d.join("x.safetensors"), size: 5, sha256: "0".repeat(64), bearer: None };
        assert!(download(&req, &crate::NoProgress).is_err());
        assert_eq!(human(2_132_696_762), "2.1 GB");
        let _ = std::fs::remove_dir_all(&d);
    }
}
