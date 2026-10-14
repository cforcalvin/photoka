//! RGBA8 images and 8-bit masks exchanged with the engines, plus the small amount of pixel work
//! generation needs (resampling, PNG for ComfyUI).

use crate::{MlError, Result};

/// Largest side the engines accept (SD 1.5 is trained at 512; 2048 is already 16× the pixels).
pub const MAX_SIDE: u32 = 2048;

/// Straight (not premultiplied) RGBA, 8 bits per channel, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

fn check_size(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(MlError::BadParams(format!("image size {width}×{height} is outside 1…{MAX_SIDE}")));
    }
    Ok(width as usize * height as usize)
}

impl Rgba8 {
    /// Transparent black.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let n = check_size(width, height)?;
        Ok(Rgba8 { width, height, data: vec![0; n * 4] })
    }

    pub fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let n = check_size(width, height)?;
        if data.len() != n * 4 {
            return Err(MlError::BadParams(format!("RGBA data is {} bytes, expected {}", data.len(), n * 4)));
        }
        Ok(Rgba8 { width, height, data })
    }

    /// Bilinear resample to `width`×`height`.
    pub fn resized(&self, width: u32, height: u32) -> Result<Rgba8> {
        if (width, height) == (self.width, self.height) {
            return Ok(self.clone());
        }
        let data = resample(&self.data, self.width, self.height, 4, width, height)?;
        Rgba8::from_raw(width, height, data)
    }

    /// PNG bytes (for the ComfyUI upload).
    pub fn to_png(&self) -> Result<Vec<u8>> {
        let mut img = photoka_codecs::Image::new(self.width, self.height, photoka_codecs::ChannelLayout::Rgba, photoka_codecs::SampleType::U8)
            .map_err(|e| MlError::Other(format!("PNG: {e}")))?;
        let dst = img.data_mut();
        if dst.len() != self.data.len() {
            return Err(MlError::Other("PNG: buffer size mismatch".into()));
        }
        dst.copy_from_slice(&self.data);
        photoka_codecs::encode(&img, photoka_codecs::Format::Png, &photoka_codecs::EncodeOptions::default()).map_err(|e| MlError::Other(format!("PNG: {e}")))
    }

    /// Decode any image the codecs read (ComfyUI returns PNG) into RGBA8.
    pub fn decode(bytes: &[u8]) -> Result<Rgba8> {
        let img = photoka_codecs::decode(bytes).map_err(|e| MlError::Other(format!("could not decode the generated image: {e}")))?;
        Rgba8::from_raw(img.width(), img.height(), img.to_rgba8())
    }
}

/// 8-bit mask: 255 = regenerate, 0 = keep. Same size as the image it masks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Mask {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let n = check_size(width, height)?;
        Ok(Mask { width, height, data: vec![0; n] })
    }

    pub fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let n = check_size(width, height)?;
        if data.len() != n {
            return Err(MlError::BadParams(format!("mask data is {} bytes, expected {n}", data.len())));
        }
        Ok(Mask { width, height, data })
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&v| v == 0)
    }

    pub fn resized(&self, width: u32, height: u32) -> Result<Mask> {
        if (width, height) == (self.width, self.height) {
            return Ok(self.clone());
        }
        let data = resample(&self.data, self.width, self.height, 1, width, height)?;
        Mask::from_raw(width, height, data)
    }

    /// Grow the mask by `radius` pixels (max filter), so inpainting blends past the stroke edge.
    pub fn dilated(&self, radius: u32) -> Mask {
        if radius == 0 {
            return self.clone();
        }
        let (w, h, r) = (self.width as usize, self.height as usize, radius as usize);
        let mut tmp = vec![0u8; self.data.len()];
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
                tmp[y * w + x] = self.data[y * w + a..=y * w + b].iter().copied().max().unwrap_or(0);
            }
        }
        let mut out = vec![0u8; self.data.len()];
        for y in 0..h {
            let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
            for x in 0..w {
                out[y * w + x] = (a..=b).map(|yy| tmp[yy * w + x]).max().unwrap_or(0);
            }
        }
        Mask { width: self.width, height: self.height, data: out }
    }

    /// Soften the edge over about `radius` pixels (two box blurs), for the result's layer mask.
    pub fn feathered(&self, radius: u32) -> Mask {
        if radius == 0 {
            return self.clone();
        }
        let (w, h) = (self.width as usize, self.height as usize);
        let r = (radius as usize / 2).max(1);
        let mut v: Vec<u32> = self.data.iter().map(|&x| u32::from(x)).collect();
        for _ in 0..2 {
            v = box_blur(&v, w, h, r, true);
            v = box_blur(&v, w, h, r, false);
        }
        Mask { width: self.width, height: self.height, data: v.into_iter().map(|x| x.min(255) as u8).collect() }
    }

    /// PNG with the mask in the alpha channel: ComfyUI's `LoadImage` reads a mask from alpha,
    /// where *transparent* means "inpaint", so alpha = 255 - mask.
    pub fn to_png_alpha(&self) -> Result<Vec<u8>> {
        let mut rgba = Vec::with_capacity(self.data.len() * 4);
        for &m in &self.data {
            rgba.extend_from_slice(&[0, 0, 0, 255 - m]);
        }
        Rgba8::from_raw(self.width, self.height, rgba)?.to_png()
    }
}

/// The mask generation actually uses: the user's mask plus every transparent pixel of the init
/// image (outpainting into empty canvas). `None` when nothing needs regenerating selectively.
pub fn effective_mask(init: &Rgba8, user: Option<&Mask>) -> Result<Option<Mask>> {
    let mut m = match user {
        Some(u) if (u.width, u.height) != (init.width, init.height) => u.resized(init.width, init.height)?,
        Some(u) => u.clone(),
        None => Mask::new(init.width, init.height)?,
    };
    for (v, px) in m.data.iter_mut().zip(init.data.chunks_exact(4)) {
        let empty = 255 - px[3];
        *v = (*v).max(empty);
    }
    Ok(if m.is_empty() { None } else { Some(m) })
}

/// Bilinear resample of `channels`-interleaved 8-bit data.
/// One box-blur pass of radius `r` along rows (`horizontal`) or columns; edges clamp.
fn box_blur(v: &[u32], w: usize, h: usize, r: usize, horizontal: bool) -> Vec<u32> {
    let mut out = vec![0u32; v.len()];
    let (lines, len) = if horizontal { (h, w) } else { (w, h) };
    let at = |line: usize, i: usize| if horizontal { line * w + i } else { i * w + line };
    let n = (2 * r + 1) as u32;
    for line in 0..lines {
        let get = |i: isize| v.get(at(line, i.clamp(0, len as isize - 1) as usize)).copied().unwrap_or(0);
        let mut sum: u32 = (-(r as isize)..=r as isize).map(get).sum();
        for i in 0..len {
            if let Some(o) = out.get_mut(at(line, i)) {
                *o = (sum + n / 2) / n;
            }
            sum = sum + get(i as isize + r as isize + 1) - get(i as isize - r as isize);
        }
    }
    out
}

fn resample(src: &[u8], sw: u32, sh: u32, channels: usize, dw: u32, dh: u32) -> Result<Vec<u8>> {
    let n = check_size(dw, dh)?;
    check_size(sw, sh)?;
    let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
    if src.len() != sw * sh * channels {
        return Err(MlError::BadParams("resample: source buffer size mismatch".into()));
    }
    let mut out = vec![0u8; n * channels];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(sh - 1);
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(sw - 1);
            for c in 0..channels {
                let p = |xx: usize, yy: usize| src[(yy * sw + xx) * channels + c] as f32;
                let top = p(x0, y0) * (1.0 - tx) + p(x1, y0) * tx;
                let bot = p(x0, y1) * (1.0 - tx) + p(x1, y1) * tx;
                out[(y * dw + x) * channels + c] = (top * (1.0 - ty) + bot * ty).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_checked() {
        assert!(Rgba8::new(0, 4).is_err());
        assert!(Rgba8::new(4, MAX_SIDE + 1).is_err());
        assert!(Rgba8::from_raw(2, 2, vec![0; 15]).is_err());
        assert!(Mask::from_raw(2, 2, vec![0; 3]).is_err());
        assert!(Rgba8::new(4, 4).unwrap().resized(0, 3).is_err());
    }

    #[test]
    fn resample_keeps_flat_colour_and_png_round_trips() {
        let img = Rgba8::from_raw(3, 2, [10u8, 20, 30, 255].repeat(6)).unwrap();
        let big = img.resized(7, 5).unwrap();
        assert!(big.data.chunks(4).all(|p| p == [10, 20, 30, 255]));
        let back = Rgba8::decode(&big.to_png().unwrap()).unwrap();
        assert_eq!(back, big);
    }

    #[test]
    fn effective_mask_adds_transparent_pixels() {
        let mut init = Rgba8::from_raw(2, 1, vec![1, 2, 3, 255, 0, 0, 0, 0]).unwrap();
        let m = effective_mask(&init, None).unwrap().unwrap();
        assert_eq!(m.data, vec![0, 255]);
        init.data[7] = 255;
        assert!(effective_mask(&init, None).unwrap().is_none());
        let user = Mask::from_raw(2, 1, vec![200, 0]).unwrap();
        assert_eq!(effective_mask(&init, Some(&user)).unwrap().unwrap().data, vec![200, 0]);
    }

    #[test]
    fn feather_softens_the_edge_and_keeps_flat_areas() {
        let mut m = Mask::new(64, 8).unwrap();
        for y in 0..8 {
            for x in 32..64 {
                m.data[y * 64 + x] = 255;
            }
        }
        let f = m.feathered(8);
        assert_eq!(f.data[4 * 64], 0);
        assert_eq!(f.data[4 * 64 + 63], 255);
        let edge = f.data[4 * 64 + 32];
        assert!(edge > 60 && edge < 200, "{edge}");
        assert!(f.data[4 * 64 + 28] < edge && f.data[4 * 64 + 36] > edge);
        assert_eq!(Mask::new(1, 1).unwrap().feathered(16).data, vec![0]);
    }

    #[test]
    fn dilate_grows_the_mask() {
        let mut m = Mask::new(5, 5).unwrap();
        m.data[12] = 255;
        let d = m.dilated(1);
        assert_eq!(d.data.iter().filter(|&&v| v == 255).count(), 9);
        let png = m.to_png_alpha().unwrap();
        let back = Rgba8::decode(&png).unwrap();
        assert_eq!(back.data[12 * 4 + 3], 0);
        assert_eq!(back.data[3], 255);
    }
}
