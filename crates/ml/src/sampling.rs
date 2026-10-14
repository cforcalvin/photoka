//! Noise schedules and sampler steps for Stable Diffusion, in k-diffusion's sigma form.
//!
//! The latent is `x = x0 + σ·ε`. Each step the model sees `x / sqrt(σ² + 1)` at the timestep
//! matching `σ`, predicts `ε`, and `denoised = x − σ·ε`. Every sampler here is then a linear
//! combination `x' = a·x + b·denoised + c·previous_denoised + d·noise`, so the tensor code in the
//! engines stays a few lines and all the maths is tested here on plain numbers.

use crate::params::Sampler;

pub const TRAIN_STEPS: usize = 1000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

/// SD 1.x/2.x training schedule ("scaled linear" betas).
#[derive(Clone, Debug)]
pub struct Schedule {
    alphas_cumprod: Vec<f64>,
    log_sigmas: Vec<f64>,
}

impl Default for Schedule {
    fn default() -> Self {
        Self::sd()
    }
}

impl Schedule {
    pub fn sd() -> Self {
        let (a, b) = (BETA_START.sqrt(), BETA_END.sqrt());
        let n = TRAIN_STEPS as f64 - 1.0;
        let mut prod = 1.0;
        let alphas_cumprod: Vec<f64> = (0..TRAIN_STEPS)
            .map(|i| {
                let beta = (a + (b - a) * i as f64 / n).powi(2);
                prod *= 1.0 - beta;
                prod
            })
            .collect();
        let log_sigmas = alphas_cumprod.iter().map(|ac| ((1.0 - ac) / ac).sqrt().ln()).collect();
        Schedule { alphas_cumprod, log_sigmas }
    }

    pub fn sigma_min(&self) -> f64 {
        self.log_sigmas.first().map_or(0.0292, |l| l.exp())
    }

    pub fn sigma_max(&self) -> f64 {
        self.log_sigmas.last().map_or(14.6146, |l| l.exp())
    }

    /// `ᾱ` at an integer timestep (clamped).
    pub fn alpha_cumprod(&self, t: usize) -> f64 {
        self.alphas_cumprod.get(t.min(TRAIN_STEPS - 1)).copied().unwrap_or(1.0)
    }

    /// σ at a (fractional) timestep, interpolating log σ.
    pub fn sigma_at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, TRAIN_STEPS as f64 - 1.0);
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(TRAIN_STEPS - 1);
        let w = t - lo as f64;
        let l = |i: usize| self.log_sigmas.get(i).copied().unwrap_or(0.0);
        ((1.0 - w) * l(lo) + w * l(hi)).exp()
    }

    /// The (fractional) timestep for σ: the model's time conditioning.
    pub fn timestep(&self, sigma: f64) -> f64 {
        if sigma <= 0.0 || !sigma.is_finite() {
            return 0.0;
        }
        let ls = sigma.ln();
        let idx = self.log_sigmas.partition_point(|&l| l <= ls);
        if idx == 0 {
            return 0.0;
        }
        if idx >= TRAIN_STEPS {
            return TRAIN_STEPS as f64 - 1.0;
        }
        let (lo, hi) = (self.log_sigmas.get(idx - 1).copied().unwrap_or(0.0), self.log_sigmas.get(idx).copied().unwrap_or(0.0));
        let w = if hi > lo { (ls - lo) / (hi - lo) } else { 0.0 };
        (idx - 1) as f64 + w
    }

    /// `σ` values for `steps` steps, descending, ending with 0 (length `steps + 1`).
    pub fn sigmas(&self, kind: ScheduleKind, steps: usize) -> Vec<f64> {
        let steps = steps.clamp(1, TRAIN_STEPS);
        let last = TRAIN_STEPS as f64 - 1.0;
        let mut s: Vec<f64> = match kind {
            ScheduleKind::Normal => {
                if steps == 1 {
                    vec![self.sigma_at(last)]
                } else {
                    (0..steps).map(|i| self.sigma_at(last - last * i as f64 / (steps - 1) as f64)).collect()
                }
            }
            ScheduleKind::Karras => {
                let rho = 7.0;
                let (lo, hi) = (self.sigma_min().powf(1.0 / rho), self.sigma_max().powf(1.0 / rho));
                (0..steps)
                    .map(|i| {
                        let r = if steps == 1 { 0.0 } else { i as f64 / (steps - 1) as f64 };
                        (hi + r * (lo - hi)).powf(rho)
                    })
                    .collect()
            }
            ScheduleKind::DdimUniform => {
                let c = TRAIN_STEPS / steps;
                let mut ts: Vec<usize> = (0..steps).map(|i| 1 + i * c.max(1)).filter(|&t| t < TRAIN_STEPS).collect();
                ts.reverse();
                ts.into_iter().map(|t| self.sigma_at(t as f64)).collect()
            }
            ScheduleKind::Trailing => (0..steps).map(|i| self.sigma_at((TRAIN_STEPS as f64 * (1.0 - i as f64 / steps as f64)).round() - 1.0)).collect(),
            ScheduleKind::Lcm => {
                const ORIGIN: usize = 50;
                let c = TRAIN_STEPS / ORIGIN;
                let origin: Vec<usize> = (1..=ORIGIN).map(|i| i * c - 1).collect();
                let skip = (ORIGIN / steps.min(ORIGIN)).max(1);
                origin.iter().rev().step_by(skip).take(steps).map(|&t| self.sigma_at(t as f64)).collect()
            }
        };
        s.push(0.0);
        s
    }
}

/// How steps are spaced over the training schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleKind {
    /// Evenly spaced timesteps (ComfyUI "normal").
    Normal,
    /// Karras et al. 2022 (ρ = 7).
    Karras,
    /// DDIM's uniform timesteps.
    DdimUniform,
    /// Trailing timesteps, starting at the last training step (SD-Turbo).
    Trailing,
    /// Latent Consistency Model timesteps (50 original steps).
    Lcm,
}

impl ScheduleKind {
    pub fn for_sampler(sampler: Sampler, turbo: bool) -> Self {
        match sampler {
            Sampler::Lcm => ScheduleKind::Lcm,
            _ if turbo => ScheduleKind::Trailing,
            Sampler::Dpmpp2mKarras => ScheduleKind::Karras,
            Sampler::Ddim => ScheduleKind::DdimUniform,
            Sampler::Euler | Sampler::EulerA | Sampler::Dpmpp2m => ScheduleKind::Normal,
        }
    }
}

/// `x' = x·x + denoised·denoised + old·old_denoised + noise·N(0,1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepCoeffs {
    pub x: f64,
    pub denoised: f64,
    pub old: f64,
    pub noise: f64,
}

/// Input scaling for the model at σ.
pub fn c_in(sigma: f64) -> f64 {
    1.0 / (sigma * sigma + 1.0).sqrt()
}

/// Coefficients for step `i` (from `sigmas[i]` to `sigmas[i + 1]`).
pub fn step(sampler: Sampler, sigmas: &[f64], i: usize) -> StepCoeffs {
    let s = sigmas.get(i).copied().unwrap_or(0.0);
    let sn = sigmas.get(i + 1).copied().unwrap_or(0.0);
    let to_denoised = StepCoeffs { x: 0.0, denoised: 1.0, old: 0.0, noise: 0.0 };
    if s <= 0.0 {
        return to_denoised;
    }
    match sampler {
        Sampler::Euler | Sampler::Ddim => StepCoeffs { x: sn / s, denoised: 1.0 - sn / s, old: 0.0, noise: 0.0 },
        Sampler::EulerA => {
            let up = if sn > 0.0 { (sn * sn * (s * s - sn * sn) / (s * s)).max(0.0).sqrt().min(sn) } else { 0.0 };
            let down = (sn * sn - up * up).max(0.0).sqrt();
            StepCoeffs { x: down / s, denoised: 1.0 - down / s, old: 0.0, noise: up }
        }
        Sampler::Dpmpp2m | Sampler::Dpmpp2mKarras => {
            if sn <= 0.0 {
                return to_denoised;
            }
            let ratio = sn / s;
            let base = 1.0 - ratio;
            let prev = if i == 0 { None } else { sigmas.get(i - 1).copied() };
            match prev {
                Some(sp) if sp > s => {
                    let h = (s / sn).ln();
                    let h_last = (sp / s).ln();
                    let r = h_last / h;
                    StepCoeffs { x: ratio, denoised: base * (1.0 + 1.0 / (2.0 * r)), old: -base / (2.0 * r), noise: 0.0 }
                }
                _ => StepCoeffs { x: ratio, denoised: base, old: 0.0, noise: 0.0 },
            }
        }
        Sampler::Lcm => StepCoeffs { x: 0.0, denoised: 1.0, old: 0.0, noise: sn },
    }
}

/// First step index for img2img at `strength` (1 = from pure noise, 0 = keep the image).
/// Always leaves at least one step.
pub fn img2img_start(steps: usize, strength: f32) -> usize {
    let keep = ((steps as f64) * f64::from(strength.clamp(0.0, 1.0))).round() as usize;
    steps.saturating_sub(keep.max(1))
}

/// Deterministic standard-normal noise: same seed, same image on every run.
pub fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1 = ((next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = ((next() >> 11) as f64) / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f64::consts::TAU * u2;
        out.push((r * th.cos()) as f32);
        out.push((r * th.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Seed for the ancestral noise of step `i`, distinct from the initial noise.
pub fn step_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407 ^ (i as u64 + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_sd() {
        let s = Schedule::sd();
        assert!((s.sigma_max() - 14.6146).abs() < 1e-3, "{}", s.sigma_max());
        assert!((s.sigma_min() - 0.0292).abs() < 1e-3, "{}", s.sigma_min());
        for t in [0.0, 1.0, 250.5, 999.0] {
            assert!((s.timestep(s.sigma_at(t)) - t).abs() < 1e-6);
        }
        assert_eq!(s.timestep(0.0), 0.0);
        assert_eq!(s.timestep(1e9), 999.0);
    }

    #[test]
    fn schedules_are_descending_and_end_at_zero() {
        let s = Schedule::sd();
        for kind in [ScheduleKind::Normal, ScheduleKind::Karras, ScheduleKind::DdimUniform, ScheduleKind::Trailing, ScheduleKind::Lcm] {
            for steps in [1, 2, 4, 20, 50, 150] {
                let sig = s.sigmas(kind, steps);
                assert_eq!(sig.last(), Some(&0.0));
                assert!(sig.windows(2).all(|w| w[0] > w[1]), "{kind:?} {steps}: {sig:?}");
                if kind != ScheduleKind::DdimUniform && kind != ScheduleKind::Lcm {
                    assert_eq!(sig.len(), steps + 1, "{kind:?}");
                }
            }
        }
        let turbo = s.sigmas(ScheduleKind::Trailing, 4);
        assert!((s.timestep(turbo[1]) - 749.0).abs() < 1e-6);
        let lcm = s.sigmas(ScheduleKind::Lcm, 4);
        let ts: Vec<f64> = lcm[..4].iter().map(|&x| s.timestep(x).round()).collect();
        assert_eq!(ts, vec![999.0, 759.0, 519.0, 279.0]);
    }

    /// With a perfect denoiser and no noise, every sampler must land exactly on x0.
    #[test]
    fn samplers_converge_with_an_exact_denoiser() {
        let s = Schedule::sd();
        for sampler in Sampler::ALL {
            let sig = s.sigmas(ScheduleKind::for_sampler(sampler, false), 12);
            let x0 = 0.37;
            let mut x = x0 + sig[0] * 1.3;
            let mut old = 0.0;
            for i in 0..sig.len() - 1 {
                let c = step(sampler, &sig, i);
                let denoised = x0;
                x = c.x * x + c.denoised * denoised + c.old * old;
                old = denoised;
            }
            assert!((x - x0).abs() < 1e-9, "{sampler:?}: {x}");
        }
    }

    #[test]
    fn img2img_start_and_noise() {
        assert_eq!(img2img_start(20, 1.0), 0);
        assert_eq!(img2img_start(20, 0.5), 10);
        assert_eq!(img2img_start(20, 0.0), 19);
        assert_eq!(img2img_start(20, f32::NAN), 19);
        let a = gaussian(42, 10_001);
        assert_eq!(a, gaussian(42, 10_001));
        assert_ne!(a[..10], gaussian(43, 10)[..]);
        let mean = a.iter().map(|&v| f64::from(v)).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.05, "{mean} {var}");
        assert_ne!(step_seed(1, 0), step_seed(1, 1));
    }
}
