//! A cost model for Neon recipes.
//!
//! The model is deliberately table-driven rather than curve-fitted. RustFFT has a finite and
//! small set of primitives (a couple of dozen butterflies, a few dozen valid Radix4 shapes), so
//! their costs can simply be measured and stored. That removes the extrapolation error that a
//! fitted closed form introduces, which is what made the 2021 scalar attempt mis-rank a direct
//! Radix4 against a split one.
//!
//! Only the composing algorithms need a fitted number, and each needs exactly one: the cost per
//! element of the transposes and twiddle multiplies they add on top of their inner FFTs.

use rustfft::tuning::Recipe;
use std::collections::HashMap;

/// Which composing algorithm a recipe is, for the purpose of attributing overhead.
pub fn kind(recipe: &Recipe) -> &'static str {
    match recipe {
        Recipe::MixedRadix { .. } => "mr",
        Recipe::MixedRadixSmall { .. } => "mrs",
        Recipe::GoodThomasAlgorithm { .. } => "gt",
        Recipe::GoodThomasAlgorithmSmall { .. } => "gts",
        Recipe::RadersAlgorithm { .. } => "rad",
        Recipe::BluesteinsAlgorithm { .. } => "bs",
        Recipe::Radix4 { .. } => "r4",
        Recipe::Dft(_) => "dft",
        _ => "butterfly",
    }
}

/// What an algorithm's per-element overhead scales with.
///
/// For most algorithms that is simply their own length. Bluestein's is the exception: its
/// pointwise multiply and zero-padding run over the padded inner length, which can be nearly
/// four times the outer length, so charging it per outer element makes the fitted coefficient
/// depend on how much padding a particular length happened to need.
pub fn overhead_scale(recipe: &Recipe) -> f64 {
    match recipe {
        Recipe::BluesteinsAlgorithm { inner_fft, .. } => inner_fft.len() as f64,
        other => other.len() as f64,
    }
}

pub fn children(recipe: &Recipe) -> Vec<&Recipe> {
    match recipe {
        Recipe::Radix4 { base_fft, .. } => vec![base_fft],
        Recipe::RadersAlgorithm { inner_fft } | Recipe::BluesteinsAlgorithm { inner_fft, .. } => {
            vec![inner_fft]
        }
        Recipe::MixedRadix {
            left_fft,
            right_fft,
        }
        | Recipe::MixedRadixSmall {
            left_fft,
            right_fft,
        }
        | Recipe::GoodThomasAlgorithm {
            left_fft,
            right_fft,
        }
        | Recipe::GoodThomasAlgorithmSmall {
            left_fft,
            right_fft,
        } => vec![left_fft, right_fft],
        _ => Vec::new(),
    }
}

/// Which log2 bucket a recipe's overhead belongs in.
///
/// Bucketed on the same quantity the overhead is charged per, so that Bluestein's is placed by
/// the padded inner length it actually works over.
pub fn bucket_of(recipe: &Recipe) -> u32 {
    overhead_scale(recipe).log2() as u32
}

#[derive(Default, Clone)]
pub struct Model {
    /// Measured nanoseconds for one FFT, by butterfly length.
    pub butterfly: HashMap<usize, f64>,
    /// Measured nanoseconds for one FFT, by (base length, k).
    pub radix4: HashMap<(usize, u32), f64>,
    /// Fitted nanoseconds per element of overhead, by algorithm kind, as a curve over log2 of
    /// the working set. A single constant is not enough: GoodThomas reindexes with a scatter
    /// rather than a transpose, and its per-element cost roughly triples once the array stops
    /// fitting in L1. Entries are sorted by bucket.
    pub overhead: HashMap<&'static str, Vec<(u32, f64)>>,
}

impl Model {
    /// Overhead per element for `kind` at a working set of `scale` elements, linearly
    /// interpolated between measured buckets and clamped outside the measured range.
    pub fn overhead_at(&self, kind: &str, scale: f64) -> Option<f64> {
        let table = self.overhead.get(kind)?;
        match table.len() {
            0 => None,
            1 => Some(table[0].1),
            _ => {
                let x = scale.log2();
                if x <= table[0].0 as f64 {
                    return Some(table[0].1);
                }
                if x >= table[table.len() - 1].0 as f64 {
                    return Some(table[table.len() - 1].1);
                }
                for pair in table.windows(2) {
                    let (lo_bucket, lo_value) = pair[0];
                    let (hi_bucket, hi_value) = pair[1];
                    if x <= hi_bucket as f64 {
                        let span = (hi_bucket - lo_bucket) as f64;
                        let t = if span > 0.0 {
                            (x - lo_bucket as f64) / span
                        } else {
                            0.0
                        };
                        return Some(lo_value + t * (hi_value - lo_value));
                    }
                }
                Some(table[table.len() - 1].1)
            }
        }
    }

    /// Estimated nanoseconds for one FFT of this recipe.
    ///
    /// Returns `None` if the recipe uses a primitive that was never measured, so that a missing
    /// table entry is a loud failure rather than a silently wrong ranking.
    pub fn cost(&self, recipe: &Recipe) -> Option<f64> {
        let len = recipe.len() as f64;
        Some(match recipe {
            Recipe::Dft(n) => {
                // Dft is only ever reached for degenerate sizes; a quadratic keeps it last.
                let n = *n as f64;
                100.0 * n * n
            }
            Recipe::Radix4 { k, base_fft } => {
                *self.radix4.get(&(base_fft.len(), *k))?
            }
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            }
            | Recipe::MixedRadixSmall {
                left_fft,
                right_fft,
            }
            | Recipe::GoodThomasAlgorithm {
                left_fft,
                right_fft,
            }
            | Recipe::GoodThomasAlgorithmSmall {
                left_fft,
                right_fft,
            } => {
                let left = self.cost(left_fft)?;
                let right = self.cost(right_fft)?;
                let scale = overhead_scale(recipe);
                right_fft.len() as f64 * left
                    + left_fft.len() as f64 * right
                    + self.overhead_at(kind(recipe), scale)? * scale
            }
            // Rader's runs its inner FFT twice per transform, once forward and once to
            // invert the convolution, exactly like Bluestein's.
            Recipe::RadersAlgorithm { inner_fft } => {
                2.0 * self.cost(inner_fft)? + self.overhead_at("rad", len)? * len
            }
            Recipe::BluesteinsAlgorithm { inner_fft, .. } => {
                let scale = overhead_scale(recipe);
                2.0 * self.cost(inner_fft)? + self.overhead_at("bs", scale)? * scale
            }
            butterfly => *self.butterfly.get(&butterfly.len())?,
        })
    }

    /// The sum of the inner FFT costs only, with no overhead for `recipe` itself.
    ///
    /// Subtracting this from a measurement is what isolates one algorithm's overhead.
    pub fn inner_cost(&self, recipe: &Recipe) -> Option<f64> {
        Some(match recipe {
            Recipe::MixedRadix {
                left_fft,
                right_fft,
            }
            | Recipe::MixedRadixSmall {
                left_fft,
                right_fft,
            }
            | Recipe::GoodThomasAlgorithm {
                left_fft,
                right_fft,
            }
            | Recipe::GoodThomasAlgorithmSmall {
                left_fft,
                right_fft,
            } => {
                right_fft.len() as f64 * self.cost(left_fft)?
                    + left_fft.len() as f64 * self.cost(right_fft)?
            }
            Recipe::RadersAlgorithm { inner_fft } => 2.0 * self.cost(inner_fft)?,
            Recipe::BluesteinsAlgorithm { inner_fft, .. } => 2.0 * self.cost(inner_fft)?,
            other => self.cost(other)?,
        })
    }

    /// True if every strict descendant of `recipe` is a primitive or an already-fitted kind.
    ///
    /// Overheads have to be fitted in dependency order, because the residual for a MixedRadix
    /// containing a MixedRadixSmall is only meaningful once the Small's own overhead is known.
    pub fn descendants_known(&self, recipe: &Recipe, fitted: &[&'static str]) -> bool {
        children(recipe).iter().all(|child| {
            let k = kind(child);
            let ok = matches!(k, "butterfly" | "r4" | "dft") || fitted.contains(&k);
            ok && self.descendants_known(child, fitted)
        })
    }

    pub fn describe(&self) -> String {
        let mut kinds: Vec<(&&str, &Vec<(u32, f64)>)> = self.overhead.iter().collect();
        kinds.sort_by_key(|(k, _)| **k);
        let mut out = format!(
            "{} butterflies, {} radix4 shapes measured\n",
            self.butterfly.len(),
            self.radix4.len()
        );
        out.push_str("overhead, ns per element, by working set:\n");
        for (kind, table) in kinds {
            let rendered: Vec<String> = table
                .iter()
                .map(|(bucket, value)| format!("{}:{:.2}", 1usize << bucket, value))
                .collect();
            out.push_str(&format!("  {:<5} {}\n", kind, rendered.join("  ")));
        }
        out
    }
}
