//! Support code for tuning the Neon planner.
//!
//! This module is compiled only when the non-default `tuning` feature is enabled, and is not
//! part of the public API. It exists so that measurement tools can build, name, and enumerate
//! the exact `Recipe` values the planner works with, rather than reimplementing construction
//! and risking measuring something the planner would never actually build.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use num_integer::gcd;

pub use crate::neon::neon_planner::Recipe;
use crate::neon::neon_prime_butterflies::prime_butterfly_lens;
use crate::{Fft, FftDirection, FftNum, FftPlannerNeon};

/// Butterfly sizes the Neon planner has dedicated kernels for, excluding the prime butterflies.
const NEON_BUTTERFLIES: [usize; 14] = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 24, 32];

/// Bases that `Radix4` can be built on, ie the sizes that leave a power of four behind.
const RADIX4_BASES: [usize; 10] = [1, 2, 4, 8, 16, 32, 3, 6, 12, 24];

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

/// Render a recipe in the spec syntax accepted by [`parse`].
pub fn to_spec(recipe: &Recipe) -> String {
    match recipe {
        Recipe::Dft(len) => format!("dft({})", len),
        Recipe::Radix4 { k, base_fft } => format!("r4({},{})", k, to_spec(base_fft)),
        Recipe::MixedRadix {
            left_fft,
            right_fft,
        } => format!("mr({},{})", to_spec(left_fft), to_spec(right_fft)),
        Recipe::MixedRadixSmall {
            left_fft,
            right_fft,
        } => format!("mrs({},{})", to_spec(left_fft), to_spec(right_fft)),
        Recipe::GoodThomasAlgorithm {
            left_fft,
            right_fft,
        } => format!("gt({},{})", to_spec(left_fft), to_spec(right_fft)),
        Recipe::GoodThomasAlgorithmSmall {
            left_fft,
            right_fft,
        } => format!("gts({},{})", to_spec(left_fft), to_spec(right_fft)),
        Recipe::RadersAlgorithm { inner_fft } => format!("rad({})", to_spec(inner_fft)),
        Recipe::BluesteinsAlgorithm { len, inner_fft } => {
            format!("bs({},{})", len, to_spec(inner_fft))
        }
        other => format!("b{}", other.len()),
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

struct Parser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.s[self.pos..].chars().next()
    }

    fn eat(&mut self, c: char) -> Result<(), String> {
        match self.peek() {
            Some(got) if got == c => {
                self.pos += got.len_utf8();
                Ok(())
            }
            other => Err(format!(
                "expected '{}' at offset {}, found {:?}",
                c, self.pos, other
            )),
        }
    }

    fn ident(&mut self) -> String {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric()) {
            self.pos += 1;
        }
        self.s[start..self.pos].to_string()
    }

    fn number(&mut self) -> Result<usize, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        self.s[start..self.pos]
            .parse()
            .map_err(|_| format!("expected a number at offset {}", start))
    }

    fn recipe(&mut self) -> Result<Arc<Recipe>, String> {
        let name = self.ident();
        if name.is_empty() {
            return Err(format!("expected a recipe at offset {}", self.pos));
        }
        if let Some(size) = name.strip_prefix('b') {
            if let Ok(size) = size.parse::<usize>() {
                return butterfly(size);
            }
        }

        self.eat('(')?;
        let result = match name.as_str() {
            "dft" => Arc::new(Recipe::Dft(self.number()?)),
            "r4" => {
                let k = self.number()? as u32;
                self.eat(',')?;
                let base_fft = self.recipe()?;
                Arc::new(Recipe::Radix4 { k, base_fft })
            }
            "rad" => Arc::new(Recipe::RadersAlgorithm {
                inner_fft: self.recipe()?,
            }),
            "bs" => {
                let len = self.number()?;
                self.eat(',')?;
                let inner_fft = self.recipe()?;
                Arc::new(Recipe::BluesteinsAlgorithm { len, inner_fft })
            }
            "mr" | "mrs" | "gt" | "gts" => {
                let left_fft = self.recipe()?;
                self.eat(',')?;
                let right_fft = self.recipe()?;
                Arc::new(match name.as_str() {
                    "mr" => Recipe::MixedRadix {
                        left_fft,
                        right_fft,
                    },
                    "mrs" => Recipe::MixedRadixSmall {
                        left_fft,
                        right_fft,
                    },
                    "gt" => Recipe::GoodThomasAlgorithm {
                        left_fft,
                        right_fft,
                    },
                    _ => Recipe::GoodThomasAlgorithmSmall {
                        left_fft,
                        right_fft,
                    },
                })
            }
            other => return Err(format!("unknown recipe '{}'", other)),
        };
        self.eat(')')?;
        Ok(result)
    }
}

fn butterfly(size: usize) -> Result<Arc<Recipe>, String> {
    Ok(Arc::new(match size {
        1 => Recipe::Butterfly1,
        2 => Recipe::Butterfly2,
        3 => Recipe::Butterfly3,
        4 => Recipe::Butterfly4,
        5 => Recipe::Butterfly5,
        6 => Recipe::Butterfly6,
        8 => Recipe::Butterfly8,
        9 => Recipe::Butterfly9,
        10 => Recipe::Butterfly10,
        12 => Recipe::Butterfly12,
        15 => Recipe::Butterfly15,
        16 => Recipe::Butterfly16,
        24 => Recipe::Butterfly24,
        32 => Recipe::Butterfly32,
        len if prime_butterfly_lens().contains(&len) => Recipe::PrimeButterfly { len },
        other => return Err(format!("no Neon butterfly of size {}", other)),
    }))
}

/// Parse a recipe spec, eg `mr(r4(2,b16),b12)`.
pub fn parse(spec: &str) -> Result<Arc<Recipe>, String> {
    let cleaned: String = spec.chars().filter(|c| !c.is_whitespace()).collect();
    let mut parser = Parser {
        s: &cleaned,
        pos: 0,
    };
    let recipe = parser.recipe()?;
    if parser.pos != cleaned.len() {
        return Err(format!("trailing junk at offset {}", parser.pos));
    }
    Ok(recipe)
}

// ---------------------------------------------------------------------------
// Building and enumerating
// ---------------------------------------------------------------------------

/// Walks a recipe and fails if one length appears with two different recipes.
///
/// The planner's algorithm cache is keyed on length alone, so a tree containing two different
/// recipes of the same length would silently build the same FFT twice and quietly invalidate
/// whatever we measured. Every recipe goes through this check before it is built.
fn check_unambiguous(recipe: &Recipe, seen: &mut HashMap<usize, String>) -> Result<(), String> {
    let spec = to_spec(recipe);
    if let Some(previous) = seen.insert(recipe.len(), spec.clone()) {
        if previous != spec {
            return Err(format!(
                "length {} appears as both '{}' and '{}'; the algorithm cache cannot hold both",
                recipe.len(),
                previous,
                spec
            ));
        }
    }
    let children: Vec<&Arc<Recipe>> = match recipe {
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
    };
    for child in children {
        check_unambiguous(child, seen)?;
    }
    Ok(())
}

pub struct NeonTuner<T: FftNum> {
    planner: FftPlannerNeon<T>,
}

impl<T: FftNum> NeonTuner<T> {
    pub fn new() -> Self {
        Self {
            planner: FftPlannerNeon::new().expect("this machine does not support Neon"),
        }
    }

    /// The recipe the shipping Neon planner picks for `len`.
    pub fn plan(&mut self, len: usize) -> Arc<Recipe> {
        self.planner.design_fft_for_len(len)
    }

    /// Build an FFT for an arbitrary recipe, on a planner of its own so nothing is shared.
    pub fn build(&mut self, recipe: &Recipe, direction: FftDirection) -> Arc<dyn Fft<T>> {
        check_unambiguous(recipe, &mut HashMap::new()).expect("ambiguous recipe");
        let mut planner = FftPlannerNeon::<T>::new().expect("this machine does not support Neon");
        planner.build_fft(recipe, direction)
    }

    /// Plausible alternatives to the planner's choice for `len`, the planner's own pick first,
    /// trimmed to at most `cap` entries.
    ///
    /// A length like 100800 has hundreds of two-way splits, so a survey has to trim. Everything
    /// structural is kept (the planner's pick, Radix4 variants, Rader's and Bluestein's) and it
    /// is only the splits that get dropped, most lopsided first, on the grounds that a split
    /// with a tiny side is mostly just its large side plus a transpose.
    pub fn candidates_capped(&mut self, len: usize, cap: usize) -> Vec<Arc<Recipe>> {
        let all = self.candidates(len);
        if all.len() <= cap {
            return all;
        }

        let balance = |recipe: &Recipe| -> f64 {
            match recipe {
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
                } => (left_fft.len() as f64).ln() - (right_fft.len() as f64).ln(),
                // not a split, so never trimmed
                _ => f64::NEG_INFINITY,
            }
        };

        let mut kept: Vec<Arc<Recipe>> = Vec::with_capacity(cap);
        let mut splits: Vec<Arc<Recipe>> = Vec::new();
        for (index, recipe) in all.into_iter().enumerate() {
            if index == 0 || balance(&recipe) == f64::NEG_INFINITY {
                kept.push(recipe);
            } else {
                splits.push(recipe);
            }
        }
        splits.sort_by(|a, b| {
            balance(a)
                .abs()
                .partial_cmp(&balance(b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kept.extend(splits.into_iter().take(cap.saturating_sub(kept.len())));
        kept
    }

    /// Every candidate, uncapped.
    pub fn candidates(&mut self, len: usize) -> Vec<Arc<Recipe>> {
        let mut out: Vec<Arc<Recipe>> = vec![self.plan(len)];
        let mut seen: Vec<String> = vec![to_spec(&out[0])];
        let push = |recipe: Arc<Recipe>, out: &mut Vec<Arc<Recipe>>, seen: &mut Vec<String>| {
            debug_assert_eq!(recipe.len(), len);
            let spec = to_spec(&recipe);
            if !seen.contains(&spec) && check_unambiguous(&recipe, &mut HashMap::new()).is_ok() {
                seen.push(spec);
                out.push(recipe);
            }
        };

        // Every two-way split, in both orders, as each algorithm that can express it.
        for left_len in 2..=(len / 2) {
            if len % left_len != 0 {
                continue;
            }
            let right_len = len / left_len;
            let left_fft = self.plan(left_len);
            let right_fft = self.plan(right_len);
            let coprime = gcd(left_len, right_len) == 1;
            let small = left_len < 33 && right_len < 33;

            for (left_fft, right_fft) in [
                (Arc::clone(&left_fft), Arc::clone(&right_fft)),
                (right_fft, left_fft),
            ] {
                push(
                    Arc::new(Recipe::MixedRadix {
                        left_fft: Arc::clone(&left_fft),
                        right_fft: Arc::clone(&right_fft),
                    }),
                    &mut out,
                    &mut seen,
                );
                if small {
                    push(
                        Arc::new(Recipe::MixedRadixSmall {
                            left_fft: Arc::clone(&left_fft),
                            right_fft: Arc::clone(&right_fft),
                        }),
                        &mut out,
                        &mut seen,
                    );
                }
                if coprime {
                    push(
                        Arc::new(Recipe::GoodThomasAlgorithm {
                            left_fft: Arc::clone(&left_fft),
                            right_fft: Arc::clone(&right_fft),
                        }),
                        &mut out,
                        &mut seen,
                    );
                    if small {
                        push(
                            Arc::new(Recipe::GoodThomasAlgorithmSmall {
                                left_fft: Arc::clone(&left_fft),
                                right_fft: Arc::clone(&right_fft),
                            }),
                            &mut out,
                            &mut seen,
                        );
                    }
                }
            }
        }

        // Radix4 on every base that divides out to a power of four. NeonRadix4 requires the
        // base length to be a whole number of vector pairs, which rules out the smallest bases.
        let base_multiple = if TypeId::of::<T>() == TypeId::of::<f32>() {
            4
        } else {
            2
        };
        for base in RADIX4_BASES {
            if len % base != 0 || base % base_multiple != 0 {
                continue;
            }
            let cross = len / base;
            if !cross.is_power_of_two() || cross.trailing_zeros() % 2 != 0 {
                continue;
            }
            let base_fft = self.plan(base);
            push(
                Arc::new(Recipe::Radix4 {
                    k: cross.trailing_zeros() / 2,
                    base_fft,
                }),
                &mut out,
                &mut seen,
            );
        }

        // Prime lengths: Rader's, plus Bluestein's over a range of inner lengths.
        if len > 3 && crate::math_utils::PrimeFactors::compute(len).is_prime() {
            let inner_fft = self.plan(len - 1);
            push(
                Arc::new(Recipe::RadersAlgorithm { inner_fft }),
                &mut out,
                &mut seen,
            );

            let min_inner = 2 * len - 1;
            let mut inner_lens: Vec<usize> = Vec::new();
            for multiplier in [1usize, 3, 5, 7, 9, 15] {
                // smallest multiplier * 2^k that is large enough
                let mut candidate = multiplier;
                while candidate < min_inner {
                    candidate *= 2;
                }
                inner_lens.push(candidate);
            }
            inner_lens.sort_unstable();
            inner_lens.dedup();
            for inner_len in inner_lens {
                let inner_fft = self.plan(inner_len);
                push(
                    Arc::new(Recipe::BluesteinsAlgorithm { len, inner_fft }),
                    &mut out,
                    &mut seen,
                );
            }
        }

        // A bare butterfly, when one exists at this size.
        if NEON_BUTTERFLIES.contains(&len) || prime_butterfly_lens().contains(&len) {
            if let Ok(recipe) = butterfly(len) {
                push(recipe, &mut out, &mut seen);
            }
        }

        out
    }
}

impl<T: FftNum> Default for NeonTuner<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Every butterfly length the Neon planner can use.
pub fn butterfly_lens() -> Vec<usize> {
    let mut lens: Vec<usize> = NEON_BUTTERFLIES.to_vec();
    lens.extend_from_slice(prime_butterfly_lens());
    lens.sort_unstable();
    lens.dedup();
    lens
}

/// Every `(base, k)` shape `Radix4` can take for element type `T`, up to `max_len`.
pub fn radix4_shapes<T: FftNum>(max_len: usize) -> Vec<(usize, u32)> {
    let base_multiple = if TypeId::of::<T>() == TypeId::of::<f32>() {
        4
    } else {
        2
    };
    let mut shapes = Vec::new();
    for base in RADIX4_BASES {
        if base % base_multiple != 0 {
            continue;
        }
        let mut k = 1u32;
        while base * (1usize << (2 * k)) <= max_len {
            shapes.push((base, k));
            k += 1;
        }
    }
    shapes
}
