#![feature(test)]
//! Planner choice benchmarks for the wasm_simd backend.
//!
//! These are an exploration tool, not a regression suite, so they are opt in and a plain
//! `cargo bench` skips them. Run them by name:
//!
//! ```text
//! cargo +nightly bench --bench bench_planner_choices_wasm_simd
//! ```
//!
//! See `benches/planner_choices/body.rs` for what the groups mean and how to read the numbers.

extern crate test;

#[macro_use]
#[path = "planner_choices/body.rs"]
mod body;

use rustfft::FftPlannerWasmSimd;

planner_choice_benches!(FftPlannerWasmSimd);
