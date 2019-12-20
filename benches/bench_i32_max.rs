#![allow(unused)]

#[macro_use]
extern crate criterion;

use argmm;
use criterion::{black_box, Criterion};

use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Exp, Uniform};

fn get_array_i32() -> Vec<i32> {
    let rng = thread_rng();
    let uni = Uniform::new_inclusive(-100_000, 100_000);
    rng.sample_iter(uni).take(1024).collect()
}

fn max_i32(c: &mut Criterion) {
    let data = get_array_i32();
    c.bench_function("argmax_i32", |b| {
        b.iter(|| argmm::argmax(black_box(data.as_slice())))
    });
}

fn simd_max_i32(c: &mut Criterion) {
    let data = get_array_i32();
    c.bench_function("argmax_simd i32", |b| {
        b.iter(|| argmm::argmax_i32(black_box(data.as_slice())))
    });
}

criterion_group!(benches, max_i32, simd_max_i32);
criterion_main!(benches);
