//! Microbenchmarks for DEFLATE/zlib throughput.
//!
//! Compares comprs `deflate_zlib` against `flate2` (miniz_oxide backend)
//! across compressible and incompressible payloads, reporting throughput in bytes.

use comprs::compress::deflate::deflate_zlib;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use flate2::{write::ZlibEncoder, Compression};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use std::io::Write;

fn make_compressible(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    while out.len() < len {
        out.extend_from_slice(pattern);
    }
    out.truncate(len);
    out
}

fn make_random(len: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut rng = StdRng::seed_from_u64(seed);
    rng.fill_bytes(&mut out);
    out
}

fn bench_deflate(c: &mut Criterion) {
    let mut group = c.benchmark_group("deflate_zlib");

    let cases = [
        ("compressible_1mb", make_compressible(1 << 20)),
        ("random_1mb", make_random(1 << 20, 424242)),
    ];

    for (name, data) in cases {
        let bytes = data.len() as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("comprs", name), &data, |b, input| {
            b.iter(|| {
                let encoded = deflate_zlib(black_box(input), 6);
                black_box(encoded.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("flate2", name), &data, |b, input| {
            b.iter(|| {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(black_box(input)).unwrap();
                let encoded = encoder.finish().unwrap();
                black_box(encoded.len())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_deflate);
criterion_main!(benches);
