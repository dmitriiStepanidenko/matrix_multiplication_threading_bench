use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use matrix_criterion::gemm_parallel_with_threads;
use matrix_criterion::{gemm, gemm_parallel, gemm_transpose, gemm_parallel_transpose};
use rand::Rng;

const SAMPLE_SIZE: usize = 25;
const ITERATION_SIZE: usize = 5000;
const ITERATION_STEP_SIZE: usize = 100;
const THREAD_MAX_NUMBER_SIZE: usize = 15;

fn generate_random_matrix(size: &usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let elements = size * size;
    let mut matrix = Vec::with_capacity(elements);

    for _ in 0..elements {
        matrix.push(rng.gen()); // Generates a random f64
    }

    matrix
}

fn gemm_transpose_parallel_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix transpose and parallel");
    group.sample_size(SAMPLE_SIZE);

    for size in (800..ITERATION_SIZE).step_by(ITERATION_STEP_SIZE) {
        let matrix1 = black_box(generate_random_matrix(&size));

        let mut matrix2 = black_box(generate_random_matrix(&size));
        let mut result = black_box(vec![0.0; size * size]);

        group.bench_with_input(BenchmarkId::new("Mutlithreaded multiplication", size), &size, |b, &size| {
            b.iter(|| gemm_parallel(&matrix1, &matrix2, &mut result, black_box(size)))
        });
        group.bench_with_input(BenchmarkId::new("Mutlithreaded transpose", size), &size, |b, &size| {
            b.iter(|| {
                let mut m2 = matrix2.clone();
                gemm_parallel_transpose(&matrix1, &mut matrix2, &mut m2, black_box(size));
            })
        });
    }

    group.finish();
}

fn gemm_transpose_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix transpose");
    group.sample_size(SAMPLE_SIZE);

    for size in (400..ITERATION_SIZE).step_by(ITERATION_STEP_SIZE) {
        let matrix1 = black_box(generate_random_matrix(&size));

        let mut matrix2 = black_box(generate_random_matrix(&size));
        let mut result = black_box(vec![0.0; size * size]);

        group.bench_with_input(BenchmarkId::new("Standard", size), &size, |b, &size| {
            b.iter(|| gemm(&matrix1, &matrix2, &mut result, black_box(size)))
        });
        group.bench_with_input(BenchmarkId::new("Transpose", size), &size, |b, &size| {
            b.iter(|| {
                let mut m2 = matrix2.clone();
                gemm_transpose(&matrix1, &mut matrix2, &mut m2, black_box(size));
            })
        });
    }

    group.finish();
}

fn gemm_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication");
    group.sample_size(SAMPLE_SIZE);

    for size in (10..ITERATION_SIZE).step_by(ITERATION_STEP_SIZE) {
        let matrix1 = black_box(generate_random_matrix(&size));

        let matrix2 = black_box(generate_random_matrix(&size));
        let mut result = black_box(vec![0.0; size * size]);

        group.bench_with_input(BenchmarkId::new("Standard", size), &size, |b, &size| {
            b.iter(|| gemm(&matrix1, &matrix2, &mut result, black_box(size)))
        });
        group.bench_with_input(BenchmarkId::new("Multithread", size), &size, |b, &size| {
            b.iter(|| gemm_parallel(&matrix1, &matrix2, &mut result, black_box(size)))
        });
    }

    group.finish();
}

fn gemm_benchmark_thread_pools(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix Multiplication thread pools variants");
    group.sample_size(SAMPLE_SIZE);

    for size in (10..ITERATION_SIZE).step_by(ITERATION_STEP_SIZE) {
        let matrix1 = black_box(generate_random_matrix(&size));

        let matrix2 = black_box(generate_random_matrix(&size));
        let mut result = black_box(vec![0.0; size * size]);

        for threads_number in 2..THREAD_MAX_NUMBER_SIZE + 1 {
            group.bench_with_input(
                BenchmarkId::new(format!("Multithread {} threads", threads_number), size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        gemm_parallel_with_threads(
                            &matrix1,
                            &matrix2,
                            &mut result,
                            black_box(size),
                            black_box(threads_number),
                        )
                    })
                },
            );
        }
        group.bench_with_input(BenchmarkId::new("Standard", size), &size, |b, &size| {
            b.iter(|| gemm(&matrix1, &matrix2, &mut result.clone(), black_box(size)))
        });
    }

    group.finish();
}

//criterion_group!(benches, gemm_benchmark, gemm_benchmark_thread_pools);
//criterion_group!(benches, gemm_transpose_benchmark);
criterion_group!(benches, gemm_transpose_parallel_benchmark);
criterion_main!(benches);
