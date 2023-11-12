use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

pub fn gemm_parallel_transpose<T>(matrix1: &[T], matrix2: &mut [T], result: &mut [T], mat_size: usize)
where
    T: PartialOrd
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + Copy
        + Send
        + Sync
        + Default,
{
    for i in 1..mat_size {
        for j in i..mat_size {
            matrix2.swap(i * mat_size + j, j * mat_size + i);
        }
    }
    result
        .par_chunks_mut(mat_size)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..mat_size {
                let mut sum = T::default();
                for k in 0..mat_size {
                    let a_idx = i * mat_size + k;
                    let b_idx = j * mat_size + k;
                    sum += matrix1[a_idx] * matrix2[b_idx];
                }
                row[j] = sum;
            }
        });
}

pub fn gemm_transpose<'a, T>(
    matrix1: &'a [T],
    matrix2: &'a mut [T],
    result: &'a mut [T],
    mat_size: usize,
) -> &'a mut [T]
where
    T: PartialOrd
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + Copy,
{
    for i in 1..mat_size {
        for j in i..mat_size {
            matrix2.swap(i * mat_size + j, j * mat_size + i);
        }
    }

    for i in 0..mat_size {
        for j in 0..mat_size {
            let c_idx = i * mat_size + j;
            for k in 0..mat_size {
                let a_idx = i * mat_size + k;
                let b_idx = j * mat_size + k;
                result[c_idx] += matrix1[a_idx] * matrix2[b_idx];
            }
        }
    }
    return result;
}

pub fn gemm< T>(
    matrix1: & [T],
    matrix2: & [T],
    result: & mut [T],
    mat_size: usize,
) 
where
    T: PartialOrd
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + Copy,
{
    for i in 0..mat_size {
        for j in 0..mat_size {
            let c_idx = i * mat_size + j;
            for k in 0..mat_size {
                let a_idx = i * mat_size + k;
                let b_idx = k * mat_size + j;
                result[c_idx] += matrix1[a_idx] * matrix2[b_idx];
            }
        }
    }
}

pub fn gemm_parallel_with_threads<T>(
    matrix1: &[T],
    matrix2: &[T],
    result: &mut [T],
    mat_size: usize,
    threads_number: usize,
) where
    T: PartialOrd
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + Copy
        + Send
        + Sync
        + Default,
{
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(threads_number)
        .build()
        .unwrap();
    thread_pool.install(|| {
        result
            .par_chunks_mut(mat_size)
            .enumerate()
            .for_each(|(i, row)| {
                for j in 0..mat_size {
                    let mut sum = T::default();
                    for k in 0..mat_size {
                        let a_idx = i * mat_size + k;
                        let b_idx = k * mat_size + j;
                        sum += matrix1[a_idx] * matrix2[b_idx];
                    }
                    row[j] = sum;
                }
            });
    });
}

pub fn gemm_parallel<T>(matrix1: &[T], matrix2: &[T], result: &mut [T], mat_size: usize)
where
    T: PartialOrd
        + std::ops::Mul<Output = T>
        + std::ops::Add<Output = T>
        + std::ops::AddAssign
        + Copy
        + Send
        + Sync
        + Default,
{
    result
        .par_chunks_mut(mat_size)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..mat_size {
                let mut sum = T::default();
                for k in 0..mat_size {
                    let a_idx = i * mat_size + k;
                    let b_idx = k * mat_size + j;
                    sum += matrix1[a_idx] * matrix2[b_idx];
                }
                row[j] = sum;
            }
        });
}
