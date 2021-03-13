#![feature(test)]
extern crate test;
extern crate rustfft;

use test::Bencher;
use rustfft::algorithm::butterflies::*;

use transpose;

/*
fn cachetranspose_simple<T: Copy>(row_start: usize, row_end: usize, col_start: usize, col_end:  usize, columns: usize, rows: usize, A: &[T], B: &mut [T]) {
    let r = row_end - row_start; 
    let c = col_end - col_start;
    if (r <= 12 && c <= 12) || r<=2 || c<=2 {
        //if r*c <= 256 || c <=2 || r <=2 {
        unsafe {
            //for j in col_start..col_end {
            //    for i in row_start..row_end {
            //        *B.get_unchecked_mut(j * rows + i) = *A.get_unchecked(i * columns + j);
            //    }
            //}
            for inner_x in 0..c {
                let x = col_start + inner_x;
                let x_rows = x * rows;
                for inner_y in 0..r {
                    let y = row_start + inner_y;
                    let input_index = x + y * columns;
                    let output_index = y + x_rows;
                    *B.get_unchecked_mut(output_index) = *A.get_unchecked(input_index);
                }
            }
        }
    //if (r <= 12 && c <= 12) || r<=2 || c<=2 {
    ////if r*c <= 256 || c <=2 || r <=2 {
    //    unsafe {
    //        //for j in col_start..col_end {
    //        //    for i in row_start..row_end {
    //        //        *B.get_unchecked_mut(j * rows + i) = *A.get_unchecked(i * columns + j);
    //        //    }
    //        //}
    //        for inner_x in 0..c {
    //            let x = col_start + inner_x;
    //            let x_rows = x * rows;
    //            for inner_y in 0..r {
    //                let y = row_start + inner_y;
    //                let input_index = x + y * columns;
    //                let output_index = y + x_rows;
    //                *B.get_unchecked_mut(output_index) = *A.get_unchecked(input_index);
    //            }
    //        }
    //        
    //    }
    } else if (r >= c) {
        cachetranspose(row_start, row_start + (r / 2), col_start, col_end, columns, rows, A, B);
        cachetranspose(row_start + (r / 2), row_end, col_start, col_end, columns, rows, A, B);
    } else {
        cachetranspose(row_start, row_end, col_start, col_start + (c / 2), columns, rows, A, B);
        cachetranspose(row_start, row_end, col_start + (c / 2), col_end, columns, rows, A, B);
    }
}
*/
/*
fn cachetranspose<T: Copy>(row_start: usize, row_end: usize, col_start: usize, col_end:  usize, columns: usize, rows: usize, A: &[T], B: &mut [T]) {
    let r = row_end - row_start; 
    let c = col_end - col_start;
    if (r <= 128 && c <= 128) || r<=2 || c<=2 {
        let x_block_count = c / BLOCK_SIZE;
        let y_block_count = r / BLOCK_SIZE;

        let remainder_x = c - x_block_count * BLOCK_SIZE;
        let remainder_y = r - y_block_count * BLOCK_SIZE;


        for y_block in 0..y_block_count {
            for x_block in 0..x_block_count {
                unsafe {
                    transpose_block_segmented4(
                        A, B,
                        columns, rows,
                        col_start + x_block * BLOCK_SIZE, row_start + y_block * BLOCK_SIZE,
                        BLOCK_SIZE, BLOCK_SIZE,
                        );
                }
            }

            //if the input_width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
            if remainder_x > 0 {
                unsafe {
                    transpose_block(
                        A, B,
                        columns, rows,
                        col_start + x_block_count * BLOCK_SIZE, row_start + y_block * BLOCK_SIZE, 
                        remainder_x, BLOCK_SIZE);
                }
            }
        }

        //if the input_height is not cleanly divisible by BLOCK_SIZE, there are still a few rows that haven't been transposed
        if remainder_y > 0 {
            for x_block in 0..x_block_count {
                unsafe {
                    transpose_block(
                        A, B,
                        columns, rows, 
                        col_start + x_block * BLOCK_SIZE, row_start + y_block_count * BLOCK_SIZE,
                        BLOCK_SIZE, remainder_y,
                        );
                }
            }

            //if the input_width is not cleanly divisible by block_size, there are still a few rows+columns that haven't been transposed
            if remainder_x > 0 {
                unsafe {
                    transpose_block(
                        A, B,
                        columns, rows, 
                        col_start + x_block_count * BLOCK_SIZE,  row_start + y_block_count * BLOCK_SIZE, 
                        remainder_x, remainder_y);
                }
            }
        } 
    } else if (r >= c) {
        cachetranspose(row_start, row_start + (r / 2), col_start, col_end, columns, rows, A, B);
        cachetranspose(row_start + (r / 2), row_end, col_start, col_end, columns, rows, A, B);
    } else {
        cachetranspose(row_start, row_end, col_start, col_start + (c / 2), columns, rows, A, B);
        cachetranspose(row_start, row_end, col_start + (c / 2), col_end, columns, rows, A, B);
    }
}
*/

// fn bench_transpose_rec(b: &mut Bencher, width: usize, height: usize) {
//     let len = width * height;
//     let mut scratch = vec![0.0f64; len];
//     b.iter(|| { DC_matrix_transpose(&mut scratch, width, 0, width, 0, 0, height); });
// }

// #[bench] fn rec_64_64(b: &mut Bencher) { bench_transpose_rec(b, 64, 64); }
// #[bench] fn rec_128_128(b: &mut Bencher) { bench_transpose_rec(b, 128, 128); }
// #[bench] fn rec_512_512(b: &mut Bencher) { bench_transpose_rec(b, 512, 512); }
// #[bench] fn rec_1024_1024(b: &mut Bencher) { bench_transpose_rec(b, 1024, 1024); }
// #[bench] fn rec_2048_2048(b: &mut Bencher) { bench_transpose_rec(b, 2048, 2048); }
// #[bench] fn rec_4096_4096(b: &mut Bencher) { bench_transpose_rec(b, 4096, 4096); }
// #[bench] fn rec_600_700(b: &mut Bencher) { bench_transpose_rec(b, 600, 700); }
// #[bench] fn rec_512_700(b: &mut Bencher) { bench_transpose_rec(b, 512, 700); }
// #[bench] fn rec_700_512(b: &mut Bencher) { bench_transpose_rec(b, 700, 512); }
// #[bench] fn rec_512_513(b: &mut Bencher) { bench_transpose_rec(b, 512, 513); }
// #[bench] fn rec_123_1234(b: &mut Bencher) { bench_transpose_rec(b, 123, 1234); }
// #[bench] fn rec_1234_2345(b: &mut Bencher) { bench_transpose_rec(b, 1234, 2345); }
// #[bench] fn rec_1024_2048(b: &mut Bencher) { bench_transpose_rec(b, 1024, 2048); }
// #[bench] fn rec_2048_1024(b: &mut Bencher) { bench_transpose_rec(b, 2048, 1024); }
// #[bench] fn rec_2_1234(b: &mut Bencher) { bench_transpose_rec(b, 2, 1234); }

// fn bench_transpose_new(b: &mut Bencher, width: usize, height: usize) {
//     let len = width * height;
//     let mut buffer = vec![0.0f64; len];
//     let mut scratch = vec![0.0f64; len];
//     b.iter(|| { transpose_new(&buffer, &mut scratch, width, height); });
// }

// fn bench_transpose_rec(b: &mut Bencher, width: usize, height: usize) {
//     let len = width * height;
//     let mut scratch = vec![0.0f64; len];
//     b.iter(|| { DC_matrix_transpose(&mut scratch, width, 0, width, 0, 0, height); });
// }

fn bench_transpose_coa(b: &mut Bencher, width: usize, height: usize) {
    let len = width * height;
    let buffer = vec![0.0f64; len];
    let mut scratch = vec![0.0f64; len];
    b.iter(|| { rustfft::array_utils::transpose(&buffer, &mut scratch, width, height); });
}

#[bench] fn coa_64_64(b: &mut Bencher) { bench_transpose_coa(b, 64, 64); }
#[bench] fn coa_128_128(b: &mut Bencher) { bench_transpose_coa(b, 128, 128); }
#[bench] fn coa_512_512(b: &mut Bencher) { bench_transpose_coa(b, 512, 512); }
#[bench] fn coa_1024_1024(b: &mut Bencher) { bench_transpose_coa(b, 1024, 1024); }
#[bench] fn coa_2048_2048(b: &mut Bencher) { bench_transpose_coa(b, 2048, 2048); }
#[bench] fn coa_4096_4096(b: &mut Bencher) { bench_transpose_coa(b, 4096, 4096); }
#[bench] fn coa_600_700(b: &mut Bencher) { bench_transpose_coa(b, 600, 700); }
#[bench] fn coa_512_700(b: &mut Bencher) { bench_transpose_coa(b, 512, 700); }
#[bench] fn coa_700_512(b: &mut Bencher) { bench_transpose_coa(b, 700, 512); }
#[bench] fn coa_512_513(b: &mut Bencher) { bench_transpose_coa(b, 512, 513); }
#[bench] fn coa_123_1234(b: &mut Bencher) { bench_transpose_coa(b, 123, 1234); }
#[bench] fn coa_1234_2345(b: &mut Bencher) { bench_transpose_coa(b, 1234, 2345); }
#[bench] fn coa_1024_2048(b: &mut Bencher) { bench_transpose_coa(b, 1024, 2048); }
#[bench] fn coa_2048_1024(b: &mut Bencher) { bench_transpose_coa(b, 2048, 1024); }
#[bench] fn coa_2_1234(b: &mut Bencher) { bench_transpose_coa(b, 2, 1234); }

fn bench_transpose_sml(b: &mut Bencher, width: usize, height: usize) {
    let len = width * height;
    let buffer = vec![0.0f64; len];
    let mut scratch = vec![0.0f64; len];
    b.iter(|| unsafe { rustfft::array_utils::transpose_small(width, height, &buffer, &mut scratch); });
}

#[bench] fn sml_64_64(b: &mut Bencher) { bench_transpose_sml(b, 64, 64); }
#[bench] fn sml_128_128(b: &mut Bencher) { bench_transpose_sml(b, 128, 128); }
#[bench] fn sml_512_512(b: &mut Bencher) { bench_transpose_sml(b, 512, 512); }
#[bench] fn sml_1024_1024(b: &mut Bencher) { bench_transpose_sml(b, 1024, 1024); }
#[bench] fn sml_2048_2048(b: &mut Bencher) { bench_transpose_sml(b, 2048, 2048); }
#[bench] fn sml_4096_4096(b: &mut Bencher) { bench_transpose_sml(b, 4096, 4096); }
#[bench] fn sml_600_700(b: &mut Bencher) { bench_transpose_sml(b, 600, 700); }
#[bench] fn sml_512_700(b: &mut Bencher) { bench_transpose_sml(b, 512, 700); }
#[bench] fn sml_700_512(b: &mut Bencher) { bench_transpose_sml(b, 700, 512); }
#[bench] fn sml_512_513(b: &mut Bencher) { bench_transpose_sml(b, 512, 513); }
#[bench] fn sml_123_1234(b: &mut Bencher) { bench_transpose_sml(b, 123, 1234); }
#[bench] fn sml_1234_2345(b: &mut Bencher) { bench_transpose_sml(b, 1234, 2345); }
#[bench] fn sml_1024_2048(b: &mut Bencher) { bench_transpose_sml(b, 1024, 2048); }
#[bench] fn sml_2048_1024(b: &mut Bencher) { bench_transpose_sml(b, 2048, 1024); }
#[bench] fn sml_2_1234(b: &mut Bencher) { bench_transpose_sml(b, 2, 1234); }


fn bench_transpose_lib(b: &mut Bencher, width: usize, height: usize) {
    let len = width * height;
    let buffer = vec![0.0f64; len];
    let mut scratch = vec![0.0f64; len];
    b.iter(|| { transpose::transpose(&buffer, &mut scratch, width, height); });
}

#[bench] fn lib_64_64(b: &mut Bencher) { bench_transpose_lib(b, 64, 64); }
#[bench] fn lib_128_128(b: &mut Bencher) { bench_transpose_lib(b, 128, 128); }
#[bench] fn lib_512_512(b: &mut Bencher) { bench_transpose_lib(b, 512, 512); }
#[bench] fn lib_1024_1024(b: &mut Bencher) { bench_transpose_lib(b, 1024, 1024); }
#[bench] fn lib_2048_2048(b: &mut Bencher) { bench_transpose_lib(b, 2048, 2048); }
#[bench] fn lib_4096_4096(b: &mut Bencher) { bench_transpose_lib(b, 4096, 4096); }
#[bench] fn lib_600_700(b: &mut Bencher) { bench_transpose_lib(b, 600, 700); }
#[bench] fn lib_512_700(b: &mut Bencher) { bench_transpose_lib(b, 512, 700); }
#[bench] fn lib_700_512(b: &mut Bencher) { bench_transpose_lib(b, 700, 512); }
#[bench] fn lib_512_513(b: &mut Bencher) { bench_transpose_lib(b, 512, 513); }
#[bench] fn lib_123_1234(b: &mut Bencher) { bench_transpose_lib(b, 123, 1234); }
#[bench] fn lib_1234_2345(b: &mut Bencher) { bench_transpose_lib(b, 1234, 2345); }
#[bench] fn lib_1024_2048(b: &mut Bencher) { bench_transpose_lib(b, 1024, 2048); }
#[bench] fn lib_2048_1024(b: &mut Bencher) { bench_transpose_lib(b, 2048, 1024); }
#[bench] fn lib_2_1234(b: &mut Bencher) { bench_transpose_lib(b, 2, 1234); }