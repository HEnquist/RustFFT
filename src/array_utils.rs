/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output
/// benchmarking shows that loop tiling isn't effective for small arrays (in the range of 50x50 or smaller)
pub unsafe fn transpose_small<T: Copy>(width: usize, height: usize, input: &[T], output: &mut [T]) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

const BLOCK_SIZE: usize = 64;

/// Transpose a subset of the array, from the input into the output. The idea is that by transposing one block at a time, we can be more cache-friendly
/// SAFETY: Width * height must equal input.len() and output.len(), start_x + block_width must be <= width, start_y + block height must be <= height
/// This function is taken directly from the "transpose" library.
unsafe fn transpose_block<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, start_x: usize, start_y: usize, block_width: usize, block_height: usize) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = start_x + inner_x;
            let y = start_y + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
        }
    }
}

/// Transpose a subset of the array, from the input into the output. The idea is that by transposing one block at a time, we can be more cache-friendly
/// SAFETY: Width * height must equal input.len() and output.len(), start_x + block_width must be <= width, start_y + block height must be <= height
/// Also divide the loop into 8 segments, to avoid repeatedly reading from start to end of the whole array. This makes it more cache friendly, 
/// and for long lengths it's over 2x faster. 
unsafe fn transpose_block_segmented8<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize, start_x: usize, start_y: usize, block_width: usize, block_height: usize) {
    let height_per_div = block_height/8;
    for subblock in 0..8 {
        for inner_x in 0..block_width {
            for inner_y in 0..height_per_div {
                let x = start_x + inner_x;
                let y = start_y + inner_y + subblock*height_per_div;

                let input_index = x + y * width;
                let output_index = y + x * height;

                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}

/// Given an array of size width * height, representing a flattened 2D array,
/// transpose the rows and columns of that 2D array into the output.
/// This is a customized version of the out-of-place transpose function from the "transpose" library.
/// For the bulk of the work, it uses a modified function that tries to limit cache misses.
pub fn transpose<T: Copy>(input: &[T], output: &mut [T], input_width: usize, input_height: usize) {
    assert_eq!(input_width*input_height, input.len());
    assert_eq!(input_width*input_height, output.len());

    let x_block_count = input_width / BLOCK_SIZE;
    let y_block_count = input_height / BLOCK_SIZE;

    let remainder_x = input_width - x_block_count * BLOCK_SIZE;
    let remainder_y = input_height - y_block_count * BLOCK_SIZE;


    for y_block in 0..y_block_count {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block_segmented8(
                    input, output,
                    input_width, input_height,
                    x_block * BLOCK_SIZE, y_block * BLOCK_SIZE,
                    BLOCK_SIZE, BLOCK_SIZE,
                    );
            }
        }

        //if the input_width is not cleanly divisible by block_size, there are still a few columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_block(
                    input, output, 
                    input_width, input_height, 
                    input_width - remainder_x, y_block * BLOCK_SIZE, 
                    remainder_x, BLOCK_SIZE);
            }
        }
    }

    //if the input_height is not cleanly divisible by BLOCK_SIZE, there are still a few rows that haven't been transposed
    if remainder_y > 0 {
        for x_block in 0..x_block_count {
            unsafe {
                transpose_block(
                    input, output,
                    input_width, input_height,
                    x_block * BLOCK_SIZE, input_height - remainder_y,
                    BLOCK_SIZE, remainder_y,
                    );
            }
        }

        //if the input_width is not cleanly divisible by block_size, there are still a few rows+columns that haven't been transposed
        if remainder_x > 0 {
            unsafe {
                transpose_block(
                    input, output,
                    input_width, input_height, 
                    input_width - remainder_x, input_height - remainder_y, 
                    remainder_x, remainder_y);
            }
        }
    } 
}

#[allow(unused)]
pub unsafe fn workaround_transmute<T, U>(slice: &[T]) -> &[U] {
    let ptr = slice.as_ptr() as *const U;
    let len = slice.len();
    std::slice::from_raw_parts(ptr, len)
}
#[allow(unused)]
pub unsafe fn workaround_transmute_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    let ptr = slice.as_mut_ptr() as *mut U;
    let len = slice.len();
    std::slice::from_raw_parts_mut(ptr, len)
}

#[derive(Copy, Clone)]
pub struct RawSlice<T> {
    ptr: *const T,
    slice_len: usize,
}
impl<T> RawSlice<T> {
    #[inline(always)]
    pub fn new(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub unsafe fn new_transmuted<U>(slice: &[U]) -> Self {
        Self {
            ptr: slice.as_ptr() as *const T,
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice_len
    }
}
impl<T: Copy> RawSlice<T> {
    #[inline(always)]
    pub unsafe fn load(&self, index: usize) -> T {
        debug_assert!(index < self.slice_len);
        *self.ptr.add(index)
    }
}

/// A RawSliceMut is a normal mutable slice, but aliasable. Its functionality is severely limited.
#[derive(Copy, Clone)]
pub struct RawSliceMut<T> {
    ptr: *mut T,
    slice_len: usize,
}
impl<T> RawSliceMut<T> {
    #[inline(always)]
    pub fn new(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub unsafe fn new_transmuted<U>(slice: &mut [U]) -> Self {
        Self {
            ptr: slice.as_mut_ptr() as *mut T,
            slice_len: slice.len(),
        }
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
    #[allow(unused)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.slice_len
    }
    #[inline(always)]
    pub unsafe fn store(&self, value: T, index: usize) {
        debug_assert!(index < self.slice_len);
        *self.ptr.add(index) = value;
    }
}
//impl<T: Copy> RawSliceMut<T> {
//    #[inline(always)]
//    pub unsafe fn load(&self, index: usize) -> T {
//        debug_assert!(index < self.slice_len);
//        *self.ptr.add(index)
//    }
//}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::test_utils::random_signal;
    use num_complex::Complex;
    use num_traits::Zero;

    #[test]
    fn test_transpose_small() {
        let sizes: Vec<usize> = (1..16).collect();

        for &width in &sizes {
            for &height in &sizes {
                let len = width * height;

                let input: Vec<Complex<f32>> = random_signal(len);
                let mut output = vec![Zero::zero(); len];

                unsafe { transpose_small(width, height, &input, &mut output) };

                for x in 0..width {
                    for y in 0..height {
                        assert_eq!(
                            input[x + y * width],
                            output[y + x * height],
                            "x = {}, y = {}",
                            x,
                            y
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_transpose() {
        let sizes: Vec<usize> = (1..16).collect();

        for &width in &sizes {
            for &height in &sizes {
                let len = width * height;

                let input: Vec<Complex<f32>> = random_signal(len);
                let mut output = vec![Zero::zero(); len];

                transpose(&input, &mut output, width, height);

                for x in 0..width {
                    for y in 0..height {
                        assert_eq!(
                            input[x + y * width],
                            output[y + x * height],
                            "x = {}, y = {}",
                            x,
                            y
                        );
                    }
                }
            }
        }
    }
}

// Loop over exact chunks of the provided buffer. Very similar in semantics to ChunksExactMut, but generates smaller code and requires no modulo operations
// Returns Ok() if every element ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks<T>(
    mut buffer: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T]),
) -> Result<(), ()> {
    // Loop over the buffer, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer.len() >= chunk_size {
        let (head, tail) = buffer.split_at_mut(chunk_size);
        buffer = tail;

        chunk_fn(head);
    }

    // We have a remainder if there's data still in the buffer -- in which case we want to indicate to the caller that there was an unwanted remainder
    if buffer.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}

// Loop over exact zipped chunks of the 2 provided buffers. Very similar in semantics to ChunksExactMut.zip(ChunksExactMut), but generates smaller code and requires no modulo operations
// Returns Ok() if every element of both buffers ended up in a chunk, Err() if there was a remainder
pub fn iter_chunks_zipped<T>(
    mut buffer1: &mut [T],
    mut buffer2: &mut [T],
    chunk_size: usize,
    mut chunk_fn: impl FnMut(&mut [T], &mut [T]),
) -> Result<(), ()> {
    // If the two buffers aren't the same size, record the fact that they're different, then snip them to be the same size
    let uneven = if buffer1.len() > buffer2.len() {
        buffer1 = &mut buffer1[..buffer2.len()];
        true
    } else if buffer2.len() < buffer1.len() {
        buffer2 = &mut buffer2[..buffer1.len()];
        true
    } else {
        false
    };

    // Now that we know the two slices are the same length, loop over each one, splicing off chunk_size at a time, and calling chunk_fn on each
    while buffer1.len() >= chunk_size && buffer2.len() >= chunk_size {
        let (head1, tail1) = buffer1.split_at_mut(chunk_size);
        buffer1 = tail1;

        let (head2, tail2) = buffer2.split_at_mut(chunk_size);
        buffer2 = tail2;

        chunk_fn(head1, head2);
    }

    // We have a remainder if the 2 chunks were uneven to start with, or if there's still data in the buffers -- in which case we want to indicate to the caller that there was an unwanted remainder
    if !uneven && buffer1.len() == 0 {
        Ok(())
    } else {
        Err(())
    }
}
