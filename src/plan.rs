use std::collections::HashMap;
use std::sync::Arc;
use num_integer::gcd;

use crate::common::FFTnum;

use crate::FFT;
use crate::algorithm::*;
use crate::algorithm::butterflies::*;

use crate::math_utils;

#[derive(Debug, std::cmp::PartialEq)]
pub enum Plan {
    DFT(usize),
    MixedRadix { left_fft: Box<Plan>, right_fft: Box<Plan>},
    GoodThomas { left_fft: Box<Plan>, right_fft: Box<Plan>},
    MixedRadixDoubleButterfly(usize, usize),
    GoodThomasDoubleButterfly(usize, usize),
    Rader { len: usize, inner_fft: Box<Plan>},
    Bluestein { len: usize, inner_fft: Box<Plan>},
    Radix4(usize),
    Butterfly(usize),
}

impl Plan {
    pub fn len(&self) -> usize {
        match self {
            Plan::DFT(len) => *len,
            Plan::Radix4(len) => *len,
            Plan::Butterfly(len) => *len,
            Plan::MixedRadix { left_fft, right_fft } => left_fft.len() * right_fft.len(),
            Plan::GoodThomas { left_fft, right_fft } => left_fft.len() * right_fft.len(),
            Plan::MixedRadixDoubleButterfly(left_len, right_len) => *left_len * *right_len,
            Plan::GoodThomasDoubleButterfly(left_len, right_len) => *left_len * *right_len,
            Plan::Rader { len, .. } => *len,
            Plan::Bluestein { len , .. } => *len,
        }
    }

    pub fn cost(&self) -> usize {
        match self {
            Plan::DFT(len) => 4*len.pow(2),
            Plan::Radix4(len) => 2*len*((*len as f32).ln().round() as usize),
            Plan::Butterfly(len) => len*((*len as f32).ln().round() as usize),
            Plan::MixedRadix { left_fft, right_fft } => {
                let left_cost = left_fft.cost();
                let right_cost = right_fft.cost();
                left_cost*right_fft.len() + right_cost*left_fft.len()
            },
            Plan::GoodThomas { left_fft, right_fft } => {
                let left_cost = left_fft.cost();
                let right_cost = right_fft.cost();
                left_cost*right_fft.len() + right_cost*left_fft.len()
            },
            Plan::MixedRadixDoubleButterfly(left_len, right_len) => {
                let left_cost = left_len*((*left_len as f32).ln().round() as usize);
                let right_cost = left_len*((*left_len as f32).ln().round() as usize);
                left_len*right_cost + right_len*left_cost
            },
            Plan::GoodThomasDoubleButterfly(left_len, right_len) => {
                let left_cost = left_len*((*left_len as f32).ln().round() as usize);
                let right_cost = left_len*((*left_len as f32).ln().round() as usize);
                left_len*right_cost + right_len*left_cost
            },
            Plan::Rader { len, inner_fft } => *len + 2*inner_fft.cost(),
            Plan::Bluestein { len , inner_fft } =>  *len + 2*inner_fft.cost(),
        }
    }
}

const MIN_RADIX4_BITS: u32 = 5; // smallest size to consider radix 4 an option is 2^5 = 32
const MAX_RADIX4_BITS: u32 = 16; // largest size to consider radix 4 an option is 2^16 = 65536
const BUTTERFLIES: [usize; 16] = [2, 3, 4, 5, 6, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32];
const COMPOSITE_BUTTERFLIES: [usize; 16] = [2, 3, 4, 5, 6, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32];
//const COMPOSITE_BUTTERFLIES: [usize; 5] = [4, 6, 8, 16, 32];
const MAX_RADER_PRIME_FACTOR: usize = 23; // don't use Raders if the inner fft length has prime factor larger than this
const MIN_BLUESTEIN_MIXED_RADIX_LEN: usize = 90; // only use mixed radix for the inner fft of Bluestein if length is larger than this

macro_rules! butterfly {
    ($len:expr, $inverse:expr) => {
        match $len {
            2 => Arc::new(Butterfly2::new($inverse)),
            3 => Arc::new(Butterfly3::new($inverse)),
            4 => Arc::new(Butterfly4::new($inverse)),
            5 => Arc::new(Butterfly5::new($inverse)),
            6 => Arc::new(Butterfly6::new($inverse)),
            7 => Arc::new(Butterfly7::new($inverse)),
            8 => Arc::new(Butterfly8::new($inverse)),
            11 => Arc::new(Butterfly11::new($inverse)),
            13 => Arc::new(Butterfly13::new($inverse)),
            16 => Arc::new(Butterfly16::new($inverse)),
            17 => Arc::new(Butterfly17::new($inverse)),
            19 => Arc::new(Butterfly19::new($inverse)),
            23 => Arc::new(Butterfly23::new($inverse)),
            29 => Arc::new(Butterfly29::new($inverse)),
            31 => Arc::new(Butterfly31::new($inverse)),
            32 => Arc::new(Butterfly32::new($inverse)),
            _ => panic!("Invalid butterfly size: {}", $len),
        }
    };
}

/// The FFT planner is used to make new FFT algorithm instances.
///
/// RustFFT has several FFT algorithms available; For a given FFT size, the FFTplanner decides which of the
/// available FFT algorithms to use and then initializes them.
///
/// ~~~
/// // Perform a forward FFT of size 1234
/// use std::sync::Arc;
/// use rustfft::FFTplanner;
/// use rustfft::num_complex::Complex;
/// use rustfft::num_traits::Zero;
///
/// let mut input:  Vec<Complex<f32>> = vec![Zero::zero(); 1234];
/// let mut output: Vec<Complex<f32>> = vec![Zero::zero(); 1234];
///
/// let mut planner = FFTplanner::new(false);
/// let fft = planner.plan_fft(1234);
/// fft.process(&mut input, &mut output);
/// 
/// // The fft instance returned by the planner is stored behind an `Arc`, so it's cheap to clone
/// let fft_clone = Arc::clone(&fft);
/// ~~~
///
/// If you plan on creating multiple FFT instances, it is recommnded to reuse the same planner for all of them. This
/// is because the planner re-uses internal data across FFT instances wherever possible, saving memory and reducing
/// setup time. (FFT instances created with one planner will never re-use data and buffers with FFT instances created
/// by a different planner)
///
/// Each FFT instance owns `Arc`s to its internal data, rather than borrowing it from the planner, so it's perfectly
/// safe to drop the planner after creating FFT instances.
pub struct FFTplanner<T> {
    inverse: bool,
    algorithm_cache: HashMap<usize, Arc<FFT<T>>>,
    butterfly_cache: HashMap<usize, Arc<FFTButterfly<T>>>,
}

impl<T: FFTnum> FFTplanner<T> {
    /// Creates a new FFT planner.
    ///
    /// If `inverse` is false, this planner will plan forward FFTs. If `inverse` is true, it will plan inverse FFTs.
    pub fn new(inverse: bool) -> Self {
        FFTplanner {
            inverse: inverse,
            algorithm_cache: HashMap::new(),
            butterfly_cache: HashMap::new(),
        }
    }

    /// Returns a FFT instance which processes signals of size `len`
    /// If this is called multiple times, it will attempt to re-use internal data between instances
    pub fn plan_fft(&mut self, len: usize) -> Arc<FFT<T>> {
        if len < 2 {
            Arc::new(DFT::new(len, self.inverse)) as Arc<FFT<T>>
        } else if self.algorithm_cache.contains_key(&len) {
            Arc::clone(self.algorithm_cache.get(&len).unwrap())
        } else {
            let plan = self.make_plan_for_len(len);
            let fft = self.construct_fft(plan);
            self.algorithm_cache.insert(len, Arc::clone(&fft));
            fft
        }
    }

    // Make a plan for a length
    fn make_plan_for_len(&mut self, len: usize) -> Plan {
        if len < 2 {
            Plan::DFT(len)
        } else {    
            let factors = math_utils::prime_factors(len);
            self.make_plan_from_factors(len, &factors)
        }
    }

    // Make a plan for the given prime factors
    fn make_plan_from_factors(&mut self, len: usize, factors: &[usize]) -> Plan {
        if factors.len() == 1 || COMPOSITE_BUTTERFLIES.contains(&len) {
            //the length is either a prime or matches a butterfly
            self.make_plan_for_single_factor(len)
        } else if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS && len.is_power_of_two(){
            //the length is a power of two in the range where Radix4 is the fastest option.
            Plan::Radix4(len)
        } else {
            self.make_plan_for_mixed_radix(len, &factors)
        }
    }

    fn make_plan_for_mixed_radix(&mut self, len: usize, factors: &[usize]) -> Plan {
        if len.trailing_zeros() <= MAX_RADIX4_BITS && len.trailing_zeros() >= MIN_RADIX4_BITS {
            //the number of trailing zeroes in len is the number of `2` factors
            //ie if len = 2048 * n, len.trailing_zeros() will equal 11 because 2^11 == 2048
            let left_len = 1 << len.trailing_zeros();
            let right_len = len / left_len;
            let (left_factors, right_factors) = factors.split_at(len.trailing_zeros() as usize);
            self.make_plan_for_mixed_radix_from_factor_lists(left_len, left_factors, right_len, right_factors)
        } else {
            let sqrt = (len as f32).sqrt() as usize;
            if sqrt * sqrt == len {
                // since len is a perfect square, each of its prime factors is duplicated.
                // since we know they're sorted, we can loop through them in chunks of 2 and keep one out of each chunk
                // if the stride iterator ever becomes stabilized, it'll be cleaner to use that instead of chunks
                let mut sqrt_factors = Vec::with_capacity(factors.len() / 2);
                for chunk in factors.chunks(2) {
                    sqrt_factors.push(chunk[0]);
                }
                self.make_plan_for_mixed_radix_from_factor_lists(sqrt, &sqrt_factors, sqrt, &sqrt_factors)
            } else {
                //len isn't a perfect square. greedily take factors from the list until both sides are as close as possible to sqrt(len)
                //TODO: We can probably make this more optimal by using a more sophisticated non-greedy algorithm
                let mut product = 1;
                let mut second_half_index = 1;
                for (i, factor) in factors.iter().enumerate() {
                    if product * *factor > sqrt {
                        second_half_index = i;
                        break;
                    } else {
                        product = product * *factor;
                    }
                }

                //we now know that product is the largest it can be without being greater than len / product
                //there's one more thing we can try to make them closer together -- if product * factors[index] < len / product,
                if product * factors[second_half_index] < len / product {
                    product = product * factors[second_half_index];
                    second_half_index = second_half_index + 1;
                }

                //we now have our two FFT sizes: product and product / len
                let (left_factors, right_factors) = factors.split_at(second_half_index);
                self.make_plan_for_mixed_radix_from_factor_lists(product, left_factors, len / product, right_factors)
            }
        }
    }

    // Make a plan using mixed radix
    fn make_plan_for_mixed_radix_from_factor_lists(&mut self,
        left_len: usize,
        left_factors: &[usize],
        right_len: usize,
        right_factors: &[usize])
        -> Plan {

        let left_is_butterfly = BUTTERFLIES.contains(&left_len);
        let right_is_butterfly = BUTTERFLIES.contains(&right_len);

        //if both left_len and right_len are butterflies, use a mixed radix implementation specialized for butterfly sub-FFTs
        if left_is_butterfly && right_is_butterfly {            
            // for butterflies, if gcd is 1, we always want to use good-thomas
            if gcd(left_len, right_len) == 1 {
                Plan::GoodThomasDoubleButterfly(left_len, right_len)
            } else {
                Plan::MixedRadixDoubleButterfly(left_len, right_len)
            }
        } else {
            //neither size is a butterfly, so go with the normal algorithm
            let left_fft = Box::new(self.make_plan_from_factors(left_len, left_factors));
            let right_fft = Box::new(self.make_plan_from_factors(right_len, right_factors));
            //if gcd(left_len, right_len) == 1 {
            //    Plan::GoodThomas{left_fft, right_fft}
            //} else {
                Plan::MixedRadix{left_fft, right_fft}
            //}
        }
    }

    // Make a plan for a single factor
    fn make_plan_for_single_factor(&mut self, len: usize) -> Plan {
        match len {
            0|1=> Plan::DFT(len),
            2|3|4|5|6|7|8|16|32 => Plan::Butterfly(len),
            _ => self.make_plan_for_prime(len),
        }
    }

    // Make a plan for a prime factor
    fn make_plan_for_prime(&mut self, len: usize) -> Plan {
        let inner_fft_len_rader = len - 1;
        let factors = math_utils::prime_factors(inner_fft_len_rader);
        // If any of the prime factors is too large, Rader's gets slow and Bluestein's is the better choice
        if factors.iter().any(|val| *val > MAX_RADER_PRIME_FACTOR) {
            let inner_fft_len_pow2 = (2 * len - 1).checked_next_power_of_two().unwrap();
            // over a certain length, a shorter mixed radix inner fft is faster than a longer radix4
            let min_inner_len = 2 * len - 1;
            let mixed_radix_len = 3*inner_fft_len_pow2/4;
            let inner_fft = if mixed_radix_len >= min_inner_len && len >= MIN_BLUESTEIN_MIXED_RADIX_LEN {
                let inner_factors = math_utils::prime_factors(mixed_radix_len);
                self.make_plan_from_factors(mixed_radix_len, &inner_factors)
            }
            else {
                Plan::Radix4(inner_fft_len_pow2)
            };
            Plan::Bluestein{len, inner_fft: Box::new(inner_fft)}
        }
        else {
            let inner_fft = self.make_plan_from_factors(inner_fft_len_rader, &factors);
            Plan::Rader{len, inner_fft: Box::new(inner_fft)}
        }
    }

    // Create the fft from a plan
    fn construct_fft(&mut self, plan: Plan) -> Arc<FFT<T>> {
        match plan {
            Plan::DFT(len) => Arc::new(DFT::new(len, self.inverse)) as Arc<FFT<T>>,
            Plan::Radix4(len) => Arc::new(Radix4::new(len, self.inverse)) as Arc<FFT<T>>,
            Plan::Butterfly(len) => {
                butterfly!(len, self.inverse)
            }
            Plan::MixedRadix { left_fft, right_fft } => {
                let left_fft = self.construct_fft(*left_fft);
                let right_fft = self.construct_fft(*right_fft);
                Arc::new(MixedRadix::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Plan::GoodThomas { left_fft, right_fft } => {
                let left_fft = self.construct_fft(*left_fft);
                let right_fft = self.construct_fft(*right_fft);
                Arc::new(GoodThomasAlgorithm::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Plan::MixedRadixDoubleButterfly(left_len, right_len) => {
                let left_fft = self.construct_butterfly(left_len);
                let right_fft = self.construct_butterfly(right_len);
                Arc::new(MixedRadixDoubleButterfly::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Plan::GoodThomasDoubleButterfly(left_len, right_len) => {
                let left_fft = self.construct_butterfly(left_len);
                let right_fft = self.construct_butterfly(right_len);
                Arc::new(GoodThomasAlgorithmDoubleButterfly::new(left_fft, right_fft)) as Arc<FFT<T>>
            },
            Plan::Rader { len, inner_fft } => {
                let inner_fft = self.construct_fft(*inner_fft);
                Arc::new(RadersAlgorithm::new(len, inner_fft)) as Arc<FFT<T>>
            },
            Plan::Bluestein { len , inner_fft } => {
                let inner_fft = self.construct_fft(*inner_fft);
                Arc::new(Bluesteins::new(len, inner_fft)) as Arc<FFT<T>>
            },
        }
    }

    // Create a butterfly
    fn construct_butterfly(&mut self, len: usize) -> Arc<FFTButterfly<T>> {
        let inverse = self.inverse;
        let instance = self.butterfly_cache.entry(len).or_insert_with(|| 
            butterfly!(len, inverse)
        );
        Arc::clone(instance)
    }
}



#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_plan_trivial() {
        // Length 0 and 1 should use DFT
        let mut planner = FFTplanner::<f64>::new(false);
        for len in 0..2 {
            let plan = planner.make_plan_for_len(len);
            assert_eq!(plan, Plan::DFT(len));
        }
    }

    #[test]
    fn test_plan_mediumpoweroftwo() {
        // Powers of 2 between 64 and 32768 should use Radix4
        let mut planner = FFTplanner::<f64>::new(false);
        for pow in 6..16 {
            let len = 1 << pow;
            let plan = planner.make_plan_for_len(len);
            assert_eq!(plan, Plan::Radix4(len));
        }
    }

    #[test]
    fn test_plan_largepoweroftwo() {
        // Powers of 2 from 65536 and up should use MixedRadix
        let mut planner = FFTplanner::<f64>::new(false);
        for pow in 17..32 {
            let len = 1 << pow;
            let plan = planner.make_plan_for_len(len);
            assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
        }
    }

    #[test]
    fn test_plan_butterflies() {
        // Check that all butterflies are used
        let mut planner = FFTplanner::<f64>::new(false);
        for len in [2,3,4,5,6,7,8,16,32].iter() {
            let plan = planner.make_plan_for_len(*len);
            assert_eq!(plan, Plan::Butterfly(*len));
        }
    }

    #[test]
    fn test_plan_mixedradix() {
        // Products of several different primes should become MixedRadix
        let mut planner = FFTplanner::<f64>::new(false);
        for pow2 in 1..3 {
            for pow3 in 1..3 {
                for pow5 in 1..3 {
                    for pow7 in 1..3 {
                        let len = 2usize.pow(pow2) * 3usize.pow(pow3) * 5usize.pow(pow5) * 7usize.pow(pow7);
                        let plan = planner.make_plan_for_len(len);
                        assert!(is_mixedradix(&plan), "Expected MixedRadix, got {:?}", plan);
                    }
                }
            }
        }
    }

    fn is_mixedradix(plan: &Plan) -> bool {
        match plan {
            &Plan::MixedRadix{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_mixedradixbutterfly() {
        // Products of two existing butterfly lengths that have a common divisor >1, and isn't a power of 2 should be MixedRadixDoubleButterfly
        let mut planner = FFTplanner::<f64>::new(false);
        for len in [4*6, 3*6, 3*3].iter() {
            let plan = planner.make_plan_for_len(*len);
            assert!(is_mixedradixbutterfly(&plan), "Expected MixedRadixDoubleButterfly, got {:?}", plan);
        }
    }

    fn is_mixedradixbutterfly(plan: &Plan) -> bool {
        match plan {
            &Plan::MixedRadixDoubleButterfly{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_goodthomasbutterfly() {
        let mut planner = FFTplanner::<f64>::new(false);
        for len in [3*4, 3*5, 3*7, 5*7].iter() {
            let plan = planner.make_plan_for_len(*len);
            assert!(is_goodthomasbutterfly(&plan), "Expected GoodThomasDoubleButterfly, got {:?}", plan);
        }
    }

    fn is_goodthomasbutterfly(plan: &Plan) -> bool {
        match plan {
            &Plan::GoodThomasDoubleButterfly{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_plan_bluestein() {
        let primes: [usize; 5] = [179, 359, 719, 1439, 2879];

        let mut planner = FFTplanner::<f64>::new(false);
        for len in primes.iter() {
            let plan = planner.make_plan_for_len(*len);
            assert!(is_bluesteins(&plan), "Expected Bluesteins, got {:?}", plan);
        }
    }

    fn is_bluesteins(plan: &Plan) -> bool {
        match plan {
            &Plan::Bluestein{..} => true,
            _ => false,
        }
    }

    #[test]
    fn test_dft_cost() {
        let lengths: [usize; 3] = [2, 4, 6];
        for len in lengths.iter() {
            let cost = 4*len.pow(2);
            let plan = Plan::DFT(*len);
            assert_eq!(plan.cost(), cost);
        }
    }
}
