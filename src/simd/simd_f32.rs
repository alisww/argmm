use crate::generic::{simple_argmax, simple_argmin};
use crate::task::{find_final_index_max, find_final_index_min, split_array};

pub fn argmin_f32(arr: &[f32]) -> Option<usize> {
    if is_x86_feature_detected!("sse") {
        // if is_x86_feature_detected!("avx2") {
        //     match split_array(arr, 8) {
        //         (Some(rem), Some(sim)) => {
        //             let rem_min_index = simple_argmin(rem);
        //             let rem_result = (rem[rem_min_index], rem_min_index);
        //             let sim_result = unsafe { core_argmin_256(sim, rem.len()) };
        //             find_final_index_min(rem_result, sim_result)
        //         }
        //         (Some(rem), None) => Some(simple_argmin(rem)),
        //         (None, Some(sim)) => {
        //             let sim_result = unsafe { core_argmin_256(sim, 0) };
        //             Some(sim_result.1)
        //         }
        //         (None, None) => None,
        //     }
        // } else {
        match split_array(arr, 4) {
            (Some(rem), Some(sim)) => {
                let rem_min_index = simple_argmin(rem);
                let rem_result = (rem[rem_min_index], rem_min_index);
                let sim_result = unsafe { core_argmin_128(sim, rem.len()) };
                find_final_index_min(rem_result, sim_result)
            }
            (Some(rem), None) => Some(simple_argmin(rem)),
            (None, Some(sim)) => {
                let sim_result = unsafe { core_argmin_128(sim, 0) };
                Some(sim_result.1)
            }
            (None, None) => None,
        }
    } else {
        Some(simple_argmin(arr))
    }
}

unsafe fn core_argmin_128(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let offset = _mm_set1_ps(rem_offset as f32);
    let mut index_low = _mm_add_ps(_mm_set_ps(3.0, 2.0, 1.0, 0.0), offset);

    let increment = _mm_set1_ps(4.0);
    let mut new_index_low = index_low;

    let mut values_low = _mm_loadu_ps(sim_arr.as_ptr() as *const f32);

    sim_arr.chunks_exact(4).skip(1).for_each(|step| {
        new_index_low = _mm_add_ps(new_index_low, increment);

        let new_values = _mm_loadu_ps(step.as_ptr() as *const f32);
        let lt_mask = _mm_cmplt_ps(new_values, values_low);

        values_low = _mm_min_ps(new_values, values_low);
        index_low = _mm_or_ps(
            _mm_and_ps(new_index_low, lt_mask),
            _mm_andnot_ps(lt_mask, index_low),
        );
    });

    let highpack = _mm_unpackhi_ps(values_low, values_low);
    let lowpack = _mm_unpacklo_ps(values_low, values_low);
    let lowest = _mm_min_ps(highpack, lowpack);

    let highpack = _mm_unpackhi_ps(lowest, lowest);
    let lowpack = _mm_unpacklo_ps(lowest, lowest);
    let lowest = _mm_min_ps(highpack, lowpack);

    let low_mask = _mm_cmpeq_ps(lowest, values_low);

    index_low = _mm_or_ps(
        _mm_and_ps(index_low, low_mask),
        _mm_andnot_ps(low_mask, _mm_set1_ps(std::f32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128, [f32; 4]>(values_low);
    let index_array = std::mem::transmute::<__m128, [f32; 4]>(index_low);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn core_argmin_256(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let offset = _mm256_set1_ps(rem_offset as f32);
    let mut index_low = _mm256_add_ps(
        _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0),
        offset,
    );

    let increment = _mm256_set1_ps(8.0);
    let mut new_index_low = index_low;
    let mut values_low = _mm256_loadu_ps(sim_arr.as_ptr() as *const f32);

    sim_arr.chunks_exact(8).skip(1).for_each(|step| {
        new_index_low = _mm256_add_ps(new_index_low, increment);

        let new_values = _mm256_loadu_ps(step.as_ptr() as *const f32);
        let lt_mask = _mm256_cmp_ps(new_values, values_low, 17); // _CMP_LT_OQ

        values_low = _mm256_blendv_ps(values_low, new_values, lt_mask);
        index_low = _mm256_blendv_ps(index_low, new_index_low, lt_mask);
    });

    let highpack = _mm256_unpackhi_ps(values_low, values_low);
    let lowpack = _mm256_unpacklo_ps(values_low, values_low);
    let lowest = _mm256_min_ps(highpack, lowpack);

    let highpack = _mm256_unpackhi_ps(lowest, lowest);
    let lowpack = _mm256_unpacklo_ps(lowest, lowest);
    let lowest = _mm256_min_ps(highpack, lowpack);

    let low_mask = _mm256_cmp_ps(lowest, values_low, 0); // _CMP_EQ_OQ

    index_low = _mm256_or_ps(
        _mm256_and_ps(index_low, low_mask),
        _mm256_andnot_ps(low_mask, _mm256_set1_ps(std::f32::MAX)),
    );

    let value_array = std::mem::transmute::<__m256, [f32; 8]>(values_low);
    let index_array = std::mem::transmute::<__m256, [f32; 8]>(index_low);

    let min_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(min_index);
    let index = *index_array.get_unchecked(min_index);

    (value, index as usize)
}

pub fn argmax_f32(arr: &[f32]) -> Option<usize> {
    match split_array(arr, 4) {
        (Some(rem), Some(sim)) => {
            let rem_min_index = simple_argmax(rem);
            let rem_result = (rem[rem_min_index], rem_min_index);
            let sim_result = unsafe { core_argmax(sim, rem.len()) };
            find_final_index_max(rem_result, sim_result)
        }
        (Some(rem), None) => Some(simple_argmax(rem)),
        (None, Some(sim)) => {
            let sim_result = unsafe { core_argmax(sim, 0) };
            Some(sim_result.1)
        }
        (None, None) => None,
    }
}

unsafe fn core_argmax(sim_arr: &[f32], rem_offset: usize) -> (f32, usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let offset = _mm_set1_ps(rem_offset as f32);
    let mut index_high = _mm_add_ps(_mm_set_ps(3.0, 2.0, 1.0, 0.0), offset);
    let mut new_index_high = index_high;

    let increment = _mm_set1_ps(4.0);

    let mut values_high = _mm_loadu_ps(sim_arr.as_ptr() as *const f32);

    sim_arr.chunks_exact(4).skip(1).for_each(|step| {
        new_index_high = _mm_add_ps(new_index_high, increment);

        let new_values = _mm_loadu_ps(step.as_ptr() as *const f32);
        let gt_mask = _mm_cmpgt_ps(new_values, values_high);

        values_high = _mm_max_ps(new_values, values_high);
        index_high = _mm_or_ps(
            _mm_and_ps(new_index_high, gt_mask),
            _mm_andnot_ps(gt_mask, index_high),
        );
    });

    let highpack = _mm_unpackhi_ps(values_high, values_high);
    let lowpack = _mm_unpacklo_ps(values_high, values_high);
    let highest = _mm_max_ps(highpack, lowpack);

    let highpack = _mm_unpackhi_ps(highest, highest);
    let lowpack = _mm_unpacklo_ps(highest, highest);
    let highest = _mm_max_ps(highpack, lowpack);

    let high_mask = _mm_cmpeq_ps(highest, values_high);

    index_high = _mm_or_ps(
        _mm_and_ps(index_high, high_mask),
        _mm_andnot_ps(high_mask, _mm_set1_ps(std::f32::MAX)),
    );

    let value_array = std::mem::transmute::<__m128, [f32; 4]>(values_high);
    let index_array = std::mem::transmute::<__m128, [f32; 4]>(index_high);

    let max_index = simple_argmin(&index_array);
    let value = *value_array.get_unchecked(max_index);
    let index = *index_array.get_unchecked(max_index);

    (value, index as usize)
}

#[cfg(test)]
mod tests {
    use super::{argmax_f32, argmin_f32, simple_argmax, simple_argmin};
    use rand::{thread_rng, Rng};
    use rand_distr::Uniform;

    fn get_array_f32(n: usize) -> Vec<f32> {
        let rng = thread_rng();
        let uni = Uniform::new_inclusive(std::f32::MIN, std::f32::MAX);
        rng.sample_iter(uni).take(n).collect()
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = get_array_f32(1025);
        assert_eq!(data.len() % 4, 1);

        let min_index = argmin_f32(&data).unwrap();
        let max_index = argmax_f32(&data).unwrap();
        let argmin_index = simple_argmin(&data);
        let argmax_index = simple_argmax(&data);

        assert_eq!(argmin_index, min_index);
        assert_eq!(argmax_index, max_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [
            10.,
            std::f32::MAX,
            6.,
            std::f32::NEG_INFINITY,
            std::f32::NEG_INFINITY,
            std::f32::MAX,
            10_000.0,
        ];
        let argmin_index = simple_argmin(&data);
        let argmin_simd_index = argmin_f32(&data).unwrap();
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, 3);

        let argmax_index = simple_argmax(&data);
        let argmax_simd_index = argmax_f32(&data).unwrap();
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, 1);
    }
}
