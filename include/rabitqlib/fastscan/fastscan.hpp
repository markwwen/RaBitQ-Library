#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/fastscan/fastscan_common.hpp"
#include "rabitqlib/fastscan/fastscan_scalar.hpp"

#if defined(__AVX512BW__) || defined(__AVX2__)
#include "rabitqlib/fastscan/fastscan_x86.hpp"
#elif defined(__ARM_FEATURE_SVE)
#include "rabitqlib/fastscan/fastscan_sve.hpp"
#endif

namespace rabitqlib::fastscan {

inline void pack_codes(
    size_t padded_dim, const uint8_t* quantization_code, size_t num, uint8_t* blocks
) {
    scalar::pack_codes(padded_dim, quantization_code, num, blocks);
}

inline void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
#if defined(__AVX512BW__) || defined(__AVX2__)
    x86::accumulate(codes, lp_table, result, dim);
#elif defined(__ARM_FEATURE_SVE)
    sve::accumulate(codes, lp_table, result, dim);
#else
    scalar::accumulate(codes, lp_table, result, dim);
#endif
}

template <typename T>
inline void pack_lut(size_t dim, const T* __restrict__ query, T* __restrict__ lut) {
    scalar::pack_lut(dim, query, lut);
}

}  // namespace rabitqlib::fastscan