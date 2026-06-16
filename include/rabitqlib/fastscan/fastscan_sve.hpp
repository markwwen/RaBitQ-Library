#pragma once

#include <arm_sve.h>

#include <cstddef>
#include <cstdint>

#include "rabitqlib/fastscan/fastscan_scalar.hpp"

namespace rabitqlib::fastscan::sve {

using scalar::pack_codes;
using scalar::pack_lut;

inline void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    alignas(64) uint16_t acc[kBatchSize] = {};
    alignas(64) uint8_t lo_buf[16];
    alignas(64) uint8_t hi_buf[16];

    size_t num_codebook = dim >> 2;
    svbool_t pg16 = svwhilelt_b8(static_cast<uint64_t>(0), static_cast<uint64_t>(16));

    for (size_t m = 0; m < num_codebook; ++m) {
        const uint8_t* code_ptr = codes + (m * 16);
        const uint8_t* lut_ptr = lp_table + (m * 16);

        svuint8_t c = svld1_u8(pg16, code_ptr);
        svuint8_t lut = svld1_u8(pg16, lut_ptr);

        svuint8_t lo_idx = svand_n_u8_x(pg16, c, 0x0f);
        svuint8_t hi_idx = svand_n_u8_x(pg16, svlsr_n_u8_x(pg16, c, 4), 0x0f);

        svuint8_t lo_val = svtbl_u8(lut, lo_idx);
        svuint8_t hi_val = svtbl_u8(lut, hi_idx);

        svst1_u8(pg16, lo_buf, lo_val);
        svst1_u8(pg16, hi_buf, hi_val);

        for (size_t j = 0; j < 16; ++j) {
            size_t permuted_index = static_cast<size_t>(kPerm0[j]);
            acc[permuted_index] += lo_buf[j];
            acc[permuted_index + 16] += hi_buf[j];
        }
    }

    for (size_t i = 0; i < kBatchSize; ++i) {
        result[i] = acc[i];
    }
}

}  // namespace rabitqlib::fastscan::sve