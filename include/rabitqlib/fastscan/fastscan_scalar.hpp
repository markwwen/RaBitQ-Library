#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/fastscan/fastscan_common.hpp"

namespace rabitqlib::fastscan::scalar {

inline void pack_codes(
    size_t padded_dim, const uint8_t* quantization_code, size_t num, uint8_t* blocks
) {
    size_t num_rd = (num + 31) & ~31;
    size_t cols = padded_dim / 8;

    std::array<uint8_t, 32> col;
    std::array<uint8_t, 32> col_0;
    std::array<uint8_t, 32> col_1;

    for (size_t row = 0; row < num_rd; row += kBatchSize) {
        for (size_t i = 0; i < cols; ++i) {
            get_column(quantization_code, num, cols, row, i, col);
            for (size_t j = 0; j < 32; ++j) {
                col_0[j] = col[j] >> 4;
                col_1[j] = col[j] & 15;
            }
            for (size_t j = 0; j < 16; ++j) {
                uint8_t val0 = col_0[kPerm0[j]] | (col_0[kPerm0[j] + 16] << 4);
                uint8_t val1 = col_1[kPerm0[j]] | (col_1[kPerm0[j] + 16] << 4);
                blocks[j] = val0;
                blocks[j + 16] = val1;
            }
            blocks += 32;
        }
    }
}

inline void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    for (size_t i = 0; i < kBatchSize; ++i) {
        result[i] = 0;
    }

    size_t num_codebook = dim >> 2;
    for (size_t m = 0; m < num_codebook; ++m) {
        const uint8_t* code_ptr = codes + (m * 16);
        const uint8_t* lut_ptr = lp_table + (m * 16);
        for (size_t j = 0; j < 16; ++j) {
            uint8_t packed_code = code_ptr[j];
            size_t permuted_index = static_cast<size_t>(kPerm0[j]);
            result[permuted_index] += lut_ptr[packed_code & 0x0f];
            result[permuted_index + 16] += lut_ptr[packed_code >> 4];
        }
    }
}

template <typename T>
inline void pack_lut(size_t dim, const T* __restrict__ query, T* __restrict__ lut) {
    size_t num_codebook = dim >> 2;
    for (size_t i = 0; i < num_codebook; ++i) {
        lut[0] = 0;
        for (size_t j = 1; j < 16; ++j) {
            lut[j] = lut[j - LOWBIT(j)] + query[kPos[j]];
        }
        lut += 16;
        query += 4;
    }
}

}  // namespace rabitqlib::fastscan::scalar