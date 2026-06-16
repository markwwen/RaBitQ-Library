#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::fastscan {

constexpr static size_t kBatchSize = 32;  // number of vectors in each batch

constexpr static std::array<int, 16> kPos = {
    3 /*0000*/,
    3 /*0001*/,
    2 /*0010*/,
    3 /*0011*/,
    1 /*0100*/,
    3 /*0101*/,
    2 /*0110*/,
    3 /*0111*/,
    0 /*1000*/,
    3 /*1001*/,
    2 /*1010*/,
    3 /*1011*/,
    1 /*1100*/,
    3 /*1101*/,
    2 /*1110*/,
    3 /*1111*/,
};

constexpr static std::array<int, 16> kPerm0 = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
};

template <typename T, class TA>
static inline void get_column(
    const T* src, size_t rows, size_t cols, size_t row, size_t col, TA& dest
) {
    size_t k = 0;
    size_t max_k = std::min(rows - row, dest.size());
    for (; k < max_k; ++k) {
        dest[k] = src[((k + row) * cols) + col];
    }
    if (k < dest.size()) {
        std::fill(dest.begin() + k, dest.end(), 0);
    }
}

}  // namespace rabitqlib::fastscan