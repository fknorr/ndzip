#pragma once

#include "common.hh"

#include <stdexcept>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef NDZIP_OPENMP_SUPPORT
#include <atomic>
#include <omp.h>
#include <queue>

#include <boost/container/static_vector.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/thread/thread.hpp>
#endif


namespace ndzip::detail::cpu {

constexpr static const size_t simd_width_bytes = 32;

template<typename T>
[[gnu::always_inline]] T *assume_simd_aligned(T *x) {
    assert(reinterpret_cast<uintptr_t>(x) % simd_width_bytes == 0);
    return static_cast<T *>(__builtin_assume_aligned(x, simd_width_bytes));
}

template<typename T>
class simd_aligned_buffer {
    static_assert(simd_width_bytes >= alignof(T) && simd_width_bytes % alignof(T) == 0);

  public:
    simd_aligned_buffer() = default;

    explicit simd_aligned_buffer(size_t size) {
        assert(size % simd_width_bytes == 0);
        _memory = std::aligned_alloc(simd_width_bytes, size * sizeof(T));
        if (!_memory) { throw std::bad_alloc(); }
    }

    simd_aligned_buffer(simd_aligned_buffer &&other) noexcept { *this = std::move(other); }

    simd_aligned_buffer &operator=(simd_aligned_buffer &&other) noexcept {
        std::free(_memory);
        _memory = other._memory;
        other._memory = nullptr;
        return *this;
    }

    ~simd_aligned_buffer() { std::free(_memory); }

    explicit operator bool() const { return _memory != nullptr; }

    T *data() { return assume_simd_aligned(static_cast<T *>(_memory)); }

    const T *data() const { return assume_simd_aligned(static_cast<const T *>(_memory)); }

    T &operator[](size_t i) { return data()[i]; };

    const T &operator[](size_t i) const { return data()[i]; };

  private:
    void *_memory = nullptr;
};

template<typename Profile>
[[gnu::noinline]] void load_hypercube(const extent<Profile::dimensions> &hc_offset,
        const slice<const typename Profile::data_type, Profile::dimensions> &data,
        typename Profile::bits_type *cube) {
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

    map_hypercube_slices<Profile>(
            hc_offset, data, cube, [](const data_type *src, bits_type *dest, size_t n_elems) {
                memcpy(assume_simd_aligned(dest), src, n_elems * sizeof(data_type));
            });
}

template<typename Profile>
[[gnu::noinline]] void store_hypercube(const extent<Profile::dimensions> &hc_offset,
        const typename Profile::bits_type *cube,
        const slice<typename Profile::data_type, Profile::dimensions> &data) {
    using data_type = typename Profile::data_type;
    using bits_type = typename Profile::bits_type;

    map_hypercube_slices<Profile>(
            hc_offset, data, cube, [](data_type *dest, const bits_type *src, size_t n_elems) {
                memcpy(dest, assume_simd_aligned(src), n_elems * sizeof(data_type));
            });
}


#ifdef __AVX2__

[[gnu::always_inline]] inline __m256i load_aligned_256(const void *p) {
    return _mm256_load_si256(static_cast<const __m256i *>(p));
}

[[gnu::always_inline]] inline __m256i load_unaligned_256(const void *p) {
    return _mm256_loadu_si256(static_cast<const __m256i *>(p));
}

[[gnu::always_inline]] inline void store_aligned_256(void *p, __m256i x) {
    _mm256_store_si256(static_cast<__m256i *>(p), x);
}

template<typename Bits>
[[gnu::always_inline]] __m256i add_packed(__m256i a, __m256i b) {
    if constexpr (bitsof<Bits> == 32) {
        return _mm256_add_epi32(a, b);
    } else {
        return _mm256_add_epi64(a, b);
    }
}

template<typename Bits>
[[gnu::always_inline]] __m256i subtract_packed(__m256i a, __m256i b) {
    if constexpr (bitsof<Bits> == 32) {
        return _mm256_sub_epi32(a, b);
    } else {
        return _mm256_sub_epi64(a, b);
    }
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] void block_transform_horizontal_avx2(Bits *line) {
    constexpr auto n_256bit_lanes = sizeof(Bits) * SideLength / simd_width_bytes;
    constexpr auto words_per_256bit_lane = simd_width_bytes / sizeof(Bits);

    // TODO is there a better option than the setr sequence? vpmaskmov and overflowing vmovdqu +
    // blend are both slower.
    __m256i top_n;
    if constexpr (bitsof<Bits> == 32) {
        top_n = _mm256_setr_epi32(0, line[0], line[1], line[2], line[3], line[4], line[5], line[6]);
    } else {
        top_n = _mm256_setr_epi64x(0, line[0], line[1], line[2]);
    }

    for (unsigned j = 0; j < n_256bit_lanes - 1; ++j) {
        auto top = top_n;
        top_n = load_unaligned_256(line + (j + 1) * words_per_256bit_lane - 1);
        auto bottom = load_aligned_256(line + j * words_per_256bit_lane);
        store_aligned_256(line + j * words_per_256bit_lane, subtract_packed<Bits>(bottom, top));
    }
    auto bottom = load_aligned_256(line + (n_256bit_lanes - 1) * words_per_256bit_lane);
    store_aligned_256(line + (n_256bit_lanes - 1) * words_per_256bit_lane,
            subtract_packed<Bits>(bottom, top_n));
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] inline void block_transform_vertical_avx2(Bits *x) {
    // TODO investigate whether SW pipelining leads to spilling / reloading for 2D double (2*64/4 =
    // 32 YMM registers)
    constexpr auto n_256bit_lanes = sizeof(Bits) * SideLength / simd_width_bytes;

    __m256i lanes_a[n_256bit_lanes];
    __m256i lanes_b[n_256bit_lanes];
    __builtin_memcpy(lanes_b, assume_simd_aligned(x), sizeof lanes_b);

    for (size_t i = 1; i < SideLength; ++i) {
        __builtin_memcpy(lanes_a, lanes_b, sizeof lanes_b);
        __builtin_memcpy(lanes_b, assume_simd_aligned(x + i * SideLength), sizeof lanes_b);
        for (size_t j = 0; j < n_256bit_lanes; ++j) {
            lanes_a[j] = subtract_packed<Bits>(lanes_b[j], lanes_a[j]);
        }
        __builtin_memcpy(assume_simd_aligned(x + i * SideLength), lanes_a, sizeof lanes_a);
    }
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] inline void block_transform_planes_avx2(Bits *x) {
    constexpr auto n_256bit_lanes = sizeof(Bits) * SideLength / simd_width_bytes;
    constexpr auto n = SideLength;

    for (size_t i = 0; i < n * n; i += n) {
        __m256i lanes_a[n_256bit_lanes];
        __m256i lanes_b[n_256bit_lanes];
        __builtin_memcpy(lanes_b, assume_simd_aligned(x + i), sizeof lanes_b);
        for (size_t j = n * n; j < n * n * n; j += n * n) {
            __builtin_memcpy(lanes_a, lanes_b, sizeof lanes_b);
            __builtin_memcpy(lanes_b, assume_simd_aligned(x + i + j), sizeof lanes_b);
            for (size_t k = 0; k < n_256bit_lanes; ++k) {
                lanes_a[k] = subtract_packed<Bits>(lanes_b[k], lanes_a[k]);
            }
            __builtin_memcpy(assume_simd_aligned(x + i + j), lanes_a, sizeof lanes_a);
        }
    }
}

template<typename Profile>
void block_transform_avx2(typename Profile::bits_type *x) {
    constexpr size_t dims = Profile::dimensions;
    constexpr size_t side_length = Profile::hypercube_side_length;

    x = assume_simd_aligned(x);

    for (size_t i = 0; i < ipow(side_length, Profile::dimensions); ++i) {
        x[i] = rotate_left_1(x[i]);
    }

    if constexpr (dims == 1) {
        block_transform_horizontal_avx2<side_length>(x);
    } else if constexpr (dims == 2) {
        for (size_t i = 0; i < side_length; ++i) {
            block_transform_horizontal_avx2<side_length>(x + i * side_length);
        }
        block_transform_vertical_avx2<side_length>(x);
    } else if constexpr (dims == 3) {
        for (size_t i = 0; i < side_length * side_length * side_length; i += side_length) {
            block_transform_horizontal_avx2<side_length>(x + i);
        }
        for (size_t i = 0; i < side_length * side_length * side_length;
                i += side_length * side_length) {
            block_transform_vertical_avx2<side_length>(x + i);
        }
        block_transform_planes_avx2<side_length>(x);
    }

    for (size_t i = 0; i < ipow(side_length, Profile::dimensions); ++i) {
        x[i] = complement_negative(x[i]);
    }
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] inline void inverse_block_transform_horizontal_sequential(Bits *x) {
    for (size_t i = 1; i < SideLength; ++i) {
        x[i] += x[i - 1];
    }
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] inline void inverse_block_transform_horizontal_interleaved(Bits *x) {
    constexpr auto interleave = 4;
    Bits vec[interleave];
    for (size_t i = 0; i < ipow(SideLength, 2); i += SideLength * interleave) {
        for (size_t k = 0; k < interleave; ++k) {
            vec[k] = x[i + k * SideLength];
        }
        for (size_t j = 1; j < SideLength; ++j) {
            for (size_t k = 0; k < interleave; ++k) {
                vec[k] += x[i + j + k * SideLength];
            }
            for (size_t k = 0; k < interleave; ++k) {
                x[i + j + k * SideLength] = vec[k];
            }
        }
    }
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] inline void inverse_block_transform_vertical_avx2(Bits *x) {
    constexpr auto n_256bit_lanes = sizeof(Bits) * SideLength / simd_width_bytes;
    constexpr auto words_per_256bit_lane = simd_width_bytes / sizeof(Bits);

    __m256i lanes_a[n_256bit_lanes];
    __builtin_memcpy(lanes_a, assume_simd_aligned(x), sizeof lanes_a);

    for (size_t i = 1; i < SideLength; ++i) {
        for (size_t j = 0; j < n_256bit_lanes; ++j) {
            __m256i b = load_aligned_256(x + i * SideLength + j * words_per_256bit_lane);
            lanes_a[j] = add_packed<Bits>(lanes_a[j], b);
        }
        __builtin_memcpy(assume_simd_aligned(x + i * SideLength), lanes_a, sizeof lanes_a);
    }
}

template<unsigned SideLength, typename Bits>
[[gnu::always_inline]] inline void inverse_block_transform_planes_avx2(Bits *x) {
    constexpr auto n_256bit_lanes = sizeof(Bits) * SideLength / simd_width_bytes;
    constexpr auto words_per_256bit_lane = simd_width_bytes / sizeof(Bits);
    constexpr auto n = SideLength;

    for (size_t i = 0; i < n * n; i += n) {
        __m256i lanes_a[n_256bit_lanes];
        __builtin_memcpy(lanes_a, assume_simd_aligned(x + i), sizeof lanes_a);
        for (size_t j = n * n; j < n * n * n; j += n * n) {
            for (size_t k = 0; k < n_256bit_lanes; ++k) {
                __m256i b = load_aligned_256(x + i + j + k * words_per_256bit_lane);
                lanes_a[k] = add_packed<Bits>(lanes_a[k], b);
            }
            __builtin_memcpy(assume_simd_aligned(x + i + j), lanes_a, sizeof lanes_a);
        }
    }
}

template<typename Profile>
void inverse_block_transform_avx2(typename Profile::bits_type *x) {
    constexpr size_t dims = Profile::dimensions;
    constexpr size_t side_length = Profile::hypercube_side_length;

    x = assume_simd_aligned(x);

    for (size_t i = 0; i < ipow(side_length, Profile::dimensions); ++i) {
        x[i] = complement_negative(x[i]);
    }

    if constexpr (dims == 1) {
        inverse_block_transform_horizontal_sequential<side_length>(x);
    } else if constexpr (dims == 2) {
        inverse_block_transform_horizontal_interleaved<side_length>(x);
        inverse_block_transform_vertical_avx2<side_length>(x);
    } else if constexpr (dims == 3) {
        for (size_t i = 0; i < ipow(side_length, 3); i += ipow(side_length, 2)) {
            inverse_block_transform_horizontal_interleaved<side_length>(x + i);
        }
        for (size_t i = 0; i < ipow(side_length, 3); i += ipow(side_length, 2)) {
            inverse_block_transform_vertical_avx2<side_length>(x + i);
        }
        inverse_block_transform_planes_avx2<side_length>(x);
    }

    for (size_t i = 0; i < ipow(side_length, Profile::dimensions); ++i) {
        x[i] = rotate_right_1(x[i]);
    }
}

#endif  // __AVX2__

template<typename Profile>
[[gnu::noinline]] void block_transform(typename Profile::bits_type *x) {
#ifdef __AVX2__
    block_transform_avx2<Profile>(x);
#else
    ndzip::detail::block_transform(x, Profile::dimensions, Profile::hypercube_side_length);
#endif
}

template<typename Profile>
[[gnu::noinline]] void inverse_block_transform(typename Profile::bits_type *x) {
#ifdef __AVX2__
    inverse_block_transform_avx2<Profile>(x);
#else
    ndzip::detail::inverse_block_transform(x, Profile::dimensions, Profile::hypercube_side_length);
#endif
}


template<typename T>
bool all_zero(const T *u) {
#ifdef __AVX2__
    __m256i por{};
    for (unsigned i = 0; i < bitsof<T> * sizeof(T) / 32; ++i) {
        por |= _mm256_load_si256(reinterpret_cast<const __m256i *>(u) + i);
    }
    return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi32(por, __m256i{})))
            == ~uint32_t{};
#else
    u = assume_simd_aligned(u);
    bool zero = true;
    for (unsigned j = 0; j < bitsof<T>; ++j) {
        zero &= u[j] == 0;
    }
    return zero;
#endif
}


template<typename T>
[[gnu::always_inline]] void transpose_bits_trivial(const T *__restrict vs, T *__restrict out) {
    for (unsigned i = 0; i < bitsof<T>; ++i) {
        out[i] = 0;
        for (unsigned j = 0; j < bitsof<T>; ++j) {
            out[i] |= ((vs[j] >> (bitsof<T> - 1 - i)) & 1u) << (bitsof<T> - 1 - j);
        }
    }
}

#ifdef __AVX2__

[[gnu::always_inline]] inline void transpose_bits_avx2(
        const uint32_t *__restrict vs, uint32_t *__restrict out) {
    __m256i unpck0[4];
    __builtin_memcpy(unpck0, assume_simd_aligned(vs), sizeof unpck0);

    // 1. In order to generate 32 transpositions using only 32 *movemask* instructions,
    // first shuffle bytes so that the i-th 256-bit vector contains the i-th byte of
    // each float

    __m256i shuf[4];
    {
        // interpret each 128-bit lane as a 4x4 matrix and transpose it
        // also correct endianess for later
        uint8_t idx8[] = {
                12,
                8,
                4,
                0,
                13,
                9,
                5,
                1,
                14,
                10,
                6,
                2,
                15,
                11,
                7,
                3,
                12,
                8,
                4,
                0,
                13,
                9,
                5,
                1,
                14,
                10,
                6,
                2,
                15,
                11,
                7,
                3,
        };
        __m256i idx;
        __builtin_memcpy(&idx, idx8, sizeof idx);
        for (unsigned i = 0; i < 4; ++i) {
            shuf[i] = _mm256_shuffle_epi8(unpck0[i], idx);
        }
    }

    __m256i perm[4];
    {
        // interleave doublewords within each 256-bit vector
        // quadword i will contain elements {i, i+4, i+8, ... i+28}
        uint32_t idx32[] = {7, 3, 6, 2, 5, 1, 4, 0};
        __m256i idx;
        __builtin_memcpy(&idx, idx32, sizeof idx);
        for (unsigned i = 0; i < 4; ++i) {
            perm[i] = _mm256_permutevar8x32_epi32(shuf[i], idx);
        }
    }

    __m256i unpck1[4];
    for (unsigned i = 0; i < 4; i += 2) {
        // interleave quadwords of neighboring 256-bit vectors
        // each double-quadword will contain elements with stride 4
        unpck1[i + 1] = _mm256_unpackhi_epi64(perm[i + 1], perm[i + 0]);
        unpck1[i + 0] = _mm256_unpacklo_epi64(perm[i + 1], perm[i + 0]);
    }

    // is _mm256_inserti128_si256 faster here?
    __m256i perm2[4] = {
            // combine matching 128-bit lanes
            _mm256_permute2x128_si256(unpck1[2], unpck1[0], 0x20),
            _mm256_permute2x128_si256(unpck1[3], unpck1[1], 0x20),
            _mm256_permute2x128_si256(unpck1[2], unpck1[0], 0x31),
            _mm256_permute2x128_si256(unpck1[3], unpck1[1], 0x31),
    };

    // 2. Transpose by extracting the 32 MSBs of each byte of each 256-byte vector

    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 8; ++j) {
            out[i * 8 + j] = _mm256_movemask_epi8(perm2[i]);
            perm2[i] = _mm256_slli_epi32(perm2[i], 1);
        }
    }
}

[[gnu::always_inline]] inline void transpose_bits_avx2(
        const uint64_t *__restrict vs, uint64_t *__restrict out) {
    __m256i in[16];
    __builtin_memcpy(in, __builtin_assume_aligned(vs, 32), sizeof in);

    // TODO Clang does a lot of spilling and reloading here because it's running out of YMM
    // registers. GCC less so. Can this be improved, e.g. by merging operations that can operate on
    // a subset of the 16 vectors?

    __m256i unpck0[16];
    for (unsigned i = 0; i < 16; i += 2) {
        unpck0[i + 1] = _mm256_unpackhi_epi8(in[i + 0], in[i + 1]);
        unpck0[i + 0] = _mm256_unpacklo_epi8(in[i + 0], in[i + 1]);
    }

    __m256i unpck1[16];
    for (unsigned i = 0; i < 16; i += 4) {
        for (unsigned j = 0; j < 2; ++j) {
            unpck1[i + 2 * j + 1] = _mm256_unpackhi_epi16(unpck0[i + j], unpck0[i + 2 + j]);
            unpck1[i + 2 * j + 0] = _mm256_unpacklo_epi16(unpck0[i + j], unpck0[i + 2 + j]);
        }
    }

    __m256i unpck2[16];
    for (unsigned i = 0; i < 16; i += 8) {
        for (unsigned j = 0; j < 4; ++j) {
            unpck2[i + 2 * j + 1] = _mm256_unpackhi_epi32(unpck1[i + j], unpck1[i + 4 + j]);
            unpck2[i + 2 * j + 0] = _mm256_unpacklo_epi32(unpck1[i + j], unpck1[i + 4 + j]);
        }
    }

    __m256i unpck3[16];
    for (unsigned i = 0; i < 16; i += 8) {
        for (unsigned j = 0; j < 4; ++j) {
            unpck3[i + 2 * j + 1] = _mm256_unpackhi_epi8(unpck2[i + j], unpck2[i + 4 + j]);
            unpck3[i + 2 * j + 0] = _mm256_unpacklo_epi8(unpck2[i + j], unpck2[i + 4 + j]);
        }
    }

    __m256i perm0[16];
    for (unsigned i = 0; i < 16; ++i) {
        perm0[15 - i] = _mm256_permute4x64_epi64(unpck3[i], 0b00'10'01'11);
    }

    __m256i shuf0[16];
    {
        const uint8_t idx8[] = {
                7,
                6,
                15,
                14,
                5,
                4,
                13,
                12,
                3,
                2,
                11,
                10,
                1,
                0,
                9,
                8,
                7,
                6,
                15,
                14,
                5,
                4,
                13,
                12,
                3,
                2,
                11,
                10,
                1,
                0,
                9,
                8,
        };
        __m256i idx;
        __builtin_memcpy(&idx, idx8, sizeof idx);
        for (unsigned i = 0; i < 8; ++i) {
            shuf0[2 * i] = _mm256_shuffle_epi8(perm0[i], idx);
            shuf0[2 * i + 1] = _mm256_shuffle_epi8(perm0[i + 8], idx);
        }
    }

    auto dwords = reinterpret_cast<uint32_t *>(out);
    for (unsigned i = 0; i < 8; ++i) {
        for (unsigned j = 0; j < 8; ++j) {
            auto half1 = static_cast<uint32_t>(_mm256_movemask_epi8(shuf0[2 * i]));
            auto half2 = static_cast<uint32_t>(_mm256_movemask_epi8(shuf0[2 * i + 1]));
            __builtin_memcpy(dwords + 2 * (i * 8 + j), &half1, 4);
            __builtin_memcpy(dwords + 2 * (i * 8 + j) + 1, &half2, 4);
            shuf0[2 * i] = _mm256_slli_epi32(shuf0[2 * i], 1);
            shuf0[2 * i + 1] = _mm256_slli_epi32(shuf0[2 * i + 1], 1);
        }
    }
}

#endif  // __AVX2__

template<typename T>
[[gnu::noinline]] void transpose_bits(const T *__restrict in, T *__restrict out) {
#ifdef __AVX2__
    return transpose_bits_avx2(in, out);
#else
    return transpose_bits_trivial(in, out);
#endif
}

template<typename T>
[[gnu::always_inline]] size_t compact_zero_words(const T *shifted, std::byte *out0) {
    auto out = out0 + sizeof(T);
    T head = 0;
    for (unsigned i = 0; i < bitsof<T>; ++i) {
        if (shifted[i] != 0) {
            head |= T{1} << i;
            store_aligned(out, shifted[i]);
            out += sizeof(T);
        }
    }
    store_aligned(out0, head);
    return out - out0;
}

template<typename T>
[[gnu::always_inline]] size_t expand_zero_words(const std::byte *in0, T *shifted) {
    auto in = in0 + sizeof(T);
    auto head = load_aligned<T>(in0);
    for (unsigned i = 0; i < bitsof<T>; ++i) {
        if ((head >> i) & T{1}) {
            shifted[i] = load_aligned<T>(in);
            in += sizeof(T);
        } else {
            shifted[i] = 0;
        }
    }
    return in - in0;
}


template<typename Profile>
[[gnu::noinline]] size_t
zero_bit_encode(const typename Profile::bits_type *cube, std::byte *stream) {
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, Profile::dimensions);

    size_t pos = 0;
    for (size_t i = 0; i < hc_size; i += detail::bitsof<bits_type>) {
        if (all_zero(cube + i)) {
            // fast path (all_zero is relatively common, transpose+compact is expensive)
            store_aligned(stream + pos, bits_type{});
            pos += sizeof(bits_type);
        } else {
            alignas(simd_width_bytes) bits_type transposed[detail::bitsof<bits_type>];
            detail::cpu::transpose_bits(cube + i, transposed);
            pos += detail::cpu::compact_zero_words(transposed, stream + pos);
        }
    }

    return pos;
}

template<typename Profile>
[[gnu::noinline]] size_t
zero_bit_decode(const std::byte *stream, typename Profile::bits_type *cube) {
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, Profile::dimensions);

    size_t pos = 0;
    for (size_t i = 0; i < hc_size; i += detail::bitsof<bits_type>) {
        alignas(simd_width_bytes) bits_type transposed[detail::bitsof<bits_type>];
        if (load_aligned<bits_type>(stream + pos) == 0) {
            // fast path (all_zero is relatively common, transpose+compact is expensive)
            memset(cube + i, 0, sizeof transposed);
            pos += sizeof(bits_type);
        } else {
            pos += detail::cpu::expand_zero_words(stream + pos, transposed);
            detail::cpu::transpose_bits(transposed, cube + i);
        }
    }
    return pos;
}

}  // namespace ndzip::detail::cpu

template<typename T, unsigned Dims>
struct ndzip::cpu_encoder<T, Dims>::impl {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    constexpr static auto side_length = profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, profile::dimensions);

    detail::cpu::simd_aligned_buffer<bits_type> cube{hc_size};
};


template<typename T, unsigned Dims>
ndzip::cpu_encoder<T, Dims>::cpu_encoder() : _pimpl(std::make_unique<impl>()) {
}

template<typename T, unsigned Dims>
ndzip::cpu_encoder<T, Dims>::~cpu_encoder() = default;

template<typename T, unsigned Dims>
size_t ndzip::cpu_encoder<T, Dims>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    if (reinterpret_cast<uintptr_t>(stream) % sizeof(bits_type) != 0) {
        throw std::invalid_argument("stream is not properly aligned");
    }

    detail::file<profile> file(data.size());
    size_t stream_pos = file.file_header_length();

    auto &cube = _pimpl->cube;
    file.for_each_hypercube([&](auto hc_offset, auto hc_index) {
        auto header_pos = stream_pos;
        detail::cpu::load_hypercube<profile>(hc_offset, data, cube.data());
        detail::cpu::block_transform<profile>(cube.data());
        stream_pos += detail::cpu::zero_bit_encode<profile>(
                cube.data(), static_cast<std::byte *>(stream) + stream_pos);

        if (hc_index > 0) {
            auto file_offset_address = static_cast<std::byte *>(stream)
                    + (hc_index - 1) * sizeof(detail::file_offset_type);
            auto file_offset = static_cast<detail::file_offset_type>(header_pos);
            detail::store_aligned(file_offset_address, detail::endian_transform(file_offset));
        }
    });

    if (file.num_hypercubes() > 0) {
        auto file_offset_address = static_cast<char *>(stream)
                + (file.num_hypercubes() - 1) * sizeof(detail::file_offset_type);
        detail::store_aligned(file_offset_address, detail::endian_transform(stream_pos));
    }
    stream_pos += detail::pack_border(
            static_cast<char *>(stream) + stream_pos, data, profile::hypercube_side_length);
    return stream_pos;
}


template<typename T, unsigned Dims>
size_t ndzip::cpu_encoder<T, Dims>::decompress(
        const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    if (reinterpret_cast<uintptr_t>(stream) % sizeof(bits_type) != 0) {
        throw std::invalid_argument("stream is not properly aligned");
    }

    detail::file<profile> file(data.size());

    auto &cube = _pimpl->cube;
    size_t stream_pos = file.file_header_length();  // simply skip the header
    file.for_each_hypercube([&](auto hc_offset) {
        stream_pos += detail::cpu::zero_bit_decode<profile>(
                static_cast<const std::byte *>(stream) + stream_pos, cube.data());
        detail::cpu::inverse_block_transform<profile>(cube.data());
        detail::cpu::store_hypercube<profile>(hc_offset, cube.data(), data);
    });
    stream_pos += detail::unpack_border(data, static_cast<const std::byte *>(stream) + stream_pos,
            profile::hypercube_side_length);
    return stream_pos;
}


#if NDZIP_OPENMP_SUPPORT

template<typename T, unsigned Dims>
struct ndzip::mt_cpu_encoder<T, Dims>::impl {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    constexpr static auto side_length = profile::hypercube_side_length;
    constexpr static auto hc_size = detail::ipow(side_length, profile::dimensions);
    constexpr static unsigned num_hcs_per_chunk = 64 / sizeof(data_type);
    constexpr static unsigned num_write_buffers = 30;

    struct cube_buffer {
        alignas(detail::cpu::simd_width_bytes) std::array<bits_type, hc_size> cube;

        bits_type *data() { return detail::cpu::assume_simd_aligned(cube.data()); }

        const bits_type *data() const { return detail::cpu::assume_simd_aligned(cube.data()); }
    };

    struct write_buffer {
        size_t first_hc_index = SIZE_MAX;
        alignas(8) std::array<std::byte,
                profile::compressed_block_size_bound * num_hcs_per_chunk> stream;
        boost::container::static_vector<uint32_t, num_hcs_per_chunk> offsets_after_hcs;

        size_t num_hypercubes() const { return offsets_after_hcs.size(); }

        size_t compressed_size() const { return offsets_after_hcs.back(); }
    };

    struct hc_index_order {
        bool operator()(write_buffer *left, write_buffer *right) const {
            return left->first_hc_index > right->first_hc_index;  // >: priority_queue is a max-heap
        }
    };

    const size_t num_threads;
    std::vector<cube_buffer> thread_cubes{num_threads};
    std::vector<write_buffer> write_buffers{num_write_buffers};
    std::priority_queue<write_buffer *, std::vector<write_buffer *>, hc_index_order>
            write_task_queue;
    boost::lockfree::queue<write_buffer *, boost::lockfree::capacity<num_write_buffers>>
            free_write_buffers;

    impl(std::optional<size_t> opt_num_threads)
        : num_threads(
                opt_num_threads ? opt_num_threads.value() : boost::thread::physical_concurrency()) {
        // priority_queue does not expose vector::reserve, push nonsense instead which will be
        // cleared by prepare()
        for (auto &wb : write_buffers) {
            write_task_queue.push(&wb);
        }
    }

    void prepare() {
        while (!write_task_queue.empty()) {
            write_task_queue.pop();
        }
        free_write_buffers.consume_all([](auto) {});
        for (auto &b : write_buffers) {
            free_write_buffers.push(&b);
        }
    }
};


template<typename T, unsigned Dims>
ndzip::mt_cpu_encoder<T, Dims>::mt_cpu_encoder() : _pimpl(std::make_unique<impl>(std::nullopt)) {
}

template<typename T, unsigned Dims>
ndzip::mt_cpu_encoder<T, Dims>::mt_cpu_encoder(size_t num_threads)
    : _pimpl(std::make_unique<impl>(num_threads)) {
}

template<typename T, unsigned Dims>
ndzip::mt_cpu_encoder<T, Dims>::~mt_cpu_encoder() = default;

template<typename T, unsigned Dims>
size_t ndzip::mt_cpu_encoder<T, Dims>::compress(
        const slice<const data_type, dimensions> &data, void *stream) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;
    using write_buffer = typename impl::write_buffer;

    constexpr auto side_length = profile::hypercube_side_length;
    constexpr auto num_hcs_per_chunk = impl::num_hcs_per_chunk;

    if (reinterpret_cast<uintptr_t>(stream) % sizeof(bits_type) != 0) {
        throw std::invalid_argument("stream is not properly aligned");
    }

    _pimpl->prepare();
    auto &write_task_queue = _pimpl->write_task_queue;
    auto &free_write_buffers = _pimpl->free_write_buffers;

    const detail::file<profile> file(data.size());
    const auto num_threads = _pimpl->num_threads;

    std::atomic<size_t> next_hc_index_to_read = 0;
    std::atomic<size_t> next_hc_index_to_write = 0;
    std::atomic<size_t> next_available_write_task_hc_index = SIZE_MAX;
    size_t stream_offset = file.file_header_length();

    const auto num_hypercubes = file.num_hypercubes();

#pragma omp parallel num_threads(num_threads)
#pragma omp single nowait
    for (size_t tid = 0; tid < num_threads; ++tid)
#pragma omp task firstprivate(tid)
    {
        auto &cube = _pimpl->thread_cubes[tid];

        // memory_order_relaxed: we only depend on next_hc_index_to_write for correctness and modify
        // it inside a critical section; outside, we can tolerate missed updates (the loop can
        // produce no-op iterations, the conditional around the critical section is just an
        // optimization to reduce contention)
        while (next_hc_index_to_write.load(std::memory_order_relaxed) < num_hypercubes) {
            write_buffer *write_task = nullptr;
            size_t task_stream_offset = 0;

            if (next_available_write_task_hc_index.load(std::memory_order_relaxed)
                    == next_hc_index_to_write.load(
                            std::memory_order_relaxed))  // reduce contention of critical section
#pragma omp critical(queue)
            {
                if (!write_task_queue.empty()
                        && write_task_queue.top()->first_hc_index
                                == next_hc_index_to_write.load(std::memory_order_relaxed)) {
                    write_task = write_task_queue.top();
                    write_task_queue.pop();
                    next_available_write_task_hc_index.store(write_task_queue.empty()
                                    ? SIZE_MAX
                                    : write_task_queue.top()->first_hc_index,
                            std::memory_order_relaxed);
                    next_hc_index_to_write.fetch_add(
                            write_task->num_hypercubes(), std::memory_order_relaxed);
                    task_stream_offset = stream_offset;
                    stream_offset += write_task->compressed_size();
                }
            }

            if (write_task) {
                memcpy(static_cast<std::byte *>(stream) + task_stream_offset,
                        write_task->stream.data(), write_task->compressed_size());
                auto task_file_offset = task_stream_offset;
                for (size_t task_hc_index = 0; task_hc_index < write_task->num_hypercubes();
                        ++task_hc_index) {
                    auto hc_index = write_task->first_hc_index + task_hc_index;
                    if (hc_index > 0) {
                        auto file_offset_address = static_cast<std::byte *>(stream)
                                + (hc_index - 1) * sizeof(detail::file_offset_type);
                        detail::store_aligned<detail::file_offset_type>(
                                file_offset_address, detail::endian_transform(task_file_offset));
                    }
                    task_file_offset
                            = task_stream_offset + write_task->offsets_after_hcs[task_hc_index];
                }
                free_write_buffers.push(write_task);
            } else  // compression task
            {
                if (free_write_buffers.pop(write_task)) {
                    // memory_order_relaxed: There is no synchronization involved, only atomicity is
                    // required. The ordering of the following operations is determined by the
                    // returned value.
                    auto first_hc_index = next_hc_index_to_read.fetch_add(
                            num_hcs_per_chunk, std::memory_order_relaxed);

                    if (first_hc_index < num_hypercubes) {
                        write_task->first_hc_index = first_hc_index;
                        write_task->offsets_after_hcs.clear();
                        for (size_t task_hc_index = 0;
                                task_hc_index + first_hc_index < num_hypercubes
                                && task_hc_index < num_hcs_per_chunk;
                                ++task_hc_index) {
                            auto hc_index = first_hc_index + task_hc_index;
                            auto hc_offset = detail::extent_from_linear_id(
                                                     hc_index, data.size() / side_length)
                                    * side_length;
                            detail::cpu::load_hypercube<profile>(hc_offset, data, cube.data());
                            detail::cpu::block_transform<profile>(cube.data());

                            task_stream_offset += detail::cpu::zero_bit_encode<profile>(
                                    cube.data(), write_task->stream.data() + task_stream_offset);
                            write_task->offsets_after_hcs.push_back(task_stream_offset);
                        }

#pragma omp critical(queue)
                        {
                            write_task_queue.push(write_task);
                            next_available_write_task_hc_index.store(
                                    write_task_queue.top()->first_hc_index,
                                    std::memory_order_relaxed);
                        }
                    } else {
                        free_write_buffers.push(write_task);
                    }
                }
            }
        }
    }

    if (file.num_hypercubes() > 0) {
        auto file_offset_address = static_cast<char *>(stream)
                + (file.num_hypercubes() - 1) * sizeof(detail::file_offset_type);
        detail::store_aligned(file_offset_address, detail::endian_transform(stream_offset));
    }
    stream_offset
            += detail::pack_border(static_cast<char *>(stream) + stream_offset, data, side_length);
    return stream_offset;
}


template<typename T, unsigned Dims>
size_t ndzip::mt_cpu_encoder<T, Dims>::decompress(
        const void *stream, size_t bytes, const slice<data_type, dimensions> &data) const {
    using profile = detail::profile<T, Dims>;
    using bits_type = typename profile::bits_type;

    constexpr static auto side_length = profile::hypercube_side_length;

    if (reinterpret_cast<uintptr_t>(stream) % sizeof(bits_type) != 0) {
        throw std::invalid_argument("stream is not properly aligned");
    }

    detail::file<profile> file(data.size());
    const auto num_hypercubes = file.num_hypercubes();
    const auto num_threads = _pimpl->num_threads;

#pragma omp parallel num_threads(num_threads)
    {
        auto tid = omp_get_thread_num();
        auto &cube = _pimpl->thread_cubes[tid];

#pragma omp for schedule(static) nowait
        for (size_t hc_index = 0; hc_index < num_hypercubes; ++hc_index) {
            auto hc_offset = detail::extent_from_linear_id(hc_index, data.size() / side_length)
                    * side_length;

            size_t stream_pos;
            if (hc_index > 0) {
                auto file_offset_address = static_cast<const std::byte *>(stream)
                        + (hc_index - 1) * sizeof(detail::file_offset_type);
                auto file_offset = detail::endian_transform(
                        detail::load_aligned<detail::file_offset_type>(file_offset_address));
                stream_pos = static_cast<size_t>(file_offset);
            } else {
                stream_pos = file.file_header_length();
            }

            detail::cpu::zero_bit_decode<profile>(
                    static_cast<const std::byte *>(stream) + stream_pos, cube.data());
            detail::cpu::inverse_block_transform<profile>(cube.data());
            detail::cpu::store_hypercube<profile>(hc_offset, cube.data(), data);
        }
    }

    size_t stream_pos;
    if (file.num_hypercubes() > 0) {
        auto file_offset_address = static_cast<const std::byte *>(stream)
                + (file.num_hypercubes() - 1) * sizeof(detail::file_offset_type);
        auto file_offset = detail::endian_transform(
                detail::load_aligned<detail::file_offset_type>(file_offset_address));
        stream_pos = static_cast<size_t>(file_offset);
    } else {
        stream_pos = file.file_header_length();
    }
    stream_pos += detail::unpack_border(
            data, static_cast<const std::byte *>(stream) + stream_pos, side_length);
    return stream_pos;
}

#endif  // NDZIP_OPENMP_SUPPORT


namespace ndzip {

extern template class cpu_encoder<float, 1>;
extern template class cpu_encoder<float, 2>;
extern template class cpu_encoder<float, 3>;
extern template class cpu_encoder<double, 1>;
extern template class cpu_encoder<double, 2>;
extern template class cpu_encoder<double, 3>;

#if NDZIP_OPENMP_SUPPORT
extern template class mt_cpu_encoder<float, 1>;
extern template class mt_cpu_encoder<float, 2>;
extern template class mt_cpu_encoder<float, 3>;
extern template class mt_cpu_encoder<double, 1>;
extern template class mt_cpu_encoder<double, 2>;
extern template class mt_cpu_encoder<double, 3>;
#endif  // NDZIP_OPENMP_SUPPORT

}  // namespace ndzip
