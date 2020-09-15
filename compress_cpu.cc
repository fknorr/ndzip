#include <assert.h>
#include <immintrin.h>
#include <limits.h>
#include <algorithm>
#include <type_traits>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NOINLINE [[gnu::noinline]]


template<typename T>
struct float_bits_traits;

template<>
struct float_bits_traits<float> {
    static_assert(sizeof(float) == 4);
    static_assert(std::numeric_limits<float>::is_iec559); // IEE 754

    using bits_type = uint32_t;

    constexpr static unsigned sign_bits = 1;
    constexpr static unsigned exponent_bits = 8;
    constexpr static unsigned mantissa_bits = 23;
};

template<>
struct float_bits_traits<double> {
    static_assert(sizeof(double) == 8);
    static_assert(std::numeric_limits<double>::is_iec559); // IEE 754

    using bits_type = uint64_t;

    constexpr static unsigned sign_bits = 1;
    constexpr static unsigned exponent_bits = 11;
    constexpr static unsigned mantissa_bits = 52;
};


template<typename T, typename Enable=void>
struct positional_bits_repr;

template<typename T>
struct positional_bits_repr<T, std::enable_if_t<std::is_floating_point_v<T>>> {
    constexpr static unsigned sign_bits = float_bits_traits<T>::sign_bits;
    constexpr static unsigned exponent_bits = float_bits_traits<T>::exponent_bits;
    constexpr static unsigned mantissa_bits = float_bits_traits<T>::mantissa_bits;

    using numeric_type = T;
    using bits_type = typename float_bits_traits<T>::bits_type;

    static bits_type to_bits(bits_type bits) {
        auto exponent = (bits << 1u) & ~(~bits_type{0} >> exponent_bits);
        auto sign = (bits & ~(~bits_type{0} >> sign_bits)) >> exponent_bits;
        auto mantissa = bits & ~(~bits_type{0} << mantissa_bits);
        return exponent | sign | mantissa;
    }

    static bits_type from_bits(bits_type bits) {
        auto sign = (bits << exponent_bits) & ~(~bits_type{0} >> sign_bits);
        auto exponent = (bits & ~(~bits_type{0} >> exponent_bits)) >> sign_bits;
        auto mantissa = bits & ~(~bits_type{0} << mantissa_bits);
        return sign | exponent | mantissa;
    }
};



inline uint32_t load(const void *addr) {
    uint32_t v;
    __builtin_memcpy(&v, addr, sizeof v);
    return v;
}

inline void store(void *addr, uint32_t v) {
    __builtin_memcpy(addr, &v, sizeof v);
}

inline uint32_t shl(uint32_t v, unsigned s) {
    return s < 32 ? v << s : 0;
}

inline uint32_t shr(uint32_t v, unsigned s) {
    return s < 32 ? v >> s : 0;
}

inline uint32_t shift(uint32_t v, int s) {
    return shl(v, s) | shr(v, -s);
}


template<typename T>
void xt_step(T *x, size_t n, size_t s) {
    T a, b;
    b = x[0*s];
    for (size_t i = 1; i < n; ++i) {
        a = b;
        b = x[i*s];
        x[i*s] = a ^ b;
    }
}

constexpr unsigned block = 8;

template<typename T>
void xt1d(T *x, size_t n) {
    xt_step(x, n, 1);
}

template<typename T>
NOINLINE
void xt2d_0(T *x, size_t n, size_t m) {
#pragma omp parallel for
    for (size_t i = 0; i < n/block*block; i += block) {
        T a[block], b[block];
        for (size_t j = 0; j < block; ++j) {
            b[j] = x[(i+j)*m];
        }
        for (size_t k = 1; k < m; ++k) {
            for (size_t j = 0; j < block; ++j) {
                a[j] = b[j];
                b[j] = x[(i+j)*m+k];
                x[(i+j)*m+k] = a[j] ^ b[j];
            }
        }
    }
    for (size_t i = n/block*block; i < n; ++i) {
        xt_step(x + i*m, m, 1);
    }
}

template<typename T>
NOINLINE
void xt2d_1(T *x, size_t n, size_t m) {
#pragma omp parallel for
    for (size_t i = 0; i < m/block*block; i += block) {
        T a[block], b[block];
        for (size_t j = 0; j < block; ++j) {
            b[j] = x[i+j];
        }
        for (size_t k = 1; k < n; ++k) {
            for (size_t j = 0; j < block; ++j) {
                a[j] = b[j];
                b[j] = x[i+j+k*m];
                x[i+j+k*m] = a[j] ^ b[j];
            }
        }
    }
    for (size_t i = m/block*block; i < m; ++i) {
        xt_step(x + i, n, m);
    }
}

template<typename T>
void xt2d(T *x, size_t n, size_t m) {
    xt2d_0(x, n, m);
    xt2d_1(x, n, m);
}

template<typename T>
NOINLINE
void xt3d_0(T *x, size_t n, size_t m, size_t l) {
#pragma omp parallel for
    for (size_t i = 0; i < n*m/block*block; i += block) {
        T a[block], b[block];
        for (size_t j = 0; j < block; ++j) {
            b[j] = x[(i+j)*l];
        }
        for (size_t k = 1; k < l; ++k) {
            for (size_t j = 0; j < block; ++j) {
                a[j] = b[j];
                b[j] = x[(i+j)*l+k];
                x[(i+j)*l+k] = a[j] ^ b[j];
            }
        }
    }
    for (size_t i = n*m/block*block; i < n*m; ++i) {
        xt_step(x + i*l, l, 1);
    }
}

template<typename T>
NOINLINE
void xt3d_1(T *x, size_t n, size_t m, size_t l) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < l/block*block; j += block) {
            T a[block], b[block];
            for (size_t k = 0; k < block; ++k) {
                b[k] = x[i*m*l+j+k];
            }
            for (size_t h = 1; h < m; ++h) {
                for (size_t k = 0; k < block; ++k) {
                    a[k] = b[k];
                    b[k] = x[i*m*l+j+k+h*l];
                    x[i*m*l+j+k+h*l] = a[k] ^ b[k];
                }
            }
        }
        for (size_t j = l/block*block; j < l; ++j) {
            xt_step(x + i*m*l + j, m, l);
        }
    }
}

template<typename T>
NOINLINE
void xt3d_2(T *x, size_t n, size_t m, size_t l) {
#pragma omp parallel for
    for (size_t i = 0; i < m*l/block*block; i += block) {
        T a[block], b[block];
        for (size_t j = 0; j < block; ++j) {
            b[j] = x[i+j];
        }
        for (size_t j = 0; j < block; ++j) {
            for (size_t k = 1; k < n; ++k) {
                a[j] = b[j];
                b[j] = x[i+j+k*m*l];
                x[i+j+k*m*l] = a[j] ^ b[j];
            }
        }
    }
    for (size_t i = m*l/block*block; i < m*l; ++i) {
        xt_step(x + i, n, m*l);
    }
}


template<typename T>
void xt3d(T *x, size_t n, size_t m, size_t l) {
    xt3d_0(x, n, m, l);
    xt3d_1(x, n, m, l);
    xt3d_2(x, n, m, l);
}

NOINLINE
size_t writev(uint32_t *vs, char *out0) {
    auto out = out0;
    for (unsigned i = 0; i < 256; i += 2) {
        unsigned length_codes[2];
        for (unsigned j = 0; j < 2; ++j) {
            if (vs[i+j]) {
                auto lz = __builtin_clz(vs[i+j]);
                auto code = lz >= 14 ? 1 : 15 - lz;
                length_codes[j] = code;
            } else {
                length_codes[j] = 0;
            }
        }
        *out++ = (length_codes[0] << 4) | length_codes[1];
    }

    store(out, 0);
    unsigned bit_pos = 0;

    for (unsigned i = 0; i < 256; i += 1) {
        if (vs[i]) {
            auto lz = __builtin_clz(vs[i]) ;
            auto n_bits = lz >= 14 ? 18 : 31 - lz;

            uint32_t u[2] = {};
            __builtin_memcpy(u, out, 4);
            uint64_t v;
            __builtin_memcpy(&v, &u, sizeof v);
            auto bits = vs[i] & ~(~uint64_t{} << n_bits);
            v |= __builtin_bswap64(uint64_t(bits) << (64u - bit_pos - n_bits));
            __builtin_memcpy(out, &v, 8);

            bit_pos += n_bits;
            out += bit_pos / 32 * 4;
            bit_pos %= 32;
        }
    }

    return out-out0+(bit_pos+7)/8;
}


void transpose32_trivial(uint32_t *__restrict vs, uint32_t *__restrict out) {
    for (unsigned i = 0; i < 32; ++i) {
        out[i] = 0;
        for (unsigned j = 0; j < 32; ++j) {
            out[i] |= ((vs[j] >> (31-i)) & 1) << (31-j);
        }
    }
}


void transpose32_avx(uint32_t *__restrict vs, uint32_t *__restrict out) {
    __m256 in[4];
    __builtin_memcpy(&in, vs, sizeof in);
    uint8_t *out_bytes = reinterpret_cast<uint8_t*>(out);
    for (unsigned i = 0; i < 32; ++i) {
        for (unsigned j = 0; j < 4; ++j) {
            uint8_t mask = _mm256_movemask_ps(in[j]);
            out_bytes[4*i + 3-j] = mask;
        }
        for (unsigned j = 0; j < 4; ++j) {
            auto v2 = _mm256_castps_si256(in[j]);
            v2 = _mm256_slli_epi32(v2, 1);
            in[j] = _mm256_castsi256_ps(v2);
        }
    }
}

[[gnu::always_inline]]
inline void transpose32_avx2_bytes(uint32_t *__restrict vs, uint32_t *__restrict out) {
    __m256i unpck0[4];
    __builtin_memcpy(unpck0, vs, sizeof unpck0);

    // 1. In order to generate 32 transpositions using only 32 *movemask* instructions,
    // first shuffle bytes so that the i-th 256-bit vector contains the i-th byte of
    // each float

    __m256i shuf[4];
    {
        // interpret each 128-bit lane as a 4x4 matrix and transpose it
        // also correct endianess for later
        uint8_t idx8[] = {
            3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12,
            3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12,
            // 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
            // 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
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
        uint32_t idx32[] = {0, 4, 1, 5, 2, 6, 3, 7};
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
        unpck1[i+1] = _mm256_unpackhi_epi64(perm[i+0], perm[i+1]);
        unpck1[i+0] = _mm256_unpacklo_epi64(perm[i+0], perm[i+1]);
    }

    __m256i perm2[4] = {
        // combine matching 128-bit lanes
        _mm256_permute2x128_si256(unpck1[0], unpck1[2], 0x20),
        _mm256_permute2x128_si256(unpck1[1], unpck1[3], 0x20),
        _mm256_permute2x128_si256(unpck1[0], unpck1[2], 0x31),
        _mm256_permute2x128_si256(unpck1[1], unpck1[3], 0x31),
    };

    // 2. Transpose by extracting the 32 MSBs of each byte of each 256-byte vector

    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = 0; j < 8; ++j) {
            out[i*8+j] = _mm256_movemask_epi8(perm2[i]);
            perm2[i] = _mm256_slli_epi32(perm2[i], 1);
        }
    }
}


[[gnu::always_inline]]
inline __m256i m256i_literal_8(std::initializer_list<uint8_t> bytes) {
    __m256i v;
    __builtin_memcpy(&v, bytes.begin(), sizeof v);
    return v;
}

[[gnu::always_inline]]
inline __m256i m256i_literal_32(std::initializer_list<uint32_t> dwords) {
    __m256i v;
    __builtin_memcpy(&v, dwords.begin(), sizeof v);
    return v;
}

[[gnu::always_inline]]
inline unsigned compact32_avx2(const uint32_t *__restrict transposed, uint32_t *__restrict out) {
    __m256i in[4];
    __builtin_memcpy(in, transposed, sizeof in);

    const __m256i zero = {0};
    __m256i eq[4];
    for (unsigned i = 0; i < 4; ++i) {
        eq[i] = _mm256_cmpeq_epi32(in[i], zero);
    }

    __m256i neq_masked[4];
    auto squash_mask = m256i_literal_8({
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
            });
    for (unsigned i = 0; i < 4; ++i) {
        neq_masked[i] = _mm256_andnot_si256(eq[i], squash_mask);
    }

    __m256i neq_squash0[4];
    for (unsigned i = 0; i < 4; ++i) {
        neq_squash0[i] = _mm256_hadd_epi32(_mm256_hadd_epi32(neq_masked[i], zero), zero);
    }

    __m256i neq_squash1 = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_permutevar8x32_epi32(neq_squash0[0], m256i_literal_32({0, 4, 1, 1, 1, 1, 1, 1})),
                _mm256_permutevar8x32_epi32(neq_squash0[1], m256i_literal_32({1, 1, 0, 4, 1, 1, 1, 1}))
                ),
            _mm256_or_si256(
                _mm256_permutevar8x32_epi32(neq_squash0[2], m256i_literal_32({1, 1, 1, 1, 0, 4, 1, 1})),
                _mm256_permutevar8x32_epi32(neq_squash0[3], m256i_literal_32({1, 1, 1, 1, 1, 1, 0, 4}))
                ));

    auto mask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_sub_epi8(zero, neq_squash1)));

    __m256i lanes = neq_squash1;
    lanes = _mm256_add_epi8(lanes, _mm256_slli_si256(lanes, 1));
    lanes = _mm256_add_epi8(lanes, _mm256_slli_si256(lanes, 2));
    lanes = _mm256_add_epi8(lanes, _mm256_slli_si256(lanes, 4));
    lanes = _mm256_add_epi8(lanes, _mm256_slli_si256(lanes, 8));

    __m128i low = _mm_broadcastb_epi8(_mm_srli_si128(_mm256_extractf128_si256(lanes, 0), 15));
    __m256i prefix_sum = _mm256_add_epi8(lanes, _mm256_insertf128_si256(zero, low, 1));

    unsigned total = _mm256_extract_epi8(prefix_sum, 31);

    uint8_t offsets[32];
    __builtin_memcpy(offsets, &prefix_sum, 32);

    for (unsigned i = 0; i < 32; ++i) {
        out[offsets[31-i]] = transposed[31-i];
    }
    out[0] = mask;

    return 1+total;
}


[[gnu::always_inline]]
inline unsigned compact32_trivial(const uint32_t *shifted, uint32_t *out) {
    unsigned nonzero = 0;
    auto head = out++;
    *head = 0;
    for (unsigned i = 0; i < 32; ++i) {
        if (shifted[i] != 0) {
            ++nonzero;
            *head |= 1u << i;
            *out++ = shifted[i];
        }
    }
    return 1 + nonzero;
}


[[gnu::noinline]]
size_t writev_shuffle(uint32_t *vs, char *const out0) {
    uint32_t *out = (uint32_t*) out0;
    for (unsigned i = 0; i < 8; ++i) {
        uint32_t shifted[32];
        transpose32_avx2_bytes(vs, shifted);
        // out += compact32_avx2(shifted, out);
        // transpose32_trivial(vs, shifted);
        out += compact32_trivial(shifted, out);
        vs += 32;
    }
    return (char*)out - out0;
}


void die(const char *what) {
    perror(what);
    exit(1);
}


int main(int argc, char **argv) {
    auto fn_in = argv[1];
    auto fn_out = argv[2];
    auto n = (unsigned)atoi(argv[3]);
    size_t m = argc > 4 ? (unsigned)atoi(argv[4]) : 1;
    size_t l = argc > 5 ? (unsigned)atoi(argv[5]) : 1;
    unsigned dims = argc - 3;

    auto fd_in = open(fn_in, O_RDONLY);
    if (fd_in == -1) die("open");

    struct stat stat_in;
    if (fstat(fd_in, &stat_in) == -1) die("stat");
    auto in_size = static_cast<size_t>(stat_in.st_size);

    auto buf_in = static_cast<uint32_t*>(mmap(NULL, in_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_in, 0));
    if (buf_in == MAP_FAILED) die("mmap");

    for (size_t i = 0; i < n*m*l; ++i) {
        buf_in[i] = positional_bits_repr<float>::to_bits(buf_in[i]);
    }
    if (dims == 1) {
        xt1d(buf_in, n);
    } else if (dims == 2) {
        xt2d(buf_in, n, m);
    } else {
        xt3d(buf_in, n, m, l);
    }

    auto fd_out = open(fn_out, O_RDWR|O_CREAT, 0666);
    if (fd_out == -1) die("open");

    size_t out_maxsize = n*m*l*5;
    if (ftruncate(fd_out, out_maxsize) == -1) die("ftruncate");

    auto buf_out = mmap(NULL, out_maxsize, PROT_READ|PROT_WRITE, MAP_SHARED, fd_out, 0);
    if (buf_out == MAP_FAILED) die("mmap");

    auto o = static_cast<char*>(buf_out);
    for (size_t i = 0; i < n*m*l; i += 256) {
        o += writev_shuffle(buf_in + i, o);
    }

    auto out_size = o-static_cast<char*>(buf_out);

    if (munmap(buf_out, out_maxsize) == -1) die("munmap");
    if (ftruncate(fd_out, out_size) == -1) die("ftruncate");
    if (close(fd_out) == -1) die("close");
    if (munmap(buf_in, in_size) == -1) die("munmap");
    if (close(fd_in) == -1) die("close");

    printf("%zu => %zu (ratio %.4g)\n", in_size, out_size, double(in_size)/out_size);
}

