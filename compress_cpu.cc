#include <assert.h>
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

#define NOINLINE


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
    constexpr static unsigned exponent_bits = float_bits_traits<T>::exponent_bits - 3;
    constexpr static unsigned mantissa_bits = float_bits_traits<T>::mantissa_bits + 3;

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

template<typename T>
void xt1d(T *x, size_t n) {
    xt_step(x, n, 1);
}

template<typename T>
NOINLINE
void xt2d_0(T *x, size_t n, size_t m) {
    for (size_t i = 0; i < n*m; i += m) {
        xt_step(x + i, m, 1);
    }
}

template<typename T>
NOINLINE
void xt2d_1(T *x, size_t n, size_t m) {
    for (size_t i = 0; i < m; ++i) {
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
    constexpr size_t block = 16;
    for (size_t i = 0; i < n*m*l; i += m*l) {
        for (size_t j = 0; j < m/block*block; j += block) {
            T a[block], b[block];
            for (size_t h = 0; h < block; ++h) {
                b[h] = x[i+j+h];
            }
            for (size_t k = 1; k < l; ++k) {
                for (size_t h = 0; h < block; ++h) {
                    a[h] = b[h];
                    b[h] = x[i+j+h + k*n];
                    x[i+j+h + k*n] = a[h] ^ b[h];
                }
            }
        }
        for (size_t j = m/block*block; j < m; ++j) {
            xt_step(x + i + j, l, n);
        }
    }
}

template<typename T>
NOINLINE
void xt3d_1(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m*l; i += n) {
        xt_step(x + i, m, 1);
    }
}

template<typename T>
NOINLINE
void xt3d_2(T *x, size_t n, size_t m, size_t l) {
    auto s = n*m;
    constexpr size_t block = 8;
    for (size_t i = 0; i < n*m/block*block; i += block) {
        T a[block], b[block];
        for (size_t j = 0; j < block; ++j) {
            b[j] = x[i+j];
        }
        for (size_t k = 1; k < l; ++k) {
            for (size_t j = 0; j < block; ++j) {
                a[j] = b[j];
                b[j] = x[i+j + k*s];
                x[i+j + k*s] = a[j] ^ b[j];
            }
        }
    }
    for (size_t j = n*m/block*block; j < n*m; ++j) {
        xt_step(x+j, l, s);
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
    for (unsigned i = 0; i < 64; i += 2) {
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

    for (unsigned i = 0; i < 64; i += 1) {
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
        xt3d(buf_in, l, m, n);
    }

    auto fd_out = open(fn_out, O_RDWR|O_CREAT, 0666);
    if (fd_out == -1) die("open");

    size_t out_maxsize = n*m*l*5;
    if (ftruncate(fd_out, out_maxsize) == -1) die("ftruncate");

    auto buf_out = mmap(NULL, out_maxsize, PROT_READ|PROT_WRITE, MAP_SHARED, fd_out, 0);
    if (buf_out == MAP_FAILED) die("mmap");

    auto o = static_cast<char*>(buf_out);
    for (size_t i = 0; i < n*m*l; i += 64) {
        o += writev(buf_in + i, o);
    }

    auto out_size = o-static_cast<char*>(buf_out);

    if (munmap(buf_out, out_maxsize) == -1) die("munmap");
    if (ftruncate(fd_out, out_size) == -1) die("ftruncate");
    if (close(fd_out) == -1) die("close");
    if (munmap(buf_in, in_size) == -1) die("munmap");
    if (close(fd_in) == -1) die("close");

    printf("%zu => %zu (ratio %.4g)\n", in_size, out_size, double(in_size)/out_size);
}

