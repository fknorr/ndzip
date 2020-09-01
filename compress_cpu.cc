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
[[gnu::noinline]]
void xt3d_0(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m*l; i += m*l) {
        for (size_t j = 0; j < m; ++j) {
            xt_step(x + i + j, l, n);
        }
    }
}

template<typename T>
[[gnu::noinline]]
void xt3d_1(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m*l; i += n) {
        xt_step(x + i, m, 1);
    }
}

template<typename T>
[[gnu::noinline]]
void xt3d_2(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m; ++i) {
        xt_step(x + i, m, l*n);
    }
}


[[gnu::noinline]]
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
    auto m = (unsigned)atoi(argv[4]);
    auto l = (unsigned)atoi(argv[5]);

    auto fd_in = open(fn_in, O_RDONLY);
    if (fd_in == -1) die("open");

    struct stat stat_in;
    if (fstat(fd_in, &stat_in) == -1) die("stat");

    auto buf_in = static_cast<uint32_t*>(malloc(stat_in.st_size+64*4));
    if (read(fd_in, buf_in, stat_in.st_size) != stat_in.st_size) die("read");
    if (close(fd_in) == -1) die("close");

    xt3d_0(buf_in, n, m, l);
    xt3d_1(buf_in, n, m, l);
    xt3d_2(buf_in, n, m, l);

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

    if (munmap(buf_out, out_maxsize) == -1) die("munmap");
    if (ftruncate(fd_out, o-static_cast<char*>(buf_out)) == -1) die("ftruncate");
    if (close(fd_out) == -1) die("close");
}

