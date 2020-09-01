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
void ixt_step(T *x, size_t n, size_t s) {
    for (size_t i = 1; i < n; ++i) {
        x[i*s] ^= x[(i-1)*s];
    }
}

template<typename T>
[[gnu::noinline]]
void ixt3d_0(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m*l; i += m*l) {
        for (size_t j = 0; j < m; ++j) {
            ixt_step(x + i + j, l, n);
        }
    }
}

template<typename T>
[[gnu::noinline]]
void ixt3d_1(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m*l; i += n) {
        ixt_step(x + i, m, 1);
    }
}

template<typename T>
[[gnu::noinline]]
void ixt3d_2(T *x, size_t n, size_t m, size_t l) {
    for (size_t i = 0; i < n*m; ++i) {
        ixt_step(x + i, m, l*n);
    }
}


[[gnu::noinline]]
size_t readv(uint32_t *vs, const char *in) {
    auto head = in;
    auto body = in + 32;
    unsigned bit_pos = 0;
    for (unsigned i = 0; i < 64; i += 2) {
        unsigned length_codes[2] = {static_cast<unsigned>(static_cast<uint8_t>(*head)) >> 4, static_cast<unsigned>(static_cast<uint8_t>(*head)) & 0xf};
        for (unsigned j = 0; j < 2; ++j) {
            if (length_codes[j] != 0) {
                uint64_t v;
                __builtin_memcpy(&v, body, sizeof v);
                v = __builtin_bswap64(v);
                unsigned n_bits;
                if (length_codes[j] == 1) {
                    auto n_bits = 18;
                    auto shift = 64u - bit_pos - n_bits;
                    vs[i+j] = (v >> shift) & ~(~uint64_t{} << n_bits);
                    bit_pos += n_bits;
                } else {
                    auto n_bits = length_codes[j] + 16;
                    auto shift = 64u - bit_pos - n_bits;
                    vs[i+j] = ((v >> shift) & ~(~uint64_t{} << n_bits)) | (1u << n_bits);
                    bit_pos += n_bits;
                }
                body += bit_pos / 32 * 4;
                bit_pos %= 32;
            } else {
                vs[i+j] = 0;
            }
        }
        ++head;
    }
    return body-in+(bit_pos+7)/8;
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

    auto buf_in = mmap(NULL, stat_in.st_size, PROT_READ, MAP_PRIVATE, fd_in, 0);
    if (buf_in == MAP_FAILED) die("mmap");

    auto fd_out = open(fn_out, O_RDWR|O_CREAT, 0666);
    if (fd_out == -1) die("open");

    size_t out_size = n*m*l*4;
    if (ftruncate(fd_out, out_size) == -1) die("ftruncate");

    auto buf_out = static_cast<uint32_t*>(mmap(NULL, out_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd_out, 0));
    if (buf_out == MAP_FAILED) die("mmap");

    auto o = static_cast<const char*>(buf_in);
    for (size_t i = 0; i < n*m*l; i += 64) {
        o += readv(buf_out + i, o);
    }

    ixt3d_0(buf_out, n, m, l);
    ixt3d_1(buf_out, n, m, l);
    ixt3d_2(buf_out, n, m, l);

    if (munmap(buf_out, out_size) == -1) die("munmap");
    if (close(fd_out) == -1) die("close");
    if (munmap(buf_in, stat_in.st_size) == -1) die("munmap");
    if (close(fd_in) == -1) die("close");
}

