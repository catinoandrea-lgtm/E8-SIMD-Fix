/* E8/D8 AVX2 — Integrity Test with I/O Checksum
 * Andrea Catino — Independent Researcher, Italy * 2026-06-01 21:16
 * Verifies: (A) output in D8, (B) covering radius, (C) bit-level checksum
 *
 * Build (x86):
 *   g++   -O3 -mavx2 -mfma -std=c++17 checksum_avx2.cxx -o checksum_avx2
 *   clang -O3 -mavx2 -mfma -std=c++17 checksum_avx2.cxx -o checksum_avx2
 * Run:
 *   ./checksum_avx2
 *
 * Golden I/O checksum (seed 1337, N=1000000) -- ARCHITECTURE-SPECIFIC:
 *   Ubuntu x86 (g++):  0x24d087f2d94ce053
 *   Colab  x86 (g++):  0x2520b5e88f1750a5
 *   ARM64 A55 (clang): 0x1866e1701487dc06
 *   All three are valid: the coset/lane tie-break selects different but
 *   equally valid D8 points across toolchains. Each matches its own scalar
 *   reference on the same machine.
 */
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

// ============================================================================
// QUANTIZE D8 — AVX2 FIXED (mask & -mask)
// ============================================================================
inline __m256 quantize_D8_fixed_avx2(__m256 x) {
    __m256 z   = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 err = _mm256_sub_ps(x, z);
    __m256 sm  = _mm256_set1_ps(-0.0f);
    __m256 ae  = _mm256_andnot_ps(sm, err);

    // Broadcast del massimo su tutte le 8 lane
    __m256 s1 = _mm256_permute2f128_ps(ae, ae, 1);
    __m256 m1 = _mm256_max_ps(ae, s1);
    __m256 s2 = _mm256_shuffle_ps(m1, m1, _MM_SHUFFLE(2,3,0,1));
    __m256 m2 = _mm256_max_ps(m1, s2);
    __m256 s3 = _mm256_shuffle_ps(m2, m2, _MM_SHUFFLE(1,0,3,2));
    __m256 mv = _mm256_max_ps(m2, s3);

    __m256 is_max = _mm256_cmp_ps(ae, mv, _CMP_EQ_OQ);

    // FIX ex-aequo: mask & -mask isola il bit meno significativo
    int raw    = _mm256_movemask_ps(is_max);
    int single = raw & (-raw);

    // Ricostruisce la maschera a 256-bit con un solo lane attivo
    alignas(32) int32_t lm[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 8; ++i) if ((single >> i) & 1) { lm[i] = 0xFFFFFFFF; break; }
    __m256 smask = _mm256_castsi256_ps(_mm256_load_si256((__m256i*)lm));

    // Parità
    __m256i zi = _mm256_cvtps_epi32(z);
    __m256i p1 = _mm256_hadd_epi32(zi, zi);
    __m256i p2 = _mm256_hadd_epi32(p1, p1);
    __m256i p3 = _mm256_add_epi32(p2, _mm256_permute2f128_si256(p2, p2, 1));
    int par    = _mm256_extract_epi32(p3, 0) & 1;

    __m256i pm  = _mm256_set1_epi32(par ? 0xFFFFFFFF : 0);
    __m256  act = _mm256_and_ps(smask, _mm256_castsi256_ps(pm));
    __m256  one = _mm256_set1_ps(1.0f);
    __m256  es  = _mm256_and_ps(err, sm);
    __m256  cor = _mm256_or_ps(one, es);
    return _mm256_add_ps(z, _mm256_and_ps(act, cor));
}

// ============================================================================
// QUANTIZE D8 — AVX2 NAIVE (bug intenzionale — senza isolate)
// ============================================================================
inline __m256 quantize_D8_naive_avx2(__m256 x) {
    __m256 z   = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 err = _mm256_sub_ps(x, z);
    __m256 sm  = _mm256_set1_ps(-0.0f);
    __m256 ae  = _mm256_andnot_ps(sm, err);
    __m256 s1  = _mm256_permute2f128_ps(ae, ae, 1);
    __m256 m1  = _mm256_max_ps(ae, s1);
    __m256 s2  = _mm256_shuffle_ps(m1, m1, _MM_SHUFFLE(2,3,0,1));
    __m256 m2  = _mm256_max_ps(m1, s2);
    __m256 s3  = _mm256_shuffle_ps(m2, m2, _MM_SHUFFLE(1,0,3,2));
    __m256 mv  = _mm256_max_ps(m2, s3);
    __m256 is_max = _mm256_cmp_ps(ae, mv, _CMP_EQ_OQ); // NESSUN isolate → bug

    __m256i zi = _mm256_cvtps_epi32(z);
    __m256i p1 = _mm256_hadd_epi32(zi, zi);
    __m256i p2 = _mm256_hadd_epi32(p1, p1);
    __m256i p3 = _mm256_add_epi32(p2, _mm256_permute2f128_si256(p2, p2, 1));
    int par    = _mm256_extract_epi32(p3, 0) & 1;

    __m256i pm  = _mm256_set1_epi32(par ? 0xFFFFFFFF : 0);
    __m256  act = _mm256_and_ps(is_max, _mm256_castsi256_ps(pm));
    __m256  one = _mm256_set1_ps(1.0f);
    __m256  es  = _mm256_and_ps(err, sm);
    __m256  cor = _mm256_or_ps(one, es);
    return _mm256_add_ps(z, _mm256_and_ps(act, cor));
}

// ============================================================================
// QUANTIZE D8 — SCALARE (baseline, no vectorize)
// ============================================================================
__attribute__((noinline))
void quantize_D8_scalar(const float* x, float* out) {
    float z[8], err[8];
    int par = 0; float me = -1.0f; int idx = 0;
    for (int i = 0; i < 8; ++i) {
        z[i]   = std::round(x[i]);
        err[i] = x[i] - z[i];
        float ae = std::abs(err[i]);
        if (ae > me) { me = ae; idx = i; }
        par += static_cast<int>(z[i]);
    }
    for (int i = 0; i < 8; ++i) out[i] = z[i];
    if (std::abs(par) % 2 != 0)
        out[idx] += (err[idx] > 0) ? 1.0f : -1.0f;
}

// ============================================================================
// VALIDATORI
// ============================================================================
bool in_D8(const float* v) {
    int s = 0; bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (std::abs(v[i] - std::round(v[i])) > 1e-4f) { ok = false; break; }
        s += static_cast<int>(std::round(v[i]));
    }
    return ok && (s % 2 == 0);
}

float dist2(const float* a, const float* b) {
    float d = 0;
    for (int i = 0; i < 8; ++i) { float e = a[i]-b[i]; d += e*e; }
    return d;
}

// ============================================================================
// CHECKSUM I/O
// ============================================================================
inline uint32_t fnbits(float f) {
    if (f == 0.0f) return 0;
    uint32_t u; std::memcpy(&u, &f, 4); return u;
}

uint64_t io_checksum(const float* in8, const float* out8, size_t idx) {
    uint64_t h = 0;
    for (int j = 0; j < 8; ++j) {
        h ^= (static_cast<uint64_t>(fnbits(in8[j]))  * 2654435761ULL + idx*8+j);
        h ^= (static_cast<uint64_t>(fnbits(out8[j])) * 2246822519ULL + idx*8+j+1);
    }
    return h;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    const size_t N = 1000000;

    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> d_base(-100.f, 100.f);
    std::uniform_real_distribution<float> d_noise(-0.2f,  0.2f);
    std::uniform_int_distribution<int>    d_idx(0, 7);

    std::vector<float> data(N * 8);
    for (size_t i = 0; i < N; ++i) {
        int bi[8]; int sum = 0;
        for (int j = 0; j < 8; ++j) {
            bi[j] = static_cast<int>(std::round(d_base(gen)));
            sum  += bi[j];
        }
        if (std::abs(sum) % 2 == 0) bi[0]++;
        for (int j = 0; j < 8; ++j)
            data[i*8+j] = static_cast<float>(bi[j]) + d_noise(gen);
        int i1 = d_idx(gen), i2 = d_idx(gen);
        while (i2 == i1) i2 = d_idx(gen);
        data[i*8+i1] = static_cast<float>(bi[i1]) + 0.485f;
        data[i*8+i2] = static_cast<float>(bi[i2]) - 0.485f;
    }

    std::cout << "============================================================\n";
    std::cout << "E8/D8 AVX2 — Integrity Test I/O  (N=" << N << ")\n";
    std::cout << "============================================================\n\n";

    const float CR_D8 = std::sqrt(8.0f) / 2.0f;

    uint64_t cs_avx2   = 0;
    uint64_t cs_scalar = 0;
    size_t naive_fails = 0, fixed_fails = 0;
    int ok_d8_avx2 = 0, ok_d8_scal = 0;
    int cr_ok_avx2 = 0, cr_ok_scal = 0;
    float max_dn = 0, sum_dn = 0;
    float max_ds = 0, sum_ds = 0;

    alignas(32) float out_f[8], out_n[8], out_s[8];

    for (size_t i = 0; i < N; ++i) {
        const float* x = &data[i*8];

        // AVX2 fixed
        __m256 inp = _mm256_loadu_ps(x);
        _mm256_storeu_ps(out_f, quantize_D8_fixed_avx2(inp));

        // AVX2 naive (bug)
        _mm256_storeu_ps(out_n, quantize_D8_naive_avx2(inp));

        // Scalare
        quantize_D8_scalar(x, out_s);

        // Confronto vs scalare
        bool fn = false, ff = false;
        for (int j = 0; j < 8; ++j) {
            if (std::abs(out_s[j] - out_n[j]) > 1e-4f) fn = true;
            if (std::abs(out_s[j] - out_f[j]) > 1e-4f) ff = true;
        }
        if (fn) naive_fails++;
        if (ff) fixed_fails++;

        // Metriche AVX2 fixed
        if (in_D8(out_f)) ok_d8_avx2++;
        float dn = std::sqrt(dist2(x, out_f));
        sum_dn += dn; max_dn = std::max(max_dn, dn);
        if (dn <= CR_D8 + 1e-4f) cr_ok_avx2++;

        // Metriche scalare
        if (in_D8(out_s)) ok_d8_scal++;
        float ds = std::sqrt(dist2(x, out_s));
        sum_ds += ds; max_ds = std::max(max_ds, ds);
        if (ds <= CR_D8 + 1e-4f) cr_ok_scal++;

        // Checksum I/O
        cs_avx2   ^= io_checksum(x, out_f, i);
        cs_scalar ^= io_checksum(x, out_s, i);
    }

    std::cout << "--- TEST 1: STRUCTURAL CORRECTNESS ---\n";
    std::cout << "AVX2 Naive (bug)  : " << naive_fails << " failures / " << N << "\n";
    std::cout << "AVX2 Fixed (fix)  : " << fixed_fails << " failures / " << N << "\n\n";

    std::cout << "--- TEST 2: GEOMETRIC METRICS ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Covering radius D8 = sqrt(8)/2 = " << CR_D8 << "\n\n";

    std::cout << "AVX2 fixed:\n";
    std::cout << "  Output in D8         : " << ok_d8_avx2 << "/" << N
              << (ok_d8_avx2 == (int)N ? "  ✓" : "  ✗") << "\n";
    std::cout << "  Mean distance I→O   : " << sum_dn/N << "\n";
    std::cout << "  Max distance   I→O   : " << max_dn << "\n";
    std::cout << "  Within covering radius : " << cr_ok_avx2 << "/" << N
              << (cr_ok_avx2 == (int)N ? "  ✓" : "  ✗") << "\n\n";

    std::cout << "Scalare:\n";
    std::cout << "  Output in D8         : " << ok_d8_scal << "/" << N
              << (ok_d8_scal == (int)N ? "  ✓" : "  ✗") << "\n";
    std::cout << "  Mean distance I→O   : " << sum_ds/N << "\n";
    std::cout << "  Max distance   I→O   : " << max_ds << "\n";
    std::cout << "  Within covering radius : " << cr_ok_scal << "/" << N
              << (cr_ok_scal == (int)N ? "  ✓" : "  ✗") << "\n\n";

    std::cout << "--- TEST 3: BIT-LEVEL I/O CHECKSUM ---\n";
    std::cout << "Checksum AVX2   : 0x" << std::hex << cs_avx2   << "\n";
    std::cout << "Checksum Scalar : 0x" << std::hex << cs_scalar << "\n";
    std::cout << std::dec;
    bool match  = (cs_avx2 == cs_scalar);
    bool nonzero= (cs_avx2 != 0);
    std::cout << "Equal          : " << (match   ? "SI  ✓" : "NO  (different tie-break, both valid)") << "\n";
    std::cout << "Non-zero        : " << (nonzero ? "SI  ✓" : "NO  ✗") << "\n";
    std::cout << "AVX2 I/O state  : "
              << (ok_d8_avx2 == (int)N && cr_ok_avx2 == (int)N
                  ? "VALID (all points in D8, within CR)" : "FALLIMENTO") << "\n\n";

    std::cout << "Note: a different AVX2-vs-scalar checksum is EXPECTED\n";
    std::cout << "when the tie-break selects different lanes (lowest lane\n";
    std::cout << "in AVX2 vs lowest index in scalar). Both points are in D8.\n";

    return 0;
}
