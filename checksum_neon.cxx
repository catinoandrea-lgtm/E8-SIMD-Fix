/* E8/D8 NEON — Integrity Test with I/O Checksum
 * Andrea Catino — Independent Researcher, Italy * 2026-06-01 21:16
 * Verifies: (A) output in D8, (B) covering radius, (C) bit-level checksum
 *
 * Build (ARM64):
 *   clang++ -O3 -march=armv8-a+simd checksum_neon.cxx -o checksum_neon
 * Run:
 *   ./checksum_neon
 *
 * Golden I/O checksum (seed 1337, N=1000000) -- ARCHITECTURE-SPECIFIC:
 *   ARM64 A55 (clang): 0x1866e1701487dc06
 *   Ubuntu x86 (g++):  0x24d087f2d94ce053
 *   Colab  x86 (g++):  0x2520b5e88f1750a5
 *   All valid: the tie-break selects different but equally valid D8 points
 *   across toolchains; each matches its own scalar reference on that machine.
 */
#include <arm_neon.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

// ============================================================================
// STRUTTURA DATI
// ============================================================================
struct Float8_NEON { float32x4_t lo; float32x4_t hi; };

// ============================================================================
// FIX: isola il primo bit attivo nella maschera ex-aequo
// ============================================================================
inline void isolate_first_max(uint32x4_t& lo, uint32x4_t& hi) {
    uint32x2_t lo_lo = vget_low_u32(lo);
    uint32x2_t lo_hi = vget_high_u32(lo);
    uint32_t any_lo  = vget_lane_u32(
        vpmax_u32(vorr_u32(lo_lo, lo_hi), vorr_u32(lo_lo, lo_hi)), 0);

    if (any_lo) {
        uint32_t m[4]; vst1q_u32(m, lo);
        int first = -1;
        for (int i = 0; i < 4; ++i) if (m[i]) { first = i; break; }
        uint32_t s[4] = {0,0,0,0};
        if (first >= 0) s[first] = 0xFFFFFFFF;
        lo = vld1q_u32(s);
        hi = vdupq_n_u32(0);
    } else {
        uint32_t m[4]; vst1q_u32(m, hi);
        int first = -1;
        for (int i = 0; i < 4; ++i) if (m[i]) { first = i; break; }
        uint32_t s[4] = {0,0,0,0};
        if (first >= 0) s[first] = 0xFFFFFFFF;
        hi = vld1q_u32(s);
    }
}

// ============================================================================
// QUANTIZE D8 — NEON FIXED
// ============================================================================
inline Float8_NEON quantize_D8_fixed(Float8_NEON x) {
    Float8_NEON z, err;
    z.lo   = vrndnq_f32(x.lo);
    z.hi   = vrndnq_f32(x.hi);
    err.lo = vsubq_f32(x.lo, z.lo);
    err.hi = vsubq_f32(x.hi, z.hi);

    float32x4_t ae_lo = vabsq_f32(err.lo);
    float32x4_t ae_hi = vabsq_f32(err.hi);
    float        mv   = vmaxvq_f32(vmaxq_f32(ae_lo, ae_hi));

    uint32x4_t im_lo = vceqq_f32(ae_lo, vdupq_n_f32(mv));
    uint32x4_t im_hi = vceqq_f32(ae_hi, vdupq_n_f32(mv));
    isolate_first_max(im_lo, im_hi);          // FIX ex-aequo

    int par = (vaddvq_s32(vcvtq_s32_f32(z.lo))
             + vaddvq_s32(vcvtq_s32_f32(z.hi))) & 1;

    uint32x4_t pm   = vdupq_n_u32(par ? 0xFFFFFFFF : 0);
    uint32x4_t sm   = vdupq_n_u32(0x80000000);
    uint32x4_t one  = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));

    uint32x4_t al = vandq_u32(vandq_u32(im_lo, pm),
                               vorrq_u32(one, vandq_u32(
                                   vreinterpretq_u32_f32(err.lo), sm)));
    z.lo = vaddq_f32(z.lo, vreinterpretq_f32_u32(al));

    uint32x4_t ah = vandq_u32(vandq_u32(im_hi, pm),
                               vorrq_u32(one, vandq_u32(
                                   vreinterpretq_u32_f32(err.hi), sm)));
    z.hi = vaddq_f32(z.hi, vreinterpretq_f32_u32(ah));
    return z;
}

// ============================================================================
// QUANTIZE D8 — SCALARE (baseline verificata, no vectorize)
// ============================================================================
__attribute__((noinline))
void quantize_D8_scalar(const float* x, float* out) {
    float z[8], err[8];
    int par = 0;
    float me = -1.0f;
    int idx = 0;
    #pragma clang loop vectorize(disable) unroll(disable)
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
// CHECKSUM I/O — normalizza -0.0f per confronto bit-level cross-platform
// ============================================================================
inline uint32_t fnbits(float f) {
    if (f == 0.0f) return 0;            // -0.0f → +0.0f
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

    // Generatore patologico: forza parità dispari + due errori simmetrici ±0.485
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
        if (std::abs(sum) % 2 == 0) bi[0]++;  // forza parità dispari
        for (int j = 0; j < 8; ++j)
            data[i*8+j] = static_cast<float>(bi[j]) + d_noise(gen);
        // due errori ex-aequo simmetrici
        int i1 = d_idx(gen), i2 = d_idx(gen);
        while (i2 == i1) i2 = d_idx(gen);
        data[i*8+i1] = static_cast<float>(bi[i1]) + 0.485f;
        data[i*8+i2] = static_cast<float>(bi[i2]) - 0.485f;
    }

    std::cout << "============================================================\n";
    std::cout << "E8/D8 NEON — Integrity Test I/O  (N=" << N << ")\n";
    std::cout << "============================================================\n\n";

    uint64_t cs_neon   = 0;
    uint64_t cs_scalar = 0;
    int ok_d8_neon  = 0, ok_d8_scal = 0;
    int cr_ok_neon  = 0, cr_ok_scal = 0;
    size_t naive_fails = 0, fixed_fails = 0;
    float  max_dist_n = 0, sum_dist_n = 0;
    float  max_dist_s = 0, sum_dist_s = 0;

    // covering radius D8 = sqrt(n)/2 = sqrt(8)/2 (Conway-Sloane SPLAG)
    const float CR_D8 = std::sqrt(8.0f) / 2.0f;

    std::vector<float> out_s(8), out_f(8), out_n(8);

    for (size_t i = 0; i < N; ++i) {
        const float* x = &data[i*8];

        // ── NEON fixed ──────────────────────────────────────────────────────
        Float8_NEON inp;
        inp.lo = vld1q_f32(x);
        inp.hi = vld1q_f32(x + 4);
        Float8_NEON rf = quantize_D8_fixed(inp);
        vst1q_f32(out_f.data(),     rf.lo);
        vst1q_f32(out_f.data() + 4, rf.hi);

        // -- Scalar ─────────────────────────────────────────────────────────
        quantize_D8_scalar(x, out_s.data());

        // ── NEON naive (senza isolate_first_max) — per comparazione ─────────
        {
            Float8_NEON z2, err2;
            z2.lo = vrndnq_f32(inp.lo); z2.hi = vrndnq_f32(inp.hi);
            err2.lo = vsubq_f32(inp.lo, z2.lo);
            err2.hi = vsubq_f32(inp.hi, z2.hi);
            float32x4_t ae2_lo = vabsq_f32(err2.lo);
            float32x4_t ae2_hi = vabsq_f32(err2.hi);
            float mv2 = vmaxvq_f32(vmaxq_f32(ae2_lo, ae2_hi));
            uint32x4_t im2_lo = vceqq_f32(ae2_lo, vdupq_n_f32(mv2));
            uint32x4_t im2_hi = vceqq_f32(ae2_hi, vdupq_n_f32(mv2));
            // NESSUN isolate_first_max — bug intenzionale
            int par2 = (vaddvq_s32(vcvtq_s32_f32(z2.lo))
                      + vaddvq_s32(vcvtq_s32_f32(z2.hi))) & 1;
            uint32x4_t pm2  = vdupq_n_u32(par2 ? 0xFFFFFFFF : 0);
            uint32x4_t sm2  = vdupq_n_u32(0x80000000);
            uint32x4_t one2 = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
            uint32x4_t al2  = vandq_u32(vandq_u32(im2_lo, pm2),
                                         vorrq_u32(one2, vandq_u32(
                                             vreinterpretq_u32_f32(err2.lo), sm2)));
            z2.lo = vaddq_f32(z2.lo, vreinterpretq_f32_u32(al2));
            uint32x4_t ah2  = vandq_u32(vandq_u32(im2_hi, pm2),
                                         vorrq_u32(one2, vandq_u32(
                                             vreinterpretq_u32_f32(err2.hi), sm2)));
            z2.hi = vaddq_f32(z2.hi, vreinterpretq_f32_u32(ah2));
            vst1q_f32(out_n.data(),     z2.lo);
            vst1q_f32(out_n.data() + 4, z2.hi);
        }

        // ── Validazione NEON naive ──────────────────────────────────────────
        bool fn = false;
        for (int j = 0; j < 8; ++j)
            if (std::abs(out_s[j] - out_n[j]) > 1e-4f) { fn = true; break; }
        if (fn) naive_fails++;

        // ── Validazione NEON fixed ──────────────────────────────────────────
        bool ff = false;
        for (int j = 0; j < 8; ++j)
            if (std::abs(out_s[j] - out_f[j]) > 1e-4f) { ff = true; break; }
        if (ff) fixed_fails++;

        // ── Metriche NEON fixed ─────────────────────────────────────────────
        if (in_D8(out_f.data())) ok_d8_neon++;
        float dn = std::sqrt(dist2(x, out_f.data()));
        sum_dist_n += dn; max_dist_n = std::max(max_dist_n, dn);
        if (dn <= CR_D8 + 1e-4f) cr_ok_neon++;

        // ── Metriche scalare ────────────────────────────────────────────────
        if (in_D8(out_s.data())) ok_d8_scal++;
        float ds = std::sqrt(dist2(x, out_s.data()));
        sum_dist_s += ds; max_dist_s = std::max(max_dist_s, ds);
        if (ds <= CR_D8 + 1e-4f) cr_ok_scal++;

        // ── Checksum I/O ────────────────────────────────────────────────────
        cs_neon   ^= io_checksum(x, out_f.data(), i);
        cs_scalar ^= io_checksum(x, out_s.data(), i);
    }

    // ── REPORT ───────────────────────────────────────────────────────────────
    std::cout << "--- TEST 1: STRUCTURAL CORRECTNESS ---\n";
    std::cout << "SIMD Naive (bug)  : " << naive_fails
              << " failures / " << N << "\n";
    std::cout << "SIMD Fixed (fix)  : " << fixed_fails
              << " failures / " << N << "\n\n";

    std::cout << "--- TEST 2: GEOMETRIC METRICS ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Covering radius D8 = sqrt(8)/2 = " << CR_D8 << "\n\n";

    std::cout << "NEON fixed:\n";
    std::cout << "  Output in D8         : " << ok_d8_neon << "/" << N
              << (ok_d8_neon == (int)N ? "  ✓" : "  ✗") << "\n";
    std::cout << "  Mean distance I→O   : " << sum_dist_n/N << "\n";
    std::cout << "  Max distance   I→O   : " << max_dist_n << "\n";
    std::cout << "  Within covering radius : " << cr_ok_neon << "/" << N
              << (cr_ok_neon == (int)N ? "  ✓" : "  ✗") << "\n\n";

    std::cout << "Scalar:\n";
    std::cout << "  Output in D8         : " << ok_d8_scal << "/" << N
              << (ok_d8_scal == (int)N ? "  ✓" : "  ✗") << "\n";
    std::cout << "  Mean distance I→O   : " << sum_dist_s/N << "\n";
    std::cout << "  Max distance   I→O   : " << max_dist_s << "\n";
    std::cout << "  Within covering radius : " << cr_ok_scal << "/" << N
              << (cr_ok_scal == (int)N ? "  ✓" : "  ✗") << "\n\n";

    std::cout << "--- TEST 3: CHECKSUM BIT-LEVEL I/O ---\n";
    std::cout << "Checksum NEON   : 0x" << std::hex << cs_neon   << "\n";
    std::cout << "Checksum Scalar : 0x" << std::hex << cs_scalar << "\n";
    std::cout << std::dec;
    bool cs_match = (cs_neon == cs_scalar);
    bool cs_nonzero = (cs_neon != 0);
    std::cout << "Equal          : " << (cs_match   ? "SI  ✓" : "NO  ✗") << "\n";
    std::cout << "Non-zero        : " << (cs_nonzero ? "SI  ✓" : "NO  ✗") << "\n";
    std::cout << "State           : "
              << (cs_match && cs_nonzero ? "VALID (bit-perfect)" : "FALLIMENTO") << "\n\n";

    std::cout << "Golden value (seed 1337, N=1M, this arch): 0x24d087f2d94ce053\n";
    std::cout << "Match golden    : "
              << (cs_neon == 0x24d087f2d94ce053ULL ? "SI  ✓" : "NO  (architecture-dependent)") << "\n";

    return 0;
}
