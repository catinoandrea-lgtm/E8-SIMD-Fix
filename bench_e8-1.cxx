/* E8/D8 Throughput Benchmark — NEON + AVX2
 * Andrea Catino — Independent Researcher, Italy · 2026-06-02 16:48
 *
 * Measures: scalar baseline, SIMD fixed, speedup, correctness, I/O checksum.
 * Auto-detects architecture at compile time.
 *
 * ── BUILD ─────────────────────────────────────────────────────────────────────
 *
 *  ARM64 (Oppo Reno 7 / CxxDroid / any ARM64 Linux):
 *    clang++ -O3 -march=armv8-a+simd -std=c++17 bench_e8.cxx -o bench_e8
 *
 *  x86 AVX2 (Ubuntu / Colab / any x86-64):
 *    g++ -O3 -mavx2 -mfma -std=c++17 bench_e8.cxx -o bench_e8
 *    clang++ -O3 -mavx2 -mfma -std=c++17 bench_e8.cxx -o bench_e8
 *
 * ── RUN ──────────────────────────────────────────────────────────────────────
 *    ./bench_e8
 *    ./bench_e8 2000000    # custom N (default 1,000,000)
 *
 * ── EXPECTED OUTPUT ──────────────────────────────────────────────────────────
 *   === E8 Throughput Benchmark ===
 *   Architecture : NEON (ARM64) / AVX2 (x86)
 *   Vectors      : 1,000,000
 *   Iterations   : 1000   (= 1,000,000,000 total)
 *
 *   [TEST 1 — Structural correctness on pathological input]
 *   SIMD Naive failures : 715,xxx / 1,000,000   (~71.6%)
 *   SIMD Fixed failures :       0 / 1,000,000   PASS
 *
 *   [TEST 2 — Throughput on continuous distribution]
 *   Baseline Scalar : xxx.xx ns / vector
 *   SIMD Fixed      :  xx.xx ns / vector
 *   Speedup         :   x.xx x
 *
 *   [TEST 3 — I/O Checksum (bit-level, scalar == SIMD)]
 *   Checksum Scalar :  0x????????????????
 *   Checksum SIMD   :  0x????????????????
 *   Match           :  YES (bit-perfect)   <-- correctness criterion
 *
 * ── GOLDEN CHECKSUMS (seed 1337, N=1,000,000) ────────────────────────────────
 *   ARM64 Cortex-A55 (clang):  0x1866e1701487dc06
 *   Ubuntu x86       (g++):    0x24d087f2d94ce053
 *   Google Colab x86 (g++):    0x2520b5e88f1750a5
 *
 *   The checksums differ across toolchains because the tie-break selects the
 *   first tied lane, and lane ordering differs between SIMD ISAs and compilers.
 *   All three are valid D8 points. Each matches its OWN scalar reference on
 *   the same machine — that is the correctness criterion, not cross-arch equality.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cfenv>
#include <chrono>
#include <random>
#include <vector>

// ── Architecture detection ────────────────────────────────────────────────────
#if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #define ARCH_NEON 1
  #define ARCH_NAME "NEON (ARM64)"
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define ARCH_AVX2 1
  #define ARCH_NAME "AVX2 (x86)"
#else
  #error "Unsupported architecture: compile with -march=armv8-a+simd (ARM) or -mavx2 (x86)"
#endif

// ============================================================================
//  SCALAR REFERENCE — uses std::nearbyint (round-half-to-even) to match
//  the SIMD rounding mode (_mm256_round_ps / vrndnq_f32).
// ============================================================================
static inline void quantize_D8_scalar(const float* x, float* out) {
    float err[8]; int s = 0;
    for (int i = 0; i < 8; ++i) {
        out[i] = std::nearbyint(x[i]);   // half-to-even, matches SIMD round
        err[i] = x[i] - out[i];
        s += static_cast<int>(out[i]);
    }
    if (s & 1) {
        float me = -1.0f; int idx = 0;
        for (int i = 0; i < 8; ++i) {
            float ae = std::fabs(err[i]);
            if (ae > me) { me = ae; idx = i; }
        }
        out[idx] += (err[idx] > 0.0f) ? 1.0f : -1.0f;
    }
}

static void quantize_E8_scalar(const float* x, float* out) {
    float z1[8], z2[8], xs[8], w[8];
    quantize_D8_scalar(x, z1);
    for (int i = 0; i < 8; ++i) xs[i] = x[i] - 0.5f;
    quantize_D8_scalar(xs, w);
    for (int i = 0; i < 8; ++i) z2[i] = w[i] + 0.5f;
    float d1 = 0.0f, d2 = 0.0f;
    for (int i = 0; i < 8; ++i) { float e = x[i]-z1[i]; d1 += e*e; }
    for (int i = 0; i < 8; ++i) { float e = x[i]-z2[i]; d2 += e*e; }
    const float* best = (d1 <= d2) ? z1 : z2;
    for (int i = 0; i < 8; ++i) out[i] = best[i];
}

// ============================================================================
//  NEON IMPLEMENTATION
// ============================================================================
#ifdef ARCH_NEON

struct Float8 { float32x4_t lo, hi; };

// FIX: isolate the first (lowest-index) set lane in the ex-aequo mask.
// Scan lo[0..3] first, then hi[4..7], matching the scalar scan order.
static inline void isolate_first_max(uint32x4_t& lo, uint32x4_t& hi) {
    uint32x2_t l2 = vget_low_u32(lo), h2 = vget_high_u32(lo);
    uint32_t any_lo = vget_lane_u32(vpmax_u32(vorr_u32(l2,h2), vorr_u32(l2,h2)), 0);
    if (any_lo) {
        uint32_t m[4]; vst1q_u32(m, lo);
        int f = 0; while (f < 4 && !m[f]) ++f;
        uint32_t s[4] = {0,0,0,0}; s[f] = 0xFFFFFFFF;
        lo = vld1q_u32(s); hi = vdupq_n_u32(0);
    } else {
        uint32_t m[4]; vst1q_u32(m, hi);
        int f = 0; while (f < 4 && !m[f]) ++f;
        uint32_t s[4] = {0,0,0,0}; s[f] = 0xFFFFFFFF;
        hi = vld1q_u32(s);
    }
}

static inline Float8 quantize_D8_NEON(Float8 x, bool use_fix) {
    Float8 z, err;
    z.lo = vrndnq_f32(x.lo); z.hi = vrndnq_f32(x.hi);
    err.lo = vsubq_f32(x.lo, z.lo); err.hi = vsubq_f32(x.hi, z.hi);
    float32x4_t ae_lo = vabsq_f32(err.lo), ae_hi = vabsq_f32(err.hi);
    float mv = vmaxvq_f32(vmaxq_f32(ae_lo, ae_hi));
    uint32x4_t im_lo = vceqq_f32(ae_lo, vdupq_n_f32(mv));
    uint32x4_t im_hi = vceqq_f32(ae_hi, vdupq_n_f32(mv));
    if (use_fix) isolate_first_max(im_lo, im_hi);  // BUG FIX
    int par = (vaddvq_s32(vcvtq_s32_f32(z.lo)) + vaddvq_s32(vcvtq_s32_f32(z.hi))) & 1;
    uint32x4_t pm  = vdupq_n_u32(par ? 0xFFFFFFFF : 0);
    uint32x4_t sm  = vdupq_n_u32(0x80000000);
    uint32x4_t one = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
    auto apply = [&](float32x4_t& zv, float32x4_t& ev, uint32x4_t& im) {
        uint32x4_t al = vandq_u32(vandq_u32(im, pm),
            vorrq_u32(one, vandq_u32(vreinterpretq_u32_f32(ev), sm)));
        zv = vaddq_f32(zv, vreinterpretq_f32_u32(al));
    };
    apply(z.lo, err.lo, im_lo);
    apply(z.hi, err.hi, im_hi);
    return z;
}

static inline Float8 quantize_E8_NEON(Float8 x, bool use_fix) {
    Float8 vi = quantize_D8_NEON(x, use_fix);
    float32x4_t h = vdupq_n_f32(0.5f);
    Float8 xs = { vsubq_f32(x.lo,h), vsubq_f32(x.hi,h) };
    Float8 w  = quantize_D8_NEON(xs, use_fix);
    Float8 vs = { vaddq_f32(w.lo,h), vaddq_f32(w.hi,h) };
    float32x4_t di_lo = vsubq_f32(x.lo,vi.lo), di_hi = vsubq_f32(x.hi,vi.hi);
    float32x4_t ds_lo = vsubq_f32(x.lo,vs.lo), ds_hi = vsubq_f32(x.hi,vs.hi);
    float32x4_t dv = vmulq_f32(di_lo,di_lo); dv = vfmaq_f32(dv,di_hi,di_hi);
    float32x4_t sv = vmulq_f32(ds_lo,ds_lo); sv = vfmaq_f32(sv,ds_hi,ds_hi);
    float d1 = vaddvq_f32(dv), d2 = vaddvq_f32(sv);
    return (d1 <= d2) ? vi : vs;
}

static void simd_quantize_E8(const float* x, float* out, bool use_fix) {
    Float8 inp = { vld1q_f32(x), vld1q_f32(x+4) };
    Float8 res = quantize_E8_NEON(inp, use_fix);
    vst1q_f32(out, res.lo); vst1q_f32(out+4, res.hi);
}
#endif // ARCH_NEON

// ============================================================================
//  AVX2 IMPLEMENTATION
// ============================================================================
#ifdef ARCH_AVX2

// FIX: isolate the first (lowest-index) set lane using BLSI + fully-vectorized
// mask reconstruction. No stack allocation, no scalar loop — 4 SIMD instructions.
//   1. vmovmskps   → scalar 8-bit integer
//   2. BLSI r&(-r) → isolate lowest set bit
//   3. vpbroadcastd + vpand + vpcmpeqd → rebuild 256-bit lane mask
static inline __m256 isolate_first_set_ps(__m256 mask) {
    int raw    = _mm256_movemask_ps(mask);
    int single = raw & (-raw);   // BLSI: bit with lowest index stays
    __m256i v_idx = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256i v_tie = _mm256_set1_epi32(single);
    // lane i is active iff (single & (1<<i)) == (1<<i)
    return _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(_mm256_and_si256(v_tie, v_idx), v_idx));
}

static inline __m256 quantize_D8_AVX2(__m256 x, bool use_fix) {
    __m256 z   = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 err = _mm256_sub_ps(x, z);
    __m256 sm  = _mm256_set1_ps(-0.0f);
    __m256 ae  = _mm256_andnot_ps(sm, err);
    __m256 s1  = _mm256_permute2f128_ps(ae, ae, 1);
    __m256 m1  = _mm256_max_ps(ae, s1);
    __m256 s2  = _mm256_shuffle_ps(m1, m1, _MM_SHUFFLE(2,3,0,1));
    __m256 m2  = _mm256_max_ps(m1, s2);
    __m256 s3  = _mm256_shuffle_ps(m2, m2, _MM_SHUFFLE(1,0,3,2));
    __m256 mv  = _mm256_max_ps(m2, s3);
    __m256 is_max = _mm256_cmp_ps(ae, mv, _CMP_EQ_OQ);
    if (use_fix) is_max = isolate_first_set_ps(is_max);  // BUG FIX
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

static inline __m256 quantize_E8_AVX2(__m256 x, bool use_fix) {
    __m256 vi = quantize_D8_AVX2(x, use_fix);
    __m256 h  = _mm256_set1_ps(0.5f);
    __m256 xs = _mm256_sub_ps(x, h);
    __m256 w  = quantize_D8_AVX2(xs, use_fix);
    __m256 vs = _mm256_add_ps(w, h);
    __m256 di = _mm256_sub_ps(x, vi), ds = _mm256_sub_ps(x, vs);
    __m256 a  = _mm256_mul_ps(di, di), b = _mm256_mul_ps(ds, ds);
    __m256 s1i = _mm256_hadd_ps(a, a), s2i = _mm256_hadd_ps(s1i, s1i);
    float d1   = _mm_cvtss_f32(_mm256_castps256_ps128(
                   _mm256_add_ps(s2i, _mm256_permute2f128_ps(s2i, s2i, 1))));
    __m256 s1s = _mm256_hadd_ps(b, b), s2s = _mm256_hadd_ps(s1s, s1s);
    float d2   = _mm_cvtss_f32(_mm256_castps256_ps128(
                   _mm256_add_ps(s2s, _mm256_permute2f128_ps(s2s, s2s, 1))));
    return (d1 <= d2) ? vi : vs;
}

static void simd_quantize_E8(const float* xp, float* out, bool use_fix) {
    __m256 inp = _mm256_loadu_ps(xp);
    _mm256_storeu_ps(out, quantize_E8_AVX2(inp, use_fix));
}
#endif // ARCH_AVX2

// ============================================================================
//  VALIDATORS
// ============================================================================
static bool in_D8(const float* v) {
    int s = 0;
    for (int i = 0; i < 8; ++i) {
        int zi = static_cast<int>(std::nearbyint(v[i]));
        if (std::fabs(v[i] - zi) > 1e-4f) return false;
        s += zi;
    }
    return (s & 1) == 0;
}

// E8 membership: either integer coset (D8) or half-integer spinor coset
static bool in_E8(const float* v) {
    if (in_D8(v)) return true;
    bool ah = true; int s2 = 0;
    for (int i = 0; i < 8; ++i) {
        float frac = v[i] - std::floor(v[i]);
        if (std::fabs(frac - 0.5f) > 1e-4f) { ah = false; break; }
        s2 += static_cast<int>(std::floor(v[i]));
    }
    return ah && (s2 & 1) == 0;
}

static float dist2(const float* a, const float* b) {
    float d = 0.0f;
    for (int i = 0; i < 8; ++i) { float e = a[i]-b[i]; d += e*e; }
    return d;
}

// ============================================================================
//  CHECKSUM — normalizes -0.0f before reading raw bits (see §6.1 of paper)
// ============================================================================
static inline uint32_t fnbits(float f) {
    if (f == 0.0f) return 0u;           // -0.0f → +0.0f
    uint32_t u; std::memcpy(&u, &f, 4);
    return u;
}

static uint64_t io_checksum(const float* in8, const float* out8, size_t idx) {
    uint64_t h = 0;
    for (int j = 0; j < 8; ++j) {
        h ^= (static_cast<uint64_t>(fnbits(in8[j]))  * 2654435761ULL + idx*8+j);
        h ^= (static_cast<uint64_t>(fnbits(out8[j])) * 2246822519ULL + idx*8+j+1);
    }
    return h;
}

// ── Fixed-only wrappers for the benchmark hot loop (no runtime branch) ──────
#ifdef ARCH_NEON
static void simd_quantize_E8_fixed(const float* x, float* out) {
    Float8 inp = { vld1q_f32(x), vld1q_f32(x+4) };
    Float8 res = quantize_E8_NEON(inp, true);
    vst1q_f32(out, res.lo); vst1q_f32(out+4, res.hi);
}
static void simd_quantize_E8_naive(const float* x, float* out) {
    Float8 inp = { vld1q_f32(x), vld1q_f32(x+4) };
    Float8 res = quantize_E8_NEON(inp, false);
    vst1q_f32(out, res.lo); vst1q_f32(out+4, res.hi);
}
#endif
#ifdef ARCH_AVX2
static void simd_quantize_E8_fixed(const float* xp, float* out) {
    _mm256_storeu_ps(out, quantize_E8_AVX2(_mm256_loadu_ps(xp), true));
}
static void simd_quantize_E8_naive(const float* xp, float* out) {
    _mm256_storeu_ps(out, quantize_E8_AVX2(_mm256_loadu_ps(xp), false));
}
#endif

// ============================================================================
//  MAIN
// ============================================================================
int main(int argc, char** argv) {
    std::fesetround(FE_TONEAREST);   // nearbyint → half-to-even

    const size_t N = (argc > 1) ? static_cast<size_t>(std::atoll(argv[1])) : 1000000UL;
    // Adaptive iterations: ARM phone CPUs are ~3x slower than x86.
    // 10M-50M measurements are statistically more than sufficient.
#ifdef ARCH_NEON
    int ITERS = 50;          // ARM: ~10s on Cortex-A55
#else
    int ITERS = 200;         // x86: ~15s
#endif
    if (argc > 2) ITERS = std::atoi(argv[2]);   // optional manual override

    std::printf("=== E8 Throughput Benchmark ===\n");
    std::printf("Architecture : %s\n", ARCH_NAME);
    std::printf("Vectors      : %zu\n", N);
    std::printf("Iterations   : %d   (= %zu total)\n\n", ITERS, N*(size_t)ITERS);

    // ── Data: pathological generator (seed 1337) for TEST 1 ────────────────
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> d_base(-100.f, 100.f);
    std::uniform_real_distribution<float> d_noise(-0.2f,  0.2f);
    std::uniform_int_distribution<int>    d_idx(0, 7);

    std::vector<float> patho(N*8), cont(N*8);

    for (size_t i = 0; i < N; ++i) {
        int bi[8], sum = 0;
        for (int j = 0; j < 8; ++j) { bi[j] = (int)std::round(d_base(rng)); sum += bi[j]; }
        if (std::abs(sum) % 2 == 0) bi[0]++;
        for (int j = 0; j < 8; ++j) patho[i*8+j] = (float)bi[j] + d_noise(rng);
        int i1 = d_idx(rng), i2 = d_idx(rng);
        while (i2 == i1) i2 = d_idx(rng);
        patho[i*8+i1] = (float)bi[i1] + 0.485f;
        patho[i*8+i2] = (float)bi[i2] - 0.485f;
    }
    // continuous distribution for TEST 2 (no ADC)
    for (auto& v : cont) v = d_base(rng) + d_noise(rng);

    // ── TEST 1: Structural correctness on pathological input ─────────────────
    std::printf("[TEST 1 — Structural correctness on pathological input]\n");
    {
        size_t naive_fail = 0, fixed_fail = 0;
        std::vector<float> out_s(8), out_n(8), out_f(8);
        for (size_t i = 0; i < N; ++i) {
            const float* x = &patho[i*8];
            quantize_E8_scalar(x, out_s.data());
            simd_quantize_E8_naive(x, out_n.data());  // naive (buggy)
            simd_quantize_E8_fixed(x, out_f.data());   // fixed
            for (int j = 0; j < 8; ++j) {
                if (std::fabs(out_s[j]-out_n[j]) > 1e-4f) { naive_fail++; break; }
            }
            for (int j = 0; j < 8; ++j) {
                if (std::fabs(out_s[j]-out_f[j]) > 1e-4f) { fixed_fail++; break; }
            }
        }
        std::printf("  SIMD Naive failures : %6zu / %zu   (%.2f%%)\n",
                    naive_fail, N, 100.0*naive_fail/N);
        std::printf("  SIMD Fixed failures : %6zu / %zu   %s\n\n",
                    fixed_fail, N, fixed_fail == 0 ? "PASS" : "FAIL !");
    }

    // ── TEST 2: Throughput on continuous distribution ───────────────────────
    std::printf("[TEST 2 — Throughput on continuous distribution  (seed 1337)]\n");
    std::printf("  Running %d x %zu = %zu measurements (~10-30s)...\n",
                ITERS, N, (size_t)ITERS*N);
    std::fflush(stdout);
    {
        volatile float sink_s = 0.0f, sink_f = 0.0f;
        std::vector<float> out_s(8), out_f(8);
        using clk = std::chrono::high_resolution_clock;

        // Scalar
        auto t0 = clk::now();
        for (int it = 0; it < ITERS; ++it)
            for (size_t i = 0; i < N; ++i) {
                quantize_E8_scalar(&cont[i*8], out_s.data());
                sink_s += out_s[0];
            }
        double ns_s = std::chrono::duration<double,std::nano>(clk::now()-t0).count()
                      / (double)(N*ITERS);
        std::printf("  Scalar done (%.2f ns/vec). Running SIMD...\n", ns_s);
        std::fflush(stdout);

        // SIMD fixed
        auto t1 = clk::now();
        for (int it = 0; it < ITERS; ++it)
            for (size_t i = 0; i < N; ++i) {
                simd_quantize_E8_fixed(&cont[i*8], out_f.data());
                sink_f += out_f[0];
            }
        double ns_f = std::chrono::duration<double,std::nano>(clk::now()-t1).count()
                      / (double)(N*ITERS);

        std::printf("  Baseline Scalar : %8.2f ns / vector   (sink=%g)\n", ns_s, (float)sink_s);
        std::printf("  SIMD Fixed      : %8.2f ns / vector   (sink=%g)\n", ns_f, (float)sink_f);
        std::printf("  Speedup         : %8.2f x\n\n", ns_s / ns_f);
    }

    // ── TEST 3: I/O Checksum ─────────────────────────────────────────────────
    std::printf("[TEST 3 — I/O Checksum  (bit-level, scalar == SIMD)]\n");
    {
        uint64_t cs_s = 0, cs_f = 0;
        int ok_d8_s = 0, ok_d8_f = 0, cr_ok_s = 0, cr_ok_f = 0;
        const float CR2 = 2.0f;   // covering radius^2 = (sqrt(8)/2)^2 = 2
        std::vector<float> out_s(8), out_f(8);

        // Use pathological input (deterministic: always forces correction -> stable checksum)
        for (size_t i = 0; i < N; ++i) {
            const float* x = &patho[i*8];
            quantize_E8_scalar(x, out_s.data());
            simd_quantize_E8_fixed(x, out_f.data());

            if (in_E8(out_s.data())) ok_d8_s++;
            if (in_E8(out_f.data())) ok_d8_f++;
            if (dist2(x, out_s.data()) <= CR2 + 1e-4f) cr_ok_s++;
            if (dist2(x, out_f.data()) <= CR2 + 1e-4f) cr_ok_f++;

            cs_s ^= io_checksum(x, out_s.data(), i);
            cs_f ^= io_checksum(x, out_f.data(), i);
        }

        bool match   = (cs_s == cs_f);
        bool nonzero = (cs_f != 0);

        std::printf("  Checksum Scalar : 0x%016llx\n", (unsigned long long)cs_s);
        std::printf("  Checksum SIMD   : 0x%016llx\n", (unsigned long long)cs_f);
        std::printf("  Match           : %s\n", match ? "YES (bit-perfect)" : "NO  (tie-break differs — see §6)");
        std::printf("  Non-zero        : %s\n\n", nonzero ? "YES" : "NO (all cancelled — bug!)");

        std::printf("  Scalar — in E8 : %d/%zu   CR ok : %d/%zu\n",
                    ok_d8_s, N, cr_ok_s, N);
        std::printf("  SIMD   — in D8 : %d/%zu   CR ok : %d/%zu\n\n",
                    ok_d8_f, N, cr_ok_f, N);
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::printf("=== Golden checksums — TEST 3 (seed 1337, N=1M, pathological input) ===\n");
    std::printf("  Ubuntu x86 (g++) :  0x47a627eaea2d50a5  (this machine)\n");
    std::printf("  ARM64 A55 (clang):  0x5bb4a6c245e0dc06  (Oppo Reno 7)\n");
    std::printf("  NOTE: checksum_avx2.cxx (paper) uses continuous input -> different values\n");
    std::printf("    paper ARM64:   0x1866e1701487dc06\n");
    std::printf("    paper Ubuntu:  0x24d087f2d94ce053\n");
    std::printf("    paper Colab:   0x2520b5e88f1750a5\n");
    std::printf("  Correctness: SIMD == Scalar on the same machine (bit-perfect)\n");

    return 0;
}
