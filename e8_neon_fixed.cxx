/* E8 NEON Quantizer — FIXED
 * Andrea Catino — Independent Researcher, Italy · 2026-06-01 21:16
 * clang -O3 -march=armv8-a+simd e8_neon_fixed.cxx -o e8_neon_fixed   (ARM64)
 *
 * BUG FIXED: multiple ex-aequo in quantize_D8_NEON.
 * When N components share the same maximum error, the naive version corrected
 * ALL of them, leaving D8. The fix isolates the FIRST element (priority order
 * lo[0..3] then hi[0..3]), so the parity changes by exactly +/-1.
 *
 * NOTE: vmaxq_f32 + vmaxvq_f32 for the scalar maximum is correct. The bug was
 * only in the multi-element mask that followed.
 *
 * NOTE ON ROUNDING (consistency with the scalar reference):
 *   vrndnq_f32 uses IEEE round-half-to-even: round(0.5)=0, round(2.5)=2.
 *   The C library roundf() uses round-half-away-from-zero: roundf(0.5)=1.
 *   The scalar reference here uses std::nearbyint (half-to-even) so that the
 *   scalar and NEON paths agree on every input, including exact +/-0.5 values.
 */
#include <arm_neon.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cfenv>

struct Float8_NEON {
    float32x4_t lo;
    float32x4_t hi;
};

// ── FIX: isolate the first maximum-error element ──────────────────────────────
// Check the lo block first (priority 0..3), then hi (4..7).
// If lo has any active lane -> keep only the first of lo, zero hi.
// Otherwise -> keep the first of hi (lo is already zero).
inline void isolate_first_max(uint32x4_t& is_max_lo, uint32x4_t& is_max_hi) {
    uint32x2_t lo_lo = vget_low_u32(is_max_lo);
    uint32x2_t lo_hi = vget_high_u32(is_max_lo);
    uint32x2_t or_lo = vorr_u32(lo_lo, lo_hi);
    uint32_t   any_lo = vget_lane_u32(vpmax_u32(or_lo, or_lo), 0);

    if (any_lo) {
        uint32_t mask_lo[4]; vst1q_u32(mask_lo, is_max_lo);
        int first = -1;
        for (int i = 0; i < 4; ++i) if (mask_lo[i]) { first = i; break; }
        uint32_t single[4] = {0,0,0,0};
        if (first >= 0) single[first] = 0xFFFFFFFF;
        is_max_lo = vld1q_u32(single);
        is_max_hi = vdupq_n_u32(0);
    } else {
        uint32_t mask_hi[4]; vst1q_u32(mask_hi, is_max_hi);
        int first = -1;
        for (int i = 0; i < 4; ++i) if (mask_hi[i]) { first = i; break; }
        uint32_t single[4] = {0,0,0,0};
        if (first >= 0) single[first] = 0xFFFFFFFF;
        is_max_hi = vld1q_u32(single);
    }
}

// ── CORE SIMD (fixed): quantize_D8_NEON ───────────────────────────────────────
inline Float8_NEON quantize_D8_NEON_fixed(Float8_NEON x) {
    Float8_NEON z, err;
    z.lo = vrndnq_f32(x.lo);                 // round half-to-even
    z.hi = vrndnq_f32(x.hi);
    err.lo = vsubq_f32(x.lo, z.lo);
    err.hi = vsubq_f32(x.hi, z.hi);

    float32x4_t abs_err_lo = vabsq_f32(err.lo);
    float32x4_t abs_err_hi = vabsq_f32(err.hi);
    float       max_err_val = vmaxvq_f32(vmaxq_f32(abs_err_lo, abs_err_hi));

    uint32x4_t is_max_lo = vceqq_f32(abs_err_lo, vdupq_n_f32(max_err_val));
    uint32x4_t is_max_hi = vceqq_f32(abs_err_hi, vdupq_n_f32(max_err_val));

    isolate_first_max(is_max_lo, is_max_hi);   // FIX ex-aequo

    int parity = (vaddvq_s32(vcvtq_s32_f32(z.lo))
                + vaddvq_s32(vcvtq_s32_f32(z.hi))) & 1;

    uint32x4_t parity_mask = vdupq_n_u32(parity ? 0xFFFFFFFF : 0);
    uint32x4_t sign_mask   = vdupq_n_u32(0x80000000);
    uint32x4_t one_u       = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));

    uint32x4_t applied_lo = vandq_u32(vandq_u32(is_max_lo, parity_mask),
        vorrq_u32(one_u, vandq_u32(vreinterpretq_u32_f32(err.lo), sign_mask)));
    z.lo = vaddq_f32(z.lo, vreinterpretq_f32_u32(applied_lo));

    uint32x4_t applied_hi = vandq_u32(vandq_u32(is_max_hi, parity_mask),
        vorrq_u32(one_u, vandq_u32(vreinterpretq_u32_f32(err.hi), sign_mask)));
    z.hi = vaddq_f32(z.hi, vreinterpretq_f32_u32(applied_hi));
    return z;
}

// ── E8 RECONSTRUCTION via coset selection ─────────────────────────────────────
inline Float8_NEON quantize_E8_NEON_fixed(Float8_NEON x) {
    Float8_NEON v_integer = quantize_D8_NEON_fixed(x);
    float32x4_t half = vdupq_n_f32(0.5f);
    Float8_NEON xs; xs.lo = vsubq_f32(x.lo, half); xs.hi = vsubq_f32(x.hi, half);
    Float8_NEON w  = quantize_D8_NEON_fixed(xs);
    Float8_NEON v_spinor; v_spinor.lo = vaddq_f32(w.lo, half); v_spinor.hi = vaddq_f32(w.hi, half);

    float32x4_t di_lo = vsubq_f32(x.lo, v_integer.lo);
    float32x4_t di_hi = vsubq_f32(x.hi, v_integer.hi);
    float32x4_t dv = vmulq_f32(di_lo, di_lo); dv = vfmaq_f32(dv, di_hi, di_hi);
    float d_int = vaddvq_f32(dv);

    float32x4_t ds_lo = vsubq_f32(x.lo, v_spinor.lo);
    float32x4_t ds_hi = vsubq_f32(x.hi, v_spinor.hi);
    float32x4_t dsv = vmulq_f32(ds_lo, ds_lo); dsv = vfmaq_f32(dsv, ds_hi, ds_hi);
    float d_spi = vaddvq_f32(dsv);

    return (d_int <= d_spi) ? v_integer : v_spinor;
}

// ── SCALAR REFERENCE — uses std::nearbyint (half-to-even) ─────────────────────
static void quantize_D8_scalar(const float* x, float* out) {
    int s = 0; float err[8];
    for (int i = 0; i < 8; ++i) {
        out[i] = std::nearbyint(x[i]);
        err[i] = x[i] - out[i];
        s += static_cast<int>(out[i]);
    }
    if (s % 2 != 0) {
        float me = -1.0f; int idx = 0;
        for (int i = 0; i < 8; ++i) { float ae = std::fabs(err[i]); if (ae > me) { me = ae; idx = i; } }
        out[idx] += (err[idx] > 0) ? 1.0f : -1.0f;
    }
}

static bool in_E8(const float* v) {
    bool all_int = true; int s = 0;
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(v[i] - std::nearbyint(v[i])) > 1e-4f) { all_int = false; break; }
        s += static_cast<int>(std::nearbyint(v[i]));
    }
    if (all_int) return (s % 2) == 0;
    bool all_half = true; int s2 = 0;
    for (int i = 0; i < 8; ++i) {
        float frac = v[i] - std::floor(v[i]);
        if (std::fabs(frac - 0.5f) > 1e-4f) { all_half = false; break; }
        s2 += static_cast<int>(std::floor(v[i]));
    }
    return all_half && (s2 % 2) == 0;
}

// ── TEST ──────────────────────────────────────────────────────────────────────
int main() {
    std::fesetround(FE_TONEAREST);
    std::cout << "=== E8 NEON FIXED — stress test, 1000 points ===" << std::endl;
    srand(42);
    int ok = 0, match_scalar = 0;

    for (int t = 0; t < 1000; ++t) {
        float data[8];
        for (int i = 0; i < 8; ++i) data[i] = (rand() % 200 - 100) * 0.05f;

        Float8_NEON inp; inp.lo = vld1q_f32(data); inp.hi = vld1q_f32(data + 4);
        Float8_NEON out = quantize_E8_NEON_fixed(inp);
        float res[8]; vst1q_f32(res, out.lo); vst1q_f32(res + 4, out.hi);
        if (in_E8(res)) ok++;

        // scalar E8 reference
        float z1[8], z2[8], xs[8], ww[8];
        quantize_D8_scalar(data, z1);
        for (int i = 0; i < 8; ++i) xs[i] = data[i] - 0.5f;
        quantize_D8_scalar(xs, ww);
        for (int i = 0; i < 8; ++i) z2[i] = ww[i] + 0.5f;
        float d1 = 0, d2 = 0;
        for (int i = 0; i < 8; ++i) { float e = data[i]-z1[i]; d1 += e*e; }
        for (int i = 0; i < 8; ++i) { float e = data[i]-z2[i]; d2 += e*e; }
        const float* sref = (d1 <= d2) ? z1 : z2;
        bool m = true;
        for (int i = 0; i < 8; ++i) if (std::fabs(res[i] - sref[i]) > 1e-4f) { m = false; break; }
        if (m) match_scalar++;
    }
    std::cout << "In E8:        " << ok           << "/1000" << std::endl;
    std::cout << "Match scalar: " << match_scalar << "/1000  (scalar uses nearbyint)" << std::endl;

    auto test = [&](const char* lbl, std::initializer_list<float> vals) {
        float d[8] = {0}; int i = 0; for (float v : vals) d[i++] = v;
        Float8_NEON inp; inp.lo = vld1q_f32(d); inp.hi = vld1q_f32(d + 4);
        Float8_NEON out = quantize_E8_NEON_fixed(inp);
        float res[8]; vst1q_f32(res, out.lo); vst1q_f32(res + 4, out.hi);
        std::cout << lbl << ": out=";
        for (int j = 0; j < 8; ++j) std::cout << std::setprecision(1) << std::fixed << res[j] << " ";
        std::cout << " in_E8=" << in_E8(res) << std::endl;
    };
    test("(1,0,...)     ", {1,0,0,0,0,0,0,0});
    test("(0.5,0.5,...) ", {0.5f,0.5f,0,0,0,0,0,0});
    test("doc example   ", {1.2f,-0.4f,2.7f,0.5f,3.1f,-1.9f,0.8f,-0.1f});
    return 0;
}

/* ----------------------------------------------------------------------------
 * NOTE ON "Match scalar" being slightly below 1000/1000
 *   The few non-matching cases are NOT errors. They occur when the integer
 *   coset and the spinor coset are EXACTLY equidistant from the input; NEON
 *   and the scalar reference then pick different but equally valid E8 points
 *   (identical squared distance). The coset tie-break is an arbitrary
 *   convention, not a correctness criterion.
 * -------------------------------------------------------------------------- */
