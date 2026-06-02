# E8-SIMD-Fix

**Silent Corruption in SIMD Lattice Quantization: A Tie-Breaking Bug in D₈/E₈ Closest-Vector Decoders and Its Deterministic Fix**

Andrea Catino — Independent Researcher, Italy
catino.andrea@gmail.com
arXiv preprint — cs.AR, cs.MS — June 2026

---

## Summary

This repository contains the complete test harness for the paper above.

We identify and fix a **silent data-corruption bug** in AVX2 and NEON
implementations of the Conway–Sloane D₈/E₈ closest-vector quantizer.

**The bug:** when two or more vector components share the maximum absolute
quantization error (*ex-aequo*), the standard SIMD mask comparison activates
all tied lanes simultaneously. The applied correction sum is ±N instead of ±1,
leaving the output outside D₈ with no hardware exception.

**Failure rate:** 71.6% on the random-base pathological generator
(715,561/1,000,000 vectors), up to 91.3% with a fixed base (worst case),
~1.5–1.7% on random continuous inputs.

**The fix:** 2 instructions — `vmovmskps` + `(mask & -mask)` on AVX2;
store-and-scan on NEON. Isolates exactly one lane. Zero failures after fix.

---

## Files

| File | Description |
|------|-------------|
| `checksum_avx2.cxx` | Full harness: bug exposure, fix validation, geometric checks, I/O checksum (x86 AVX2) |
| `checksum_neon.cxx` | Same harness for ARM NEON |
| `e8_avx2_fixed.cxx` | Standalone E8 quantizer — AVX2, corrected |
| `e8_neon_fixed.cxx` | Standalone E8 quantizer — NEON, corrected |
| `interstellar_sim_avx2_v3.cxx` | BER/SER simulator: D8 lattice coding over AWGN+ADC vs BPSK |
| `simd_lattice_paper.tex` | LaTeX source of the paper |
| `simd_lattice_paper.pdf` | Compiled paper (preprint) |

---

## Build

### AVX2 (Linux / Google Colab x86)
```bash
g++ -O3 -mavx2 -mfma -std=c++17 checksum_avx2.cxx -o checksum_avx2
./checksum_avx2
```

### NEON (ARM64 — Android/CxxDroid, or Linux ARM)
```bash
clang++ -O3 -march=armv8-a+simd checksum_neon.cxx -o checksum_neon
./checksum_neon
```

### Standalone quantizer
```bash
g++ -O3 -mavx2 -mfma -std=c++17 e8_avx2_fixed.cxx -o e8_avx2_fixed && ./e8_avx2_fixed
```

---

## Expected Output

```
--- TEST 1: STRUCTURAL CORRECTNESS ---
SIMD Naive (bug)  : 715561 failures / 1000000    # ~71.6% (random base)
SIMD Fixed (fix)  : 0 failures / 1000000          # PASS

--- TEST 2: GEOMETRIC METRICS ---
Covering radius D8 = sqrt(8)/2 = 1.414214
Fixed:
  Output in D8         : 1000000/1000000  OK
  Mean distance I->O   : ~0.761
  Max  distance I->O   : ~0.848   (< 1.414 covering radius)
  Within covering rad. : 1000000/1000000  OK

--- TEST 3: BIT-LEVEL I/O CHECKSUM ---
  Checksum == its own scalar reference  : YES (bit-perfect)
```

### Golden checksums (seed 1337, N = 1,000,000) — architecture-specific

| Platform | Toolchain | Golden checksum |
|----------|-----------|-----------------|
| ARM64 Cortex-A55 | clang | `0x1866e1701487dc06` |
| Ubuntu x86 | g++ | `0x24d087f2d94ce053` |
| Google Colab x86 | g++ | `0x2520b5e88f1750a5` |

**Why three different values:** the tie-break selects the *first* tied lane,
and the lane/coset chosen can differ across toolchains and CPUs. All three are
valid D₈ points. The correctness criterion is that **each implementation
matches its own scalar reference on the same machine** — which it does,
bit-perfectly.

---

## Two implementation notes documented in the paper

### 1. IEEE-754 signed zero (Section 6)
Bit-level checksums must normalize `-0.0f` to `+0.0f` before extracting bits,
because `vrndnq_f32`/`_mm256_round_ps` and `std::round` may disagree on the
sign of zero:
```cpp
if (f == 0.0f) f = 0.0f;          // -0.0f -> +0.0f
uint32_t u; memcpy(&u, &f, 4);
```

### 2. Rounding mode of the scalar reference
`_mm256_round_ps` and `vrndnq_f32` use **round-half-to-even** (banker's):
`round(0.5)=0`, `round(2.5)=2`. The C library `roundf` uses
**round-half-away-from-zero**: `roundf(0.5)=1`. To make the scalar reference
bit-identical to the SIMD path on exact ±0.5 inputs, the scalar **must** use
`std::nearbyint` (half-to-even), not `roundf`. The standalone files use
`std::nearbyint` for this reason.

The remaining rare scalar-vs-SIMD differences (e.g. 999/1000 in the stress
test) are **coset ties**: the integer coset and the spinor coset are exactly
equidistant from the input, so the two paths pick different but equally valid
E�� points (identical squared distance). This is a tie-break convention, not an
error.

---

## The Fix (3 lines, AVX2)

```cpp
// BEFORE (bug): is_max may have k > 1 bits set
__m256 is_max = _mm256_cmp_ps(abs_err, max_err_bcast, _CMP_EQ_OQ);

// AFTER (fix): isolate the least-significant set bit only
__m256 is_max = _mm256_cmp_ps(abs_err, max_err_bcast, _CMP_EQ_OQ);
int raw    = _mm256_movemask_ps(is_max);
int single = raw & (-raw);               // BLSI: isolates LSB
// reconstruct single-lane mask from 'single' and use in place of is_max
```

---

## Pathological Input Generator

Seed: **1337**. Forces ex-aequo in every vector:
- Base integers with **odd parity sum** (forces a correction)
- Two components at **±0.485** (symmetric, equal absolute error → tie)
- Other components: uniform noise in [-0.2, +0.2]

The ~1.5% failure rate on random continuous inputs reflects the natural
probability of a floating-point tie; the pathological generator raises it to
the worst case.

---

## Citation

```bibtex
@misc{catino2026simd,
  title  = {Silent Corruption in SIMD Lattice Quantization:
             A Tie-Breaking Bug in D8/E8 Closest-Vector Decoders
             and Its Deterministic Fix},
  author = {Catino, Andrea},
  year   = {2026},
  note   = {arXiv preprint cs.AR, cs.MS},
  url    = {https://github.com/catinoandrea-lgtm/E8-SIMD-Fix}
}
```

---

## References

1. M. Viazovska, *The sphere packing problem in dimension 8*,
   Annals of Mathematics 185(3), 991–1015, 2017.
2. J.H. Conway, N.J.A. Sloane, *Fast quantizing and decoding algorithms
   for lattice quantizers and codes*, IEEE Trans. Inf. Theory, 28(2):227–232, 1982.
3. J.H. Conway, N.J.A. Sloane, *Sphere Packings, Lattices and Groups*,
   3rd ed., Springer, 1999.
4. B.M. Kurkoski, *The E8 Lattice and Error Correction in Multi-Level Flash
   Memory*, IEEE ICC 2011.
5. J. Johnson, M. Douze, H. Jégou, *Billion-scale similarity search with GPUs*,
   IEEE Trans. Big Data 7(3), 2019.
6. H.S. Warren Jr., *Hacker's Delight*, 2nd ed., Addison-Wesley, 2012.
