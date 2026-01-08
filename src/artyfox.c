#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#define ALWAYS_INLINE __attribute__((always_inline))
#define UNUSED __attribute__((unused))
#define CLAMP(x, min, max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : (x))) 
#define M_PI 3.14159265358979323846

typedef double (*kernel_func)(double x, void *ctx);

typedef struct {
    kernel_func f;
    double radius;
    void *ctx;
} kernel_t;

typedef struct {
    float gamma, thr_to, thr_from, corr, div;
    bool strict;
} GammaData;

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int dst_width, dst_height;
    double start_w, start_h, real_w, real_h;
    kernel_t kernel_w, kernel_h;
    GammaData gamma;
    float sharp;
    bool linear, process_w, process_h;
} ResizeData;

typedef struct {
    double scale;
} area_ctx;

static double area_kernel(double x, void *ctx) {
    area_ctx *ar = (area_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5 - ar->scale / 2.0) {
        return 1.0;
    }
    if (x < 0.5 + ar->scale / 2.0) {
        return 0.5 - (x - 0.5) / ar->scale;
    }
    return 0.0;
}

static double magic_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 0.75 - x * x;
    }
    if (x < 1.5) {
        x -= 1.5;
        return 0.5 * x * x;
    }
    return 0.0;
}

static double magic_kernel_2013(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 1.0625 - 1.75 * x * x;
    }
    if (x < 1.5) {
        return (1.0 - x) * (1.75 - x);
    }
    if (x < 2.5) {
        x -= 2.5;
        return -0.125 * x * x;
    }
    return 0.0;
}

static double magic_kernel_2021(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 577.0 / 576.0 - 239.0 / 144.0 * x * x;
    }
    if (x < 1.5) {
        return 35.0 / 36.0 * (x - 1.0) * (x - 239.0 / 140.0);
    }
    if (x < 2.5) {
        return 1.0 / 6.0 * (x - 2.0) * (65.0 / 24.0 - x);
    }
    if (x < 3.5) {
        return 1.0 / 36.0 * (x - 3.0) * (x - 3.75);
    }
    if (x < 4.5) {
        x -= 4.5;
        return -1.0 / 288.0 * x * x;
    }
    return 0.0;
}

static double bilinear_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return 1.0 - x;
    }
    return 0.0;
}

typedef struct {
    double b, c;
} bicubic_ctx;

static double bicubic_kernel(double x, void *ctx) {
    bicubic_ctx *bc = (bicubic_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return (
            ((12.0 - 9.0 * bc->b - 6.0 * bc->c) * x +
            (-18.0 + 12.0 * bc->b + 6.0 * bc->c)) * x * x +
            (6.0 - 2.0 * bc->b)
        ) / 6.0;
    }
    if (x < 2.0) {
        return (
            (((-bc->b - 6.0 * bc->c) * x +
            (6.0 * bc->b + 30.0 * bc->c)) * x +
            (-12.0 * bc->b - 48.0 * bc->c)) * x +
            (8.0 * bc->b + 24.0 * bc->c)
        ) / 6.0;
    }
    return 0.0;
}

typedef struct {
    int taps;
} sinc_ctx;

static inline double sinc_function(double x) {
    if (x == 0.0) {
        return 1.0;
    }
    x *= M_PI;
    return sin(x) / (x);
}

static double lanczos_kernel(double x, void *ctx) {
    sinc_ctx *sn = (sinc_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < sn->taps) {
        return sinc_function(x) * sinc_function(x / sn->taps);
    }
    return 0.0;
}

static double spline16_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((x - 1.8) * x - 0.2) * x + 1.0;
    }
    if (x < 2.0) {
        x -= 1.0;
        return ((-1.0 / 3.0 * x + 0.8) * x - 7.0 / 15.0) * x;
    }
    return 0.0;
}

static double spline36_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
    }
    if (x < 2.0) {
        x -= 1.0;
        return ((-6.0 / 11.0 * x + 270.0 / 209.0) * x - 156.0 / 209.0) * x;
    }
    if (x < 3.0) {
        x -= 2.0;
        return ((1.0 / 11.0 * x - 45.0 / 209.0) * x + 26.0 / 209.0) * x;
    }
    return 0.0;
}

static double spline64_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((49.0 / 41.0 * x - 6387.0 / 2911.0) * x - 3.0 / 2911.0) * x + 1.0;
    }
    if (x < 2.0) {
        x -= 1.0;
        return ((-24.0 / 41.0 * x + 4032.0 / 2911.0) * x - 2328.0 / 2911.0) * x;
    }
    if (x < 3.0) {
        x -= 2.0;
        return ((6.0 / 41.0 * x - 1008.0 / 2911.0) * x + 582.0 / 2911.0) * x;
    }
    if (x < 4.0) {
        x -= 3.0;
        return ((-1.0 / 41.0 * x + 168.0 / 2911.0) * x - 97.0 / 2911.0) * x;
    }
    return 0.0;
}

static double spline100_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((61.0 / 51.0 * x - 9893.0 / 4505.0) * x - 1.0 / 13515.0) * x + 1.0;
    }
    if (x < 2.0) {
        x -= 1.0;
        return ((-10.0 / 17.0 * x + 1254.0 / 901.0) * x - 724.0 / 901.0) * x;
    }
    if (x < 3.0) {
        x -= 2.0;
        return ((8.0 / 51.0 * x - 1672.0 / 4505.0) * x + 2896.0 / 13515.0) * x;
    }
    if (x < 4.0) {
        x -= 3.0;
        return ((-2.0 / 51.0 * x + 418.0 / 4505.0) * x - 724.0 / 13515.0) * x;
    }
    if (x < 5.0) {
        x -= 4.0;
        return ((1.0 / 153.0 * x - 209.0 / 13515.0) * x + 362.0 / 40545.0) * x;
    }
    return 0.0;
}

static double spline144_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        return ((683.0 / 571.0 * x - 1240203.0 / 564719.0) * x - 3.0 / 564719.0) * x + 1.0;
    }
    if (x < 2.0) {
        x -= 1.0;
        return ((-336.0 / 571.0 * x + 786240.0 / 564719.0) * x - 453936.0 / 564719.0) * x;
    }
    if (x < 3.0) {
        x -= 2.0;
        return ((90.0 / 571.0 * x - 210600.0 / 564719.0) * x + 121590.0 / 564719.0) * x;
    }
    if (x < 4.0) {
        x -= 3.0;
        return ((-24.0 / 571.0 * x + 56160.0 / 564719.0) * x - 32424.0 / 564719.0) * x;
    }
    if (x < 5.0) {
        x -= 4.0;
        return ((6.0 / 571.0 * x - 14040.0 / 564719.0) * x + 8106.0 / 564719.0) * x;
    }
    if (x < 6.0) {
        x -= 5.0;
        return ((-1.0 / 571.0 * x + 2340.0 / 564719.0) * x - 1351.0 / 564719.0) * x;
    }
    return 0.0;
}

static double point_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 1.0;
    }
    return 0.0;
}

static double blackman_kernel(double x, void *ctx) {
    sinc_ctx *sn = (sinc_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < sn->taps) {
        double nx = x * M_PI / sn->taps;
        return sinc_function(x) * (0.42 + 0.5 * cos(nx) + 0.08 * cos(nx * 2.0));
    }
    return 0.0;
}

static double nuttall_kernel(double x, void *ctx) {
    sinc_ctx *sn = (sinc_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < sn->taps) {
        double nx = x * M_PI / sn->taps;
        return sinc_function(x) * (0.355768 + 0.487396 * cos(nx) + 0.144232 * cos(nx * 2.0) + 0.012604 * cos(nx * 3.0));
    }
    return 0.0;
}

typedef struct {
    int taps;
    double beta, i0_beta;
} kaiser_ctx;

static inline double bessel_i0(double x) {
    double sum = 1.0, y = x * x / 4.0, t = y;
    int k = 1;
    while (t > 1e-16 * sum) {
        sum += t;
        k++;
        t *= y / (k * k);
    }
    return sum;
}

static double kaiser_kernel(double x, void *ctx) {
    kaiser_ctx *ks = (kaiser_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < ks->taps) {
        double nx = x / ks->taps;
        return sinc_function(x) * bessel_i0(ks->beta * sqrt(1.0 - nx * nx)) / ks->i0_beta;
    }
    return 0.0;
}

typedef struct {
    int taps;
    double p;
} gauss_ctx;

static double gauss_kernel(double x, void *ctx) {
    gauss_ctx *gs = (gauss_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < gs->taps) {
        return pow(2.0, -gs->p * x * x);
    }
    return 0.0;
}

#if defined(__AVX2__) && defined(__FMA__)

static void to_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, GammaData d
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_thr = _mm256_set1_ps(d.thr_to);
    __m256 v_corr = _mm256_set1_ps(d.corr);
    __m256 v_corr_one = _mm256_set1_ps(1.0f + d.corr);
    __m256 v_gamma = _mm256_set1_ps(d.gamma);
    __m256 v_div = _mm256_set1_ps(d.div);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    
    if (d.strict) {
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod8_w; x += 8) {
                __m256 pix = _mm256_load_ps(srcp + x);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
                __m256 branch_1 = _mm256_div_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
                __m256 branch_1 = _mm256_div_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            srcp += stride;
            dstp += stride;
        }
    }
    else {
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod8_w; x += 8) {
                __m256 pix = _mm256_load_ps(srcp + x);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GE_OQ);
                __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
                __m256 branch_1 = _mm256_div_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GE_OQ);
                __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
                __m256 branch_1 = _mm256_div_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            srcp += stride;
            dstp += stride;
        }
    }
    _mm_sfence();
}

static void from_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, GammaData d
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_thr = _mm256_set1_ps(d.thr_from);
    __m256 v_corr = _mm256_set1_ps(d.corr);
    __m256 v_corr_one = _mm256_set1_ps(1.0f + d.corr);
    __m256 v_gamma = _mm256_set1_ps(1.0f / d.gamma);
    __m256 v_div = _mm256_set1_ps(d.div);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    
    if (d.strict) {
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod8_w; x += 8) {
                __m256 pix = _mm256_load_ps(srcp + x);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_corr_one, v_corr);
                __m256 branch_1 = _mm256_mul_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_corr_one, v_corr);
                __m256 branch_1 = _mm256_mul_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            srcp += stride;
            dstp += stride;
        }
    }
    else {
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod8_w; x += 8) {
                __m256 pix = _mm256_load_ps(srcp + x);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GE_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_corr_one, v_corr);
                __m256 branch_1 = _mm256_mul_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GE_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_corr_one, v_corr);
                __m256 branch_1 = _mm256_mul_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            srcp += stride;
            dstp += stride;
        }
    }
    _mm_sfence();
}

static void uint8_to_uint16(
    const uint8_t *restrict srcp, uint16_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int bits
) {
    int count = bits - 8;
    int tail = src_w % 16;
    int mod16_w = src_w - tail;
    
    int8_t mask_arr[16] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m128i tail_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m256i pix = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i *)(srcp + x)));
            __m256i branch = _mm256_slli_epi16(pix, count);
            _mm256_stream_si256((__m256i *)(dstp + x), branch);
        }
        if (tail) {
            __m256i pix = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tail_mask));
            __m256i branch = _mm256_slli_epi16(pix, count);
            _mm256_stream_si256((__m256i *)(dstp + x), branch);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    _mm_sfence();
}

static void uint8_to_float(
    const uint8_t *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range
) {
    int tail = src_w % 16;
    int mod16_w = src_w - tail;
    
    int8_t mask_arr[16] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m128i tail_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(chroma ? 128.0f : 16.0f);
        v_high = _mm256_set1_ps(chroma ? 224.0f : 219.0f);
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 128.0f : 0.0f);
        v_high = _mm256_set1_ps(chroma ? 256.0f : 255.0f);
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m128i pix = _mm_load_si128((__m128i *)(srcp + x));
            __m256 pix_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pix));
            __m256 pix_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(pix, _MM_SHUFFLE(1, 0, 3, 2))));
            __m256 branch_0 = _mm256_div_ps(_mm256_sub_ps(pix_0, v_low), v_high);
            __m256 branch_1 = _mm256_div_ps(_mm256_sub_ps(pix_1, v_low), v_high);
            _mm256_stream_ps(dstp + x + 0, branch_0);
            _mm256_stream_ps(dstp + x + 8, branch_1);
        }
        if (tail > 8) {
            __m128i pix = _mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tail_mask);
            __m256 pix_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pix));
            __m256 pix_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(pix, _MM_SHUFFLE(1, 0, 3, 2))));
            __m256 branch_0 = _mm256_div_ps(_mm256_sub_ps(pix_0, v_low), v_high);
            __m256 branch_1 = _mm256_div_ps(_mm256_sub_ps(pix_1, v_low), v_high);
            _mm256_stream_ps(dstp + x + 0, branch_0);
            _mm256_stream_ps(dstp + x + 8, branch_1);
        }
        else if (tail) {
            __m128i pix = _mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tail_mask);
            __m256 pix_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pix));
            __m256 branch_0 = _mm256_div_ps(_mm256_sub_ps(pix_0, v_low), v_high);
            _mm256_stream_ps(dstp + x, branch_0);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    _mm_sfence();
}

static void uint16_to_uint8(
    const uint16_t *restrict srcp, uint8_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int bits
) {
    int tail = src_w % 16;
    int mod16_w = src_w - tail;
    
    int16_t mask_arr[16] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    int count = bits - 8;
    __m256i v_half = _mm256_set1_epi16(1 << (count - 1));
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m256i pix = _mm256_load_si256((__m256i *)(srcp + x));
            __m256i branch = _mm256_srli_epi16(_mm256_adds_epu16(pix, v_half), count);
            __m128i branch_u_0 = _mm256_extracti128_si256(branch, 0);
            __m128i branch_u_1 = _mm256_extracti128_si256(branch, 1);
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(branch_u_0, branch_u_1));
        }
        if (tail) {
            __m256i pix = _mm256_and_si256(_mm256_load_si256((__m256i *)(srcp + x)), tail_mask);
            __m256i branch = _mm256_srli_epi16(_mm256_adds_epu16(pix, v_half), count);
            __m128i branch_u_0 = _mm256_extracti128_si256(branch, 0);
            __m128i branch_u_1 = _mm256_extracti128_si256(branch, 1);
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(branch_u_0, branch_u_1));
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    _mm_sfence();
}

static void uint16_to_uint16(
    const uint16_t *restrict srcp, uint16_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits
) {
    int tail = src_w % 16;
    int mod16_w = src_w - tail;
    
    int16_t mask_arr[16] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    if (src_bits < dst_bits) {
        int count = dst_bits - src_bits;
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod16_w; x += 16) {
                __m256i pix = _mm256_load_si256((__m256i *)(srcp + x));
                __m256i branch = _mm256_slli_epi16(pix, count);
                _mm256_stream_si256((__m256i *)(dstp + x), branch);
            }
            if (tail) {
                __m256i pix = _mm256_and_si256(_mm256_load_si256((__m256i *)(srcp + x)), tail_mask);
                __m256i branch = _mm256_slli_epi16(pix, count);
                _mm256_stream_si256((__m256i *)(dstp + x), branch);
            }
            srcp += src_stride;
            dstp += dst_stride;
        }
    }
    else {
        int count = src_bits - dst_bits;
        __m256i v_half = _mm256_set1_epi16(1 << (count - 1));
        __m256i v_max = _mm256_set1_epi16((1 << dst_bits) - 1);
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod16_w; x += 16) {
                __m256i pix = _mm256_load_si256((__m256i *)(srcp + x));
                __m256i branch = _mm256_min_epu16(_mm256_srli_epi16(_mm256_adds_epu16(pix, v_half), count), v_max);
                _mm256_stream_si256((__m256i *)(dstp + x), branch);
            }
            if (tail) {
                __m256i pix = _mm256_and_si256(_mm256_load_si256((__m256i *)(srcp + x)), tail_mask);
                __m256i branch = _mm256_min_epu16(_mm256_srli_epi16(_mm256_adds_epu16(pix, v_half), count), v_max);
                _mm256_stream_si256((__m256i *)(dstp + x), branch);
            }
            srcp += src_stride;
            dstp += dst_stride;
        }
    }
    _mm_sfence();
}

static void uint16_to_float(
    const uint16_t *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range, int bits
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int16_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m128i tail_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps((float)((chroma ? 128 : 16) << (bits - 8)));
        v_high = _mm256_set1_ps((float)((chroma ? 224 : 219) << (bits - 8)));
    }
    else {
        v_low = _mm256_set1_ps(chroma ? (float)(128 << (bits - 8)) : 0.0f);
        v_high = _mm256_set1_ps(chroma ? (float)(1 << bits) : (float)((1 << bits) - 1));
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m128i pix = _mm_load_si128((__m128i *)(srcp + x));
            __m256 pix_f = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(pix));
            __m256 branch = _mm256_div_ps(_mm256_sub_ps(pix_f, v_low), v_high);
            _mm256_stream_ps(dstp + x, branch);
        }
        if (tail) {
            __m128i pix = _mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tail_mask);
            __m256 pix_f = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(pix));
            __m256 branch = _mm256_div_ps(_mm256_sub_ps(pix_f, v_low), v_high);
            _mm256_stream_ps(dstp + x, branch);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    _mm_sfence();
}

static void float_to_uint8(
    const float *restrict srcp, uint8_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range
) {
    int tail = src_w % 16;
    int mod16_w = src_w - tail;
    
    int32_t mask_arr[16] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask_0 = _mm256_loadu_si256((__m256i *)mask_arr);
    __m256i tail_mask_1 = _mm256_loadu_si256((__m256i *)(mask_arr + 8));
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(chroma ? 128.5f : 16.5f);
        v_high = _mm256_set1_ps(chroma ? 224.0f : 219.0f);
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 128.5f : 0.5f);
        v_high = _mm256_set1_ps(chroma ? 256.0f : 255.0f);
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m256 pix_f_0 = _mm256_load_ps(srcp + x + 0);
            __m256 pix_f_1 = _mm256_load_ps(srcp + x + 8);
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_0, v_high, v_low));
            __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_1, v_high, v_low));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            __m128i pix_u_1 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_1, 0), _mm256_extracti128_si256(pix_i_1, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, pix_u_1));
        }
        if (tail > 8) {
            __m256 pix_f_0 = _mm256_load_ps(srcp + x + 0);
            __m256 pix_f_1 = _mm256_maskload_ps(srcp + x + 8, tail_mask_1);
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_0, v_high, v_low));
            __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_1, v_high, v_low));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            __m128i pix_u_1 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_1, 0), _mm256_extracti128_si256(pix_i_1, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, pix_u_1));
        }
        else if (tail) {
            __m256 pix_f_0 = _mm256_maskload_ps(srcp + x, tail_mask_0);
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_0, v_high, v_low));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, _mm_setzero_si128()));
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    _mm_sfence();
}

static void float_to_uint16(
    const float *restrict srcp, uint16_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range, int bits
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256i v_max = _mm256_set1_epi32((1 << bits) - 1);
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(0.5f + (float)((chroma ? 128 : 16) << (bits - 8)));
        v_high = _mm256_set1_ps((float)((chroma ? 224 : 219) << (bits - 8)));
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 0.5f + (float)(128 << (bits - 8)) : 0.5f);
        v_high = _mm256_set1_ps(chroma ? (float)(1 << bits) : (float)((1 << bits) - 1));
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix_f = _mm256_load_ps(srcp + x);
            __m256i pix_i = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f, v_high, v_low)), v_max);
            __m128i pix_u = _mm_packus_epi32(_mm256_extracti128_si256(pix_i, 0), _mm256_extracti128_si256(pix_i, 1));
            _mm_stream_si128((__m128i *)(dstp + x), pix_u);
        }
        if (tail) {
            __m256 pix_f = _mm256_maskload_ps(srcp + x, tail_mask);
            __m256i pix_i = _mm256_min_epi32(_mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f, v_high, v_low)), v_max);
            __m128i pix_u = _mm_packus_epi32(_mm256_extracti128_si256(pix_i, 0), _mm256_extracti128_si256(pix_i, 1));
            _mm_stream_si128((__m128i *)(dstp + x), pix_u);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
    _mm_sfence();
}

static void sharp_width(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float sharp
) {
    int tail = src_w % 8;
    if (!tail) tail = 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[9] = {0};
    for (int i = 0; i < tail + 1; i++) mask_arr[i] = -1;
    __m256i tail_mask_0 = _mm256_loadu_si256((__m256i *)mask_arr);
    __m256i tail_mask_1 = _mm256_loadu_si256((__m256i *)(mask_arr + 1));
    
    __m256 v_sharp = _mm256_set1_ps(sharp);
    __m256 v_mul = _mm256_set1_ps((1.0f - sharp) / 3.0f);
    __m256i left_idx =  _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
    
    int32_t right_arr[8];
    for (int i = 0; i < 8; i++) right_arr[i] = (i < tail - 1) ? i + 1 : tail - 1;
    __m256i right_idx = _mm256_loadu_si256((__m256i *)right_arr);
    
    for (int y = 0; y < src_h; y++) {
        __m256 v_1 = _mm256_load_ps(srcp);
        __m256 v_2 = _mm256_loadu_ps(srcp + 1);
        __m256 v_0 = _mm256_permutevar8x32_ps(v_1, left_idx);
        __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_2), v_mul);
        _mm256_stream_ps(dstp, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
        int x = 8;
        for (; x < mod8_w; x += 8) {
            v_0 = _mm256_loadu_ps(srcp + x - 1);
            v_1 = _mm256_load_ps(srcp + x);
            v_2 = _mm256_loadu_ps(srcp + x + 1);
            v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_2), v_mul);
            _mm256_stream_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
        }
        v_0 = _mm256_maskload_ps(srcp + x - 1, tail_mask_0);
        v_1 = _mm256_maskload_ps(srcp + x, tail_mask_1);
        v_2 = _mm256_permutevar8x32_ps(v_1, right_idx);
        v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_2), v_mul);
        _mm256_stream_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
        
        srcp += stride;
        dstp += stride;
    }
    _mm_sfence();
}

static void sharp_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float sharp
) {
    int border = src_h - 1;
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_sharp = _mm256_set1_ps(sharp);
    __m256 v_mul = _mm256_set1_ps((1.0f - sharp) / 3.0f);
    
    int x = 0;
    for (; x < mod8_w; x += 8) {
        __m256 v_1 = _mm256_load_ps(srcp + x);
        __m256 v_2 = _mm256_load_ps(srcp + x + stride);
        __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_1, v_1), v_2), v_mul);
        _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
    }
    if (tail) {
        __m256 v_1 = _mm256_maskload_ps(srcp + x, tail_mask);
        __m256 v_2 = _mm256_maskload_ps(srcp + x + stride, tail_mask);
        __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_1, v_1), v_2), v_mul);
        _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
    }
    srcp += stride;
    dstp += stride;
    
    for (int y = 1; y < border; y++) {
        x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 v_0 = _mm256_load_ps(srcp + x - stride);
            __m256 v_1 = _mm256_load_ps(srcp + x);
            __m256 v_2 = _mm256_load_ps(srcp + x + stride);
            __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_2), v_mul);
            _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
        }
        if (tail) {
            __m256 v_0 = _mm256_maskload_ps(srcp + x - stride, tail_mask);
            __m256 v_1 = _mm256_maskload_ps(srcp + x, tail_mask);
            __m256 v_2 = _mm256_maskload_ps(srcp + x + stride, tail_mask);
            __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_2), v_mul);
            _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
        }
        srcp += stride;
        dstp += stride;
    }
    
    x = 0;
    for (; x < mod8_w; x += 8) {
        __m256 v_0 = _mm256_load_ps(srcp + x - stride);
        __m256 v_1 = _mm256_load_ps(srcp + x);
        __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_1), v_mul);
        _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
    }
    if (tail) {
        __m256 v_0 = _mm256_maskload_ps(srcp + x - stride, tail_mask);
        __m256 v_1 = _mm256_maskload_ps(srcp + x, tail_mask);
        __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_1), v_mul);
        _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
    }
}

static void _mm256_transpose8_lane4_ps(__m256 *row0, __m256 *row1, __m256 *row2, __m256 *row3) {
    __m256 t0 = _mm256_unpacklo_ps(*row0, *row2);
    __m256 t1 = _mm256_unpackhi_ps(*row0, *row2);
    __m256 t2 = _mm256_unpacklo_ps(*row1, *row3);
    __m256 t3 = _mm256_unpackhi_ps(*row1, *row3);
    __m256 u0 = _mm256_unpacklo_ps(t0, t2);
    __m256 u1 = _mm256_unpackhi_ps(t0, t2);
    __m256 u2 = _mm256_unpacklo_ps(t1, t3);
    __m256 u3 = _mm256_unpackhi_ps(t1, t3);
    *row0 = _mm256_permute2f128_ps(u0, u1, 0b00100000);
    *row1 = _mm256_permute2f128_ps(u2, u3, 0b00100000);
    *row2 = _mm256_permute2f128_ps(u0, u1, 0b00110001);
    *row3 = _mm256_permute2f128_ps(u2, u3, 0b00110001);
}

static void _mm256_transpose4_lane8_ps(__m256 *row0, __m256 *row1, __m256 *row2, __m256 *row3) {
    __m256 t0 = _mm256_permute2f128_ps(*row0, *row2, 0b00100000);
    __m256 t1 = _mm256_permute2f128_ps(*row1, *row3, 0b00100000);
    __m256 t2 = _mm256_permute2f128_ps(*row0, *row2, 0b00110001);
    __m256 t3 = _mm256_permute2f128_ps(*row1, *row3, 0b00110001);
    __m256 u0 = _mm256_unpacklo_ps(t0, t1);
    __m256 u1 = _mm256_unpackhi_ps(t0, t1);
    __m256 u2 = _mm256_unpacklo_ps(t2, t3);
    __m256 u3 = _mm256_unpackhi_ps(t2, t3);
    *row0 = _mm256_unpacklo_ps(u0, u2);
    *row1 = _mm256_unpackhi_ps(u0, u2);
    *row2 = _mm256_unpacklo_ps(u1, u3);
    *row3 = _mm256_unpackhi_ps(u1, u3);
}

static void transpose_block_into_buf(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w
) {
    for (int x = 0; x < src_w; x += 8) {
        __m256 line_0 = _mm256_load_ps(srcp + stride * 0);
        __m256 line_1 = _mm256_load_ps(srcp + stride * 1);
        __m256 line_2 = _mm256_load_ps(srcp + stride * 2);
        __m256 line_3 = _mm256_load_ps(srcp + stride * 3);
        _mm256_transpose8_lane4_ps(&line_0, &line_1, &line_2, &line_3);
        _mm256_store_ps(dstp + 0, line_0);
        _mm256_store_ps(dstp + 8, line_1);
        _mm256_store_ps(dstp + 16, line_2);
        _mm256_store_ps(dstp + 24, line_3);
        srcp += 8;
        dstp += 32;
    }
}

static void transpose_block_into_buf_with_tail(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int tail
) {
    for (int x = 0; x < src_w; x += 8) {
        __m256 line_0 = _mm256_load_ps(srcp + stride * 0);
        __m256 line_1 = (tail > 1) ? _mm256_load_ps(srcp + stride * 1) : _mm256_setzero_ps();
        __m256 line_2 = (tail > 2) ? _mm256_load_ps(srcp + stride * 2) : _mm256_setzero_ps();
        __m256 line_3 = _mm256_setzero_ps();
        _mm256_transpose8_lane4_ps(&line_0, &line_1, &line_2, &line_3);
        _mm256_store_ps(dstp + 0, line_0);
        _mm256_store_ps(dstp + 8, line_1);
        _mm256_store_ps(dstp + 16, line_2);
        _mm256_store_ps(dstp + 24, line_3);
        srcp += 8;
        dstp += 32;
    }
}

static void transpose_block_from_buf(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int dst_w
) {
    for (int x = 0; x < dst_w; x += 8) {
        __m256 line_0 = _mm256_load_ps(srcp + 0);
        __m256 line_1 = _mm256_load_ps(srcp + 8);
        __m256 line_2 = _mm256_load_ps(srcp + 16);
        __m256 line_3 = _mm256_load_ps(srcp + 24);
        _mm256_transpose4_lane8_ps(&line_0, &line_1, &line_2, &line_3);
        _mm256_stream_ps(dstp + stride * 0, line_0);
        _mm256_stream_ps(dstp + stride * 1, line_1);
        _mm256_stream_ps(dstp + stride * 2, line_2);
        _mm256_stream_ps(dstp + stride * 3, line_3);
        srcp += 32;
        dstp += 8;
    }
}

static void transpose_block_from_buf_with_tail(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int dst_w, int tail
) {
    for (int x = 0; x < dst_w; x += 8) {
        __m256 line_0 = _mm256_load_ps(srcp + 0);
        __m256 line_1 = _mm256_load_ps(srcp + 8);
        __m256 line_2 = _mm256_load_ps(srcp + 16);
        __m256 line_3 = _mm256_load_ps(srcp + 24);
        _mm256_transpose4_lane8_ps(&line_0, &line_1, &line_2, &line_3);
        _mm256_stream_ps(dstp + stride * 0, line_0);
        if (tail > 1) _mm256_stream_ps(dstp + stride * 1, line_1);
        if (tail > 2) _mm256_stream_ps(dstp + stride * 2, line_2);
        srcp += 32;
        dstp += 8;
    }
}

static void resize_width(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int dst_w, double start_w, double real_w, kernel_t kernel
) {
    double factor = dst_w / real_w;
    double scale = (factor < 1.0) ? factor : 1.0;
    int min_w = (int)floor(start_w);
    int max_w = (int)ceil(real_w + start_w) - 1;
    int border = src_w - 1;
    double radius = kernel.radius / scale;
    int step = (int)ceil(radius * 2.0) + 2;
    double *weights = (double *)malloc(sizeof(double) * step * dst_w);
    int64_t *col_idx = (int64_t *)malloc(sizeof(int64_t) * step * dst_w);
    int *row_ptr = (int *)malloc(sizeof(int) * (dst_w + 1));
    row_ptr[0] = 0;
    int nnz = 0;
    
    for (int x = 0; x < dst_w; x++) {
        double center = (x + 0.5) / factor - 0.5 + start_w;
        int low = VSMAX((int)floor(center - radius), min_w);
        int high = VSMIN((int)ceil(center + radius), max_w);
        double norm = 0.0;
        for (int i = low; i <= high; i++) {
            double temp_val = kernel.f((i - center) * scale, kernel.ctx);
            if (temp_val == 0.0) continue;
            norm += temp_val;
            int64_t temp_idx = CLAMP(i, 0, border) * 4;
            if (row_ptr[x] != nnz && temp_idx == col_idx[nnz - 1]) {
                weights[nnz - 1] += temp_val;
                continue;
            }
            weights[nnz] = temp_val;
            col_idx[nnz] = temp_idx;
            nnz++;
        }
        for (int i = row_ptr[x]; i < nnz; i++) {
            weights[i] /= norm;
        }
        row_ptr[x + 1] = nnz;
    }
    
    float *restrict src_buf = (float *)_mm_malloc(sizeof(float) * src_stride * 4, 64);
    float *restrict dst_buf = (float *)_mm_malloc(sizeof(float) * dst_stride * 4, 64);
    
    int tail = src_h % 4;
    int mod4_h = src_h - tail;
    
    for (int y = 0; y < mod4_h; y += 4) {
        transpose_block_into_buf(srcp, src_buf, src_stride, src_w);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = row_ptr[x]; i < row_ptr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + col_idx[i]);
                __m256d v_weight = _mm256_set1_pd(weights[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm_store_ps(dst_buf + x * 4, _mm256_cvtpd_ps(v_acc));
        }
        transpose_block_from_buf(dst_buf, dstp, dst_stride, dst_w);
        dstp += dst_stride * 4;
        srcp += src_stride * 4;
    }
    if (tail) {
        transpose_block_into_buf_with_tail(srcp, src_buf, src_stride, src_w, tail);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = row_ptr[x]; i < row_ptr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + col_idx[i]);
                __m256d v_weight = _mm256_set1_pd(weights[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm_store_ps(dst_buf + x * 4, _mm256_cvtpd_ps(v_acc));
        }
        transpose_block_from_buf_with_tail(dst_buf, dstp, dst_stride, dst_w, tail);
    }
    _mm_sfence();
    _mm_free(dst_buf);
    _mm_free(src_buf);
    free(row_ptr);
    free(col_idx);
    free(weights);
}

static void resize_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t dst_stride,
    int src_w, int src_h, int dst_h, double start_h, double real_h, kernel_t kernel
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    double factor = dst_h / real_h;
    double scale = (factor < 1.0) ? factor : 1.0;
    int min_h = (int)floor(start_h);
    int max_h = (int)ceil(real_h + start_h) - 1;
    int border = src_h - 1;
    double radius = kernel.radius / scale;
    int step = (int)ceil(radius * 2.0) + 2;
    double *weights = (double *)malloc(sizeof(double) * step);
    int64_t *col_idx = (int64_t *)malloc(sizeof(int64_t) * step);
    
    for (int y = 0; y < dst_h; y++) {
        double center = (y + 0.5) / factor - 0.5 + start_h;
        int low = VSMAX((int)floor(center - radius), min_h);
        int high = VSMIN((int)ceil(center + radius), max_h);
        int nnz = 0;
        double norm = 0.0;
        for (int i = low; i <= high; i++) {
            double temp_val = kernel.f((i - center) * scale, kernel.ctx);
            if (temp_val == 0.0) continue;
            norm += temp_val;
            int64_t temp_idx = CLAMP(i, 0, border) * dst_stride;
            if (nnz && temp_idx == col_idx[nnz - 1]) {
                weights[nnz - 1] += temp_val;
                continue;
            }
            weights[nnz] = temp_val;
            col_idx[nnz] = temp_idx;
            nnz++;
        }
        for (int i = 0; i < nnz; i++) {
            weights[i] /= norm;
        }
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = 0; i < nnz; i++) {
                __m256 pix = _mm256_load_ps(srcp + col_idx[i] + x);
                __m256d v_weight = _mm256_set1_pd(weights[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_stream_ps(dstp + x, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        if (tail) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = 0; i < nnz; i++) {
                __m256 pix = _mm256_maskload_ps(srcp + col_idx[i] + x, tail_mask);
                __m256d v_weight = _mm256_set1_pd(weights[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_stream_ps(dstp + x, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        dstp += dst_stride;
    }
    _mm_sfence();
    free(col_idx);
    free(weights);
}

#else

static void to_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, GammaData d
) {
    if (d.strict) {
        for (int y = 0; y < src_h; y++) {
            for (int x = 0; x < src_w; x++) {
                float pix = fabsf(srcp[x]);
                float dst;
                if (pix > d.thr_to) {
                    dst = powf((pix + d.corr) / (d.corr + 1.0f), d.gamma);
                }
                else {
                    dst = pix / d.div;
                }
                dstp[x] = copysignf(dst, srcp[x]);
            }
            srcp += stride;
            dstp += stride;
        }
    }
    else {
        for (int y = 0; y < src_h; y++) {
            for (int x = 0; x < src_w; x++) {
                float pix = fabsf(srcp[x]);
                float dst;
                if (pix >= d.thr_to) {
                    dst = powf((pix + d.corr) / (d.corr + 1.0f), d.gamma);
                }
                else {
                    dst = pix / d.div;
                }
                dstp[x] = copysignf(dst, srcp[x]);
            }
            srcp += stride;
            dstp += stride;
        }
    }
}

static void from_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, GammaData d
) {
    if (d.strict) {
        for (int y = 0; y < src_h; y++) {
            for (int x = 0; x < src_w; x++) {
                float pix = fabsf(srcp[x]);
                float dst;
                if (pix > d.thr_from) {
                    dst = powf(pix, 1.0f / d.gamma) * (d.corr + 1.0f) - d.corr;
                }
                else {
                    dst = pix * d.div;
                }
                dstp[x] = copysignf(dst, srcp[x]);
            }
            srcp += stride;
            dstp += stride;
        }
    }
    else {
        for (int y = 0; y < src_h; y++) {
            for (int x = 0; x < src_w; x++) {
                float pix = fabsf(srcp[x]);
                float dst;
                if (pix >= d.thr_from) {
                    dst = powf(pix, 1.0f / d.gamma) * (d.corr + 1.0f) - d.corr;
                }
                else {
                    dst = pix * d.div;
                }
                dstp[x] = copysignf(dst, srcp[x]);
            }
            srcp += stride;
            dstp += stride;
        }
    }
}

static void uint8_to_uint16(
    const uint8_t *restrict srcp, uint16_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int bits
) {
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = (int)srcp[x] << (bits - 8);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}

static void uint8_to_float(
    const uint8_t *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range
) {
    float low, high;
    if (range) {
        low = (chroma ? 128.0f : 16.0f);
        high = (chroma ? 224.0f : 219.0f);
    }
    else {
        low = (chroma ? 128.0f : 0.0f);
        high = (chroma ? 256.0f : 255.0f);
    }
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = ((float)srcp[x] - low) / high;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}

static void uint16_to_uint8(
    const uint16_t *restrict srcp, uint8_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int bits
) {
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = fminf((float)srcp[x] / (1 << (bits - 8)) + 0.5f, 255.0f);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}

static void uint16_to_uint16(
    const uint16_t *restrict srcp, uint16_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits
) {
    if (src_bits < dst_bits) {
        for (int y = 0; y < src_h; y++) {
            for (int x = 0; x < src_w; x++) {
                dstp[x] = (int)srcp[x] << (dst_bits - src_bits);
            }
            srcp += src_stride;
            dstp += dst_stride;
        }
    }
    else {
        float full = (1 << dst_bits) - 1;
        for (int y = 0; y < src_h; y++) {
            for (int x = 0; x < src_w; x++) {
                dstp[x] = fminf((float)srcp[x] / (1 << (src_bits - dst_bits)) + 0.5f, full);
            }
            srcp += src_stride;
            dstp += dst_stride;
        }
    }
}

static void uint16_to_float(
    const uint16_t *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range, int bits
) {
    float low, high;
    if (range) {
        low = (chroma ? 128 : 16) << (bits - 8);
        high = (chroma ? 224 : 219) << (bits - 8);
    }
    else {
        low = (chroma ? 128 << (bits - 8) : 0);
        high = (chroma ? 1 << bits : (1 << bits) - 1);
    }
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = ((float)srcp[x] - low) / high;
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}

static void float_to_uint8(
    const float *restrict srcp, uint8_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range
) {
    float low, high;
    if (range) {
        low = (chroma ? 128.0f : 16.0f);
        high = (chroma ? 224.0f : 219.0f);
    }
    else {
        low = (chroma ? 128.0f : 0.0f);
        high = (chroma ? 256.0f : 255.0f);
    }
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = fmaxf(fminf(srcp[x] * high + low + 0.5f, 255.0f), 0.0f);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}

static void float_to_uint16(
    const float *restrict srcp, uint16_t *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, bool chroma, bool range, int bits
) {
    float full = (1 << bits) - 1;
    float low, high;
    if (range) {
        low = (chroma ? 128 : 16) << (bits - 8);
        high = (chroma ? 224 : 219) << (bits - 8);
    }
    else {
        low = (chroma ? 128 << (bits - 8) : 0);
        high = (chroma ? 1 << bits : full);
    }
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = fmaxf(fminf(srcp[x] * high + low + 0.5f, full), 0.0f);
        }
        srcp += src_stride;
        dstp += dst_stride;
    }
}

static void sharp_width(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float sharp
) {
    int border = src_w - 1;
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = srcp[x] * sharp + (1.0f - sharp) * (srcp[VSMAX(x - 1, 0)] + srcp[x] + srcp[VSMIN(x + 1, border)]) / 3.0f;
        }
        dstp += stride;
        srcp += stride;
    }
}

static void sharp_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float sharp
) {
    int border = src_h - 1;
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = srcp[x] * sharp + (1.0f - sharp) * (srcp[(y > 0) ? (x - stride) : x] + srcp[x] + srcp[(y < border) ? (x + stride) : x]) / 3.0f;
        }
        dstp += stride;
        srcp += stride;
    }
}

static void resize_width(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int dst_w, double start_w, double real_w, kernel_t kernel
) {
    double factor = dst_w / real_w;
    double scale = (factor < 1.0) ? factor : 1.0;
    int min_w = (int)floor(start_w);
    int max_w = (int)ceil(real_w + start_w) - 1;
    int border = src_w - 1;
    double radius = kernel.radius / scale;
	int step = (int)ceil(radius * 2.0) + 2;
    int64_t *counts = (int64_t *)malloc(sizeof(int64_t) * dst_w * step);
    double *weights = (double *)malloc(sizeof(double) * dst_w * step);
    int *lengths = (int *)malloc(sizeof(int) * dst_w);
    
    for (int x = 0; x < dst_w; x++) {
        double center = (x + 0.5) / factor - 0.5 + start_w;
        int low = VSMAX((int)floor(center - radius), min_w);
        int high = VSMIN((int)ceil(center + radius), max_w);
        lengths[x] = high - low + 1;
        double norm = 0.0;
        for (int i = 0; i < lengths[x]; i++) {
            counts[x * step + i] = CLAMP(i + low, 0, border);
            weights[x * step + i] = kernel.f((i + low - center) * scale, kernel.ctx);
            norm += weights[x * step + i];
        }
        for (int i = 0; i < lengths[x]; i++) {
            weights[x * step + i] /= norm;
        }
    }
    
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            double acc = 0.0;
            for (int i = 0; i < lengths[x]; i++) {
                acc += (double)srcp[counts[x * step + i]] * weights[x * step + i];
            }
            dstp[x] = (float)acc;
        }
        dstp += dst_stride;
        srcp += src_stride;
    }
    free(counts);
    free(weights);
    free(lengths);
}

static void resize_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t dst_stride,
    int src_w, int src_h, int dst_h, double start_h, double real_h, kernel_t kernel
) {
    double factor = dst_h / real_h;
    double scale = (factor < 1.0) ? factor : 1.0;
    int min_h = (int)floor(start_h);
    int max_h = (int)ceil(real_h + start_h) - 1;
    int border = src_h - 1;
    double radius = kernel.radius / scale;
	int step = (int)ceil(radius * 2.0) + 2;
    int64_t *counts = (int64_t *)malloc(sizeof(int64_t) * step);
    double *weights = (double *)malloc(sizeof(double) * step);
    
    for (int y = 0; y < dst_h; y++) {
        double center = (y + 0.5) / factor - 0.5 + start_h;
        int low = VSMAX((int)floor(center - radius), min_h);
        int high = VSMIN((int)ceil(center + radius), max_h);
        int length = high - low + 1;
        double norm = 0.0;
        for (int i = 0; i < length; i++) {
            counts[i] = CLAMP(i + low, 0, border) * dst_stride;
            weights[i] = kernel.f((i + low - center) * scale, kernel.ctx);
            norm += weights[i];
        }
        for (int i = 0; i < length; i++) {
            weights[i] /= norm;
        }
        for (int x = 0; x < src_w; x++) {
            double acc = 0.0;
            for (int i = 0; i < length; i++) {
                acc += (double)srcp[counts[i] + x] * weights[i];
            }
            dstp[x] = (float)acc;
        }
        dstp += dst_stride;
    }
	free(counts);
    free(weights);
}

#endif

static const VSFrame *VS_CC ResizeGetFrame(
    int n, int activationReason, void *instanceData, void **frameData UNUSED,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    ResizeData *d = (ResizeData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        const VSMap *props = vsapi->getFramePropertiesRO(src);
        bool bit_convert = fi->sampleType == stInteger;
        
        int err;
        int chromaloc = vsapi->mapGetIntSaturated(props, "_ChromaLocation", 0, &err);
        if (err || chromaloc < 0 || chromaloc > 5) {
            chromaloc = 0;
        }
        
        int range = vsapi->mapGetIntSaturated(props, "_ColorRange", 0, &err);
        if (err || range < 0 || range > 1) {
            range = (fi->colorFamily == cfRGB) ? 0 : 1;
        }
        
        VSFrame *bcu = NULL;
        VSFrame *bcd = NULL;
        if (bit_convert) {
            bcu = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, NULL, core);
            bcd = vsapi->newVideoFrame(&d->vi.format, d->dst_width, d->dst_height, NULL, core);
        }
        
        VSFrame *lin = NULL;
        VSFrame *gcr = NULL;
        if (d->linear) {
            lin = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, NULL, core);
            gcr = vsapi->newVideoFrame(&d->vi.format, d->dst_width, d->dst_height, NULL, core);
        }
        
        VSFrame *tmp = NULL;
        if (d->process_w && d->process_h) {
            tmp = vsapi->newVideoFrame(&d->vi.format, d->dst_width, d->vi.height, NULL, core);
        }
        
        VSFrame *shr = NULL;
        if (d->sharp != 1.0f) {
            shr = vsapi->newVideoFrame(&d->vi.format, d->dst_width, d->dst_height, NULL, core);
        }
        
        VSFrame *dst = vsapi->newVideoFrame(fi, d->dst_width, d->dst_height, src, core);
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const void *restrict srcp = (const void *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / fi->bytesPerSample;
            
            void *restrict dstp = NULL;
            ptrdiff_t dst_stride = 0;
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            int dst_w = vsapi->getFrameWidth(dst, plane);
            int dst_h = vsapi->getFrameHeight(dst, plane);
            
            double start_w = d->start_w;
            double start_h = d->start_h;
            double real_w = d->real_w;
            double real_h = d->real_h;
            bool chroma = plane && (fi->colorFamily == cfYUV);
            
            if (chroma) {
                if (fi->subSamplingW) {
                    start_w /= 1 << fi->subSamplingW;
                    real_w /= 1 << fi->subSamplingW;
                    if (~chromaloc & 1) {// left allign
                        double offset = ((1 << fi->subSamplingW) - 1) / 2.0;
                        start_w += offset / (1 << fi->subSamplingW) - offset * real_w / (dst_w << fi->subSamplingW);
                    }
                }
                if (fi->subSamplingH) {
                    start_h /= 1 << fi->subSamplingH;
                    real_h /= 1 << fi->subSamplingH;
                    if (chromaloc & 2) {// top allign
                        double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                        start_h += offset / (1 << fi->subSamplingH) - offset * real_h / (dst_h << fi->subSamplingH);
                    }
                    else if (chromaloc & 4) {// bottom allign
                        double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                        start_h -= offset / (1 << fi->subSamplingH) - offset * real_h / (dst_h << fi->subSamplingH);
                    }
                }
            }
            
            if (bit_convert) {
                float *restrict bcup = (float *)vsapi->getWritePtr(bcu, plane);
                ptrdiff_t bcu_stride = vsapi->getStride(bcu, plane) / sizeof(float);
                if (fi->bytesPerSample == 1) {
                    uint8_to_float(srcp, bcup, src_stride, bcu_stride, src_w, src_h, chroma, range);
                }
                else {
                    uint16_to_float(srcp, bcup, src_stride, bcu_stride, src_w, src_h, chroma, range, fi->bitsPerSample);
                }
                srcp = bcup;
                src_stride = bcu_stride;
                dstp = (void *)vsapi->getWritePtr(bcd, plane);
                dst_stride = vsapi->getStride(bcd, plane) / sizeof(float);
            }
            else {
                dstp = (void *)vsapi->getWritePtr(dst, plane);
                dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
            }
            
            if (d->linear) {
                float *restrict linp = (float *)vsapi->getWritePtr(lin, plane);
                to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                srcp = linp;
                dstp = (void *)vsapi->getWritePtr(gcr, plane);
            }
            
            if (d->process_w && d->process_h) {
                float *restrict tmpp = (float *)vsapi->getWritePtr(tmp, plane);
                resize_width(srcp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, start_w, real_w, d->kernel_w);
                resize_height(tmpp, dstp, dst_stride, dst_w, src_h, dst_h, start_h, real_h, d->kernel_h);
            }
            else if (d->process_w) {
                resize_width(srcp, dstp, src_stride, dst_stride, src_w, src_h, dst_w, start_w, real_w, d->kernel_w);
            }
            else if (d->process_h) {
                resize_height(srcp, dstp, dst_stride, dst_w, src_h, dst_h, start_h, real_h, d->kernel_h);
            }
            else {
                vsh_bitblt(dstp, sizeof(float) * dst_stride, srcp, sizeof(float) * src_stride, sizeof(float) * src_w, src_h);
            }
            
            if (d->sharp != 1.0f) {
                float *restrict shrp = (float *)vsapi->getWritePtr(shr, plane);
                sharp_width(dstp, shrp, dst_stride, dst_w, dst_h, d->sharp);
                sharp_height(shrp, dstp, dst_stride, dst_w, dst_h, d->sharp);
            }
            
            if (d->linear) {
                float *restrict gcrp = dstp;
                dstp = bit_convert ? (void *)vsapi->getWritePtr(bcd, plane) : (void *)vsapi->getWritePtr(dst, plane);
                from_linear(gcrp, dstp, dst_stride, dst_w, dst_h, d->gamma);
            }
            
            if (bit_convert) {
                float *restrict bcdp = dstp;
                dstp = (void *)vsapi->getWritePtr(dst, plane);
                ptrdiff_t bcd_stride = dst_stride;
                dst_stride = vsapi->getStride(dst, plane) / fi->bytesPerSample;
                if (fi->bytesPerSample == 1) {
                    float_to_uint8(bcdp, dstp, bcd_stride, dst_stride, dst_w, dst_h, chroma, range);
                }
                else {
                    float_to_uint16(bcdp, dstp, bcd_stride, dst_stride, dst_w, dst_h, chroma, range, fi->bitsPerSample);
                }
            }
        }
        vsapi->freeFrame(shr);
        vsapi->freeFrame(tmp);
        vsapi->freeFrame(gcr);
        vsapi->freeFrame(lin);
        vsapi->freeFrame(bcd);
        vsapi->freeFrame(bcu);
        vsapi->freeFrame(src);
        
        return dst;
    }
    return NULL;
}

static void VS_CC ResizeFree(void *instanceData, VSCore *core UNUSED, const VSAPI *vsapi) {
    ResizeData *d = (ResizeData *)instanceData;
    vsapi->freeNode(d->node);
    
    if (d->kernel_w.ctx == d->kernel_h.ctx) {
        free(d->kernel_w.ctx);
    }
    else {
        free(d->kernel_w.ctx);
        free(d->kernel_h.ctx);
    }
    
    free(d);
}

static void VS_CC ResizeCreate(const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi) {
    ResizeData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (
        !vsh_isConstantVideoFormat(&d.vi) ||
        (d.vi.format.sampleType == stInteger && (d.vi.format.bitsPerSample < 8 || d.vi.format.bitsPerSample > 16)) ||
        (d.vi.format.sampleType == stFloat && d.vi.format.bitsPerSample != 32)
    ) {
        vsapi->mapSetError(out, "Resize: only constant format 8-16bit integer or 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.dst_width = vsapi->mapGetIntSaturated(in, "width", 0, NULL);
    d.dst_height = vsapi->mapGetIntSaturated(in, "height", 0, NULL);
    
    if (d.dst_width <= 1 << d.vi.format.subSamplingW || d.dst_width > 65535) {
        vsapi->mapSetError(out, "Resize: \"width\" any of the planes must be greater than 1 and less than or equal to 65535");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_width % (1 << d.vi.format.subSamplingW)) {
        vsapi->mapSetError(out, "Resize: \"width\" must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > 65535) {
        vsapi->mapSetError(out, "Resize: \"height\" any of the planes must be greater than 1 and less than or equal to 65535");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height % (1 << d.vi.format.subSamplingH)) {
        vsapi->mapSetError(out, "Resize: \"height\" must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.start_w = vsapi->mapGetFloat(in, "src_left", 0, &err);
    if (err) {
        d.start_w = 0.0;
    }
    
    if (d.start_w <= -d.vi.width || d.start_w >= d.vi.width) {
        vsapi->mapSetError(out, "Resize: \"src_left\" must be between \"-clip.width\" and \"clip.width\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.start_h = vsapi->mapGetFloat(in, "src_top", 0, &err);
    if (err) {
        d.start_h = 0.0;
    }
    
    if (d.start_h <= -d.vi.height || d.start_h >= d.vi.height) {
        vsapi->mapSetError(out, "Resize: \"src_top\" must be between \"-clip.height\" and \"clip.height\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.real_w = vsapi->mapGetFloat(in, "src_width", 0, &err);
    if (err) {
        d.real_w = (double)d.vi.width;
    }
    
    if (d.real_w <= -d.vi.width + d.start_w || d.real_w >= d.vi.width * 2 - d.start_w) {
        vsapi->mapSetError(out, "Resize: \"src_width\" must be between \"-clip.width + src_left\" and \"clip.width * 2 - src_left\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.real_w <= 0.0) {
        d.real_w += d.vi.width - d.start_w;
    }
    
    d.real_h = vsapi->mapGetFloat(in, "src_height", 0, &err);
    if (err) {
        d.real_h = (double)d.vi.height;
    }
    
    if (d.real_h <= -d.vi.height + d.start_h || d.real_h >= d.vi.height * 2 - d.start_h) {
        vsapi->mapSetError(out, "Resize: \"src_height\" must be between \"-clip.height + src_top\" and \"clip.height * 2 - src_top\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.real_h <= 0.0) {
        d.real_h += d.vi.height - d.start_h;
    }
    
    d.linear = true;
    
    const char *gamma = vsapi->mapGetData(in, "gamma", 0, &err);
    if (err) {
        if (d.vi.format.colorFamily == cfRGB) {
            d.gamma = (GammaData){2.4f, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
        }
        else {
            d.gamma = (GammaData){1.0f / 0.45f, 0.081f, 0.018f, 0.099f, 4.5f, false};
        }
    }
    else if (!strcmp(gamma, "srgb")) {
        d.gamma = (GammaData){2.4f, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
    }
    else if (!strcmp(gamma, "smpte170m")) {
        d.gamma = (GammaData){1.0f / 0.45f, 0.081f, 0.018f, 0.099f, 4.5f, false};
    }
    else if (!strcmp(gamma, "adobe")) {
        d.gamma = (GammaData){2.19921875f, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "dcip3")) {
        d.gamma = (GammaData){2.6f, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "smpte240m")) {
        d.gamma = (GammaData){1.0f / 0.45f, 0.0913f, 0.0228f, 0.1115f, 4.0f, false};
    }
    else if (!strcmp(gamma, "none")) {
        d.linear = false;
    }
    else {
        vsapi->mapSetError(out, "Resize: invalid gamma specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.sharp = vsapi->mapGetFloatSaturated(in, "sharp", 0, &err);
    if (err) {
        d.sharp = 1.0f;
    }
    
    if (d.sharp < 0.1f || d.sharp > 5.0f) {
        vsapi->mapSetError(out, "Resize: sharp must be between 0.1 and 5");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.sharp != 1.0f && ((d.dst_width >> d.vi.format.subSamplingW) <= 8 || (d.dst_height >> d.vi.format.subSamplingH) <= 1)) {
        vsapi->mapSetError(out, "Resize: when using a sharp, the width must be greater than 8 and the height must be greater than 1");
        vsapi->freeNode(d.node);
        return;
    }
    
    const char *kernel = vsapi->mapGetData(in, "kernel", 0, &err);
    if (err || !strcmp(kernel, "area")) {
        area_ctx *ar_w = (area_ctx *)malloc(sizeof(*ar_w));
        area_ctx *ar_h = (area_ctx *)malloc(sizeof(*ar_h));
        ar_w->scale = (d.dst_width < d.real_w) ? (d.dst_width / d.real_w) : (d.real_w / d.dst_width);
        ar_h->scale = (d.dst_height < d.real_h) ? (d.dst_height / d.real_h) : (d.real_h / d.dst_height);
        d.kernel_w = (kernel_t){area_kernel, 0.5 + ar_w->scale / 2.0, ar_w};
        d.kernel_h = (kernel_t){area_kernel, 0.5 + ar_h->scale / 2.0, ar_h};
    }
    else if (!strcmp(kernel, "magic")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel, 1.5, NULL};
    }
    else if (!strcmp(kernel, "magic13")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel_2013, 2.5, NULL};
    }
    else if (!strcmp(kernel, "magic21")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel_2021, 4.5, NULL};
    }
    else if (!strcmp(kernel, "bilinear")) {
        d.kernel_w = d.kernel_h = (kernel_t){bilinear_kernel, 1.0, NULL};
    }
    else if (!strcmp(kernel, "bicubic")) {
        bicubic_ctx *bc = (bicubic_ctx *)malloc(sizeof(*bc));
        bc->b = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
            bc->b = 1.0 / 3.0;
        }
        bc->c = vsapi->mapGetFloat(in, "c", 0, &err);
        if (err) {
            bc->c = 1.0 / 3.0;
        }
        d.kernel_w = d.kernel_h = (kernel_t){bicubic_kernel, 2.0, bc};
    }
    else if (!strcmp(kernel, "lanczos")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3;
        }
        if (sn->taps < 1 || sn->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){lanczos_kernel, (double)sn->taps, sn};
    }
    else if (!strcmp(kernel, "spline16")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline16_kernel, 2.0, NULL};
    }
    else if (!strcmp(kernel, "spline36")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline36_kernel, 3.0, NULL};
    }
    else if (!strcmp(kernel, "spline64")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline64_kernel, 4.0, NULL};
    }
    else if (!strcmp(kernel, "spline100")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline100_kernel, 5.0, NULL};
    }
    else if (!strcmp(kernel, "spline144")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline144_kernel, 6.0, NULL};
    }
    else if (!strcmp(kernel, "point")) {
        d.kernel_w = d.kernel_h = (kernel_t){point_kernel, 0.5, NULL};
    }
    else if (!strcmp(kernel, "blackman")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3;
        }
        if (sn->taps < 1 || sn->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){blackman_kernel, (double)sn->taps, sn};
    }
    else if (!strcmp(kernel, "nuttall")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3;
        }
        if (sn->taps < 1 || sn->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){nuttall_kernel, (double)sn->taps, sn};
    }
    else if (!strcmp(kernel, "kaiser")) {
        kaiser_ctx *ks = (kaiser_ctx *)malloc(sizeof(*ks));
        ks->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            ks->taps = 3;
        }
        if (ks->taps < 1 || ks->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(ks);
            return;
        }
        ks->beta = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           ks->beta = 4.0;
        }
        if (ks->beta <= 0.0 || ks->beta > 32.0) {
            vsapi->mapSetError(out, "Resize: beta must be between 0 and 32");
            vsapi->freeNode(d.node);
            free(ks);
            return;
        }
        ks->i0_beta = bessel_i0(ks->beta);
        d.kernel_w = d.kernel_h = (kernel_t){kaiser_kernel, (double)ks->taps, ks};
    }
    else if (!strcmp(kernel, "gauss")) {
        gauss_ctx *gs = (gauss_ctx *)malloc(sizeof(*gs));
        gs->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            gs->taps = 3;
        }
        if (gs->taps < 1 || gs->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           gs->p = 30.0;
        }
        if (gs->p < 1.0 || gs->p > 100.0) {
            vsapi->mapSetError(out, "Resize: p must be between 1 and 100");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p /= 10.0;
        d.kernel_w = d.kernel_h = (kernel_t){gauss_kernel, (double)gs->taps, gs};
    }
    else {
        vsapi->mapSetError(out, "Resize: invalid kernel specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.process_w = (d.dst_width == d.vi.width && d.real_w == d.vi.width && d.start_w == 0.0) ? false : true;
    d.process_h = (d.dst_height == d.vi.height && d.real_h == d.vi.height && d.start_h == 0.0) ? false : true;
    
    ResizeData *data = (ResizeData *)malloc(sizeof d);
    *data = d;
    
    data->vi.format.bitsPerSample = 32;
    data->vi.format.bytesPerSample = 4;
    data->vi.format.sampleType = stFloat;
    
    d.vi.width = d.dst_width;
    d.vi.height = d.dst_height;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Resize", &d.vi, ResizeGetFrame, ResizeFree, fmParallel, deps, 1, data, core);
}

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int dst_width, dst_height;
    double start_w, start_h, real_w, real_h, lambda;
    kernel_t kernel_w, kernel_h;
    bool process_w, process_h;
} DescaleData;

static void transpose_csr(
    double *restrict values, int *restrict col_idx, int *restrict row_ptr,
    double *restrict values_tr, int *restrict col_idx_tr, int *restrict row_ptr_tr,
    int nnz, int x, int y
) {
    int *col_count = calloc(x, sizeof(int));
    for (int i = 0; i < nnz; i++) col_count[col_idx[i]]++;
    
    row_ptr_tr[0] = 0;
    for (int i = 0; i < x; i++) row_ptr_tr[i + 1] = row_ptr_tr[i] + col_count[i];
    
    int *next = malloc(sizeof(int) * x);
    memcpy(next, row_ptr_tr, sizeof(int) * x);
    
    for (int i = 0; i < y; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            int dst = next[col_idx[j]]++;
            col_idx_tr[dst] = i;
            values_tr[dst] = values[j];
        }
    }
    
    free(next);
    free(col_count);
}

static int get_ku_from_csr(int *restrict col_idx, int *restrict row_ptr, int y) {
    int ku = 0;
    
    for (int i = 0; i < y; i++) {
        int p0 = row_ptr[i];
        int p1 = row_ptr[i + 1];
        int c_min = col_idx[p0];
        int c_max = col_idx[p1 - 1];
        int w = c_max - c_min;
        if (w > ku) ku = w;
    }
    
    return ku;
}

static void packed_banded_gramian_from_csr(
    double *restrict dst, double *restrict values, int *restrict col_idx, int *restrict row_ptr, int x, int y, int ku
) {
    for (int i = 0; i < y; i++) {
        int p0 = row_ptr[i];
        int p1 = row_ptr[i + 1];
        
        for (int j = p0; j < p1; j++) {
            int cj = col_idx[j];
            double vj = values[j];
            
            for (int k = j; k < p1; k++) {
                int ck = col_idx[k];
                double vk = values[k];
                int col = ku + cj - ck;
                dst[col * x + ck] += vj * vk;
            }
        }
    }
}

static void packed_banded_cholesky_from_gramian(double *restrict srcp, int x, int ku) {
    for (int i = 0; i < x; i++) {
        int j_start = VSMAX(i - ku, 0);
        for (int j = j_start; j < i; j++) {
            double acc = 0.0;
            int k_start = VSMAX(j - ku, j_start);
            for (int k = k_start; k < j; k++) {
                int idx_jk = (ku + k - j) * x + j;
                int idx_ik = (ku + k - i) * x + i;
                acc += srcp[idx_jk] * srcp[idx_ik];
            }
            int idx_ij = (ku + j - i) * x + i;
            int idx_jj = ku * x + j;
            srcp[idx_ij] = (srcp[idx_ij] - acc) / srcp[idx_jj];
        }
        double acc = 0.0;
        for (int j = j_start; j < i; j++) {
            int idx_ik = (ku + j - i) * x + i;
            double lik = srcp[idx_ik];
            acc += lik * lik;
        }
        int idx_ii = ku * x + i;
        srcp[idx_ii] = sqrt(srcp[idx_ii] - acc);
    }
}

static void solve_packed_banded_cholesky_lane4(double *srcp, double *dstp, int n, int ku) {
    for (int i = 0; i < n; i++) {
        int start = VSMAX(0, i - ku);
        __m256d v_acc = _mm256_load_pd(dstp + i * 4);
        for (int j = start; j < i; j++) {
            int row_in_src = ku + j - i;
            if (row_in_src >= 0) {
                __m256d pix = _mm256_load_pd(dstp + j * 4);
                __m256d v_weight = _mm256_set1_pd(srcp[row_in_src * n + i]);
                v_acc = _mm256_fnmadd_pd(pix, v_weight, v_acc);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp[ku * n + i]);
        _mm256_store_pd(dstp + i * 4, _mm256_div_pd(v_acc, v_div));
    }
    for (int i = n - 1; i >= 0; i--) {
        int end = VSMIN(n - 1, i + ku);
        __m256d v_acc = _mm256_load_pd(dstp + i * 4);
        for (int j = i + 1; j <= end; j++) {
            int row_in_src = ku + i - j;
            if (row_in_src >= 0) {
                __m256d pix = _mm256_load_pd(dstp + j * 4);
                __m256d v_weight = _mm256_set1_pd(srcp[row_in_src * n + j]);
                v_acc = _mm256_fnmadd_pd(pix, v_weight, v_acc);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp[ku * n + i]);
        _mm256_store_pd(dstp + i * 4, _mm256_div_pd(v_acc, v_div));
    }
}

static void transpose_double_block_from_buf(
    const double *restrict srcp, float *restrict dstp, ptrdiff_t stride, int dst_w
) {
    for (int x = 0; x < dst_w; x += 8) {
        __m256 line_0 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 0)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 4))
        );
        __m256 line_1 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 8)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 12))
        );
        __m256 line_2 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 16)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 20))
        );
        __m256 line_3 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 24)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 28))
        );
        _mm256_transpose4_lane8_ps(&line_0, &line_1, &line_2, &line_3);
        _mm256_stream_ps(dstp + stride * 0, line_0);
        _mm256_stream_ps(dstp + stride * 1, line_1);
        _mm256_stream_ps(dstp + stride * 2, line_2);
        _mm256_stream_ps(dstp + stride * 3, line_3);
        srcp += 32;
        dstp += 8;
    }
}

static void transpose_double_block_from_buf_with_tail(
    const double *restrict srcp, float *restrict dstp, ptrdiff_t stride, int dst_w, int tail
) {
    for (int x = 0; x < dst_w; x += 8) {
        __m256 line_0 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 0)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 4))
        );
        __m256 line_1 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 8)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 12))
        );
        __m256 line_2 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 16)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 20))
        );
        __m256 line_3 = _mm256_setr_m128(
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 24)),
            _mm256_cvtpd_ps(_mm256_load_pd(srcp + 28))
        );
        _mm256_transpose4_lane8_ps(&line_0, &line_1, &line_2, &line_3);
        _mm256_stream_ps(dstp + stride * 0, line_0);
        if (tail > 1) _mm256_stream_ps(dstp + stride * 1, line_1);
        if (tail > 2) _mm256_stream_ps(dstp + stride * 2, line_2);
        srcp += 32;
        dstp += 8;
    }
}

static void descale_width(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int dst_w, double start_w, double real_w, double lambda, kernel_t kernel
) {
    double factor = src_w / real_w;
    int min_w = (int)floor(start_w);
    int max_w = (int)ceil(real_w + start_w) - 1;
    int border = dst_w - 1;
    int step = (int)ceil(kernel.radius * 2) + 2;
    double *weights = malloc(sizeof(double) * step * src_w);
    int *col_idx = malloc(sizeof(int) * step * src_w);
    int *row_ptr = malloc(sizeof(int) * (src_w + 1));
    row_ptr[0] = 0;
    int nnz = 0;
    
    for (int x = 0; x < src_w; x++) {
        double center = (x + 0.5) / factor - 0.5 + start_w;
        int low = VSMAX((int)floor(center - kernel.radius), min_w);
        int high = VSMIN((int)ceil(center + kernel.radius), max_w);
        double norm = 0.0;
        for (int i = low; i <= high; i++) {
            double temp_val = kernel.f(i - center, kernel.ctx);
            if (temp_val == 0.0) continue;
            norm += temp_val;
            int temp_idx = CLAMP(i, 0, border);
            if (row_ptr[x] != nnz && temp_idx == col_idx[nnz - 1]) {
                weights[nnz - 1] += temp_val;
                continue;
            }
            weights[nnz] = temp_val;
            col_idx[nnz] = temp_idx;
            nnz++;
        }
        for (int i = row_ptr[x]; i < nnz; i++) {
            weights[i] /= norm;
        }
        row_ptr[x + 1] = nnz;
    }
    
    double *weights_tr = malloc(sizeof(double) * nnz);
    int *col_idx_tr = malloc(sizeof(int) * nnz);
    int *row_ptr_tr = malloc(sizeof(int) * (dst_w + 1));
    transpose_csr(weights, col_idx, row_ptr, weights_tr, col_idx_tr, row_ptr_tr, nnz, dst_w, src_w);
    
    int ku = get_ku_from_csr(col_idx, row_ptr, src_w);
    int kt = ku + 1;
    
    double *matrix = (double *)calloc(dst_w * kt, sizeof(double));
    packed_banded_gramian_from_csr(matrix, weights, col_idx, row_ptr, dst_w, src_w, ku);
    
    for (int i = dst_w * ku; i < dst_w * kt; i++) matrix[i] += lambda;
    
    packed_banded_cholesky_from_gramian(matrix, dst_w, ku);
    
    float *restrict src_buf = (float *)_mm_malloc(sizeof(float) * src_stride * 4, 64);
    double *restrict dst_buf = (double *)_mm_malloc(sizeof(double) * dst_stride * 4, 64);
    
    int tail = src_h % 4;
    int mod4_h = src_h - tail;
    
    for (int y = 0; y < mod4_h; y += 4) {
        transpose_block_into_buf(srcp, src_buf, src_stride, src_w);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = row_ptr_tr[x]; i < row_ptr_tr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + col_idx_tr[i] * 4);
                __m256d v_weight = _mm256_set1_pd(weights_tr[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm256_store_pd(dst_buf + x * 4, v_acc);
        }
        solve_packed_banded_cholesky_lane4(matrix, dst_buf, dst_w, ku);
        transpose_double_block_from_buf(dst_buf, dstp, dst_stride, dst_w);
        dstp += dst_stride * 4;
        srcp += src_stride * 4;
    }
    if (tail) {
        transpose_block_into_buf_with_tail(srcp, src_buf, src_stride, src_w, tail);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = row_ptr_tr[x]; i < row_ptr_tr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + col_idx_tr[i] * 4);
                __m256d v_weight = _mm256_set1_pd(weights_tr[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm256_store_pd(dst_buf + x * 4, v_acc);
        }
        solve_packed_banded_cholesky_lane4(matrix, dst_buf, dst_w, ku);
        transpose_double_block_from_buf_with_tail(dst_buf, dstp, dst_stride, dst_w, tail);
    }
    _mm_sfence();
    _mm_free(dst_buf);
    _mm_free(src_buf);
    free(matrix);
    free(row_ptr_tr);
    free(col_idx_tr);
    free(weights_tr);
    free(row_ptr);
    free(col_idx);
    free(weights);
}

static void solve_packed_banded_cholesky_lane8(double *srcp, double *dstp, int n, int ku) {
    for (int i = 0; i < n; i++) {
        int start = VSMAX(0, i - ku);
        __m256d v_acc_0 = _mm256_load_pd(dstp + i * 8 + 0);
        __m256d v_acc_1 = _mm256_load_pd(dstp + i * 8 + 4);
        for (int j = start; j < i; j++) {
            int row_in_src = ku + j - i;
            if (row_in_src >= 0) {
                __m256d pix_0 = _mm256_load_pd(dstp + j * 8 + 0);
                __m256d pix_1 = _mm256_load_pd(dstp + j * 8 + 4);
                __m256d v_weight = _mm256_set1_pd(srcp[row_in_src * n + i]);
                v_acc_0 = _mm256_fnmadd_pd(pix_0, v_weight, v_acc_0);
                v_acc_1 = _mm256_fnmadd_pd(pix_1, v_weight, v_acc_1);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp[ku * n + i]);
        _mm256_store_pd(dstp + i * 8 + 0, _mm256_div_pd(v_acc_0, v_div));
        _mm256_store_pd(dstp + i * 8 + 4, _mm256_div_pd(v_acc_1, v_div));
    }
    for (int i = n - 1; i >= 0; i--) {
        int end = VSMIN(n - 1, i + ku);
        __m256d v_acc_0 = _mm256_load_pd(dstp + i * 8 + 0);
        __m256d v_acc_1 = _mm256_load_pd(dstp + i * 8 + 4);
        for (int j = i + 1; j <= end; j++) {
            int row_in_src = ku + i - j;
            if (row_in_src >= 0) {
                __m256d pix_0 = _mm256_load_pd(dstp + j * 8 + 0);
                __m256d pix_1 = _mm256_load_pd(dstp + j * 8 + 4);
                __m256d v_weight = _mm256_set1_pd(srcp[row_in_src * n + j]);
                v_acc_0 = _mm256_fnmadd_pd(pix_0, v_weight, v_acc_0);
                v_acc_1 = _mm256_fnmadd_pd(pix_1, v_weight, v_acc_1);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp[ku * n + i]);
        _mm256_store_pd(dstp + i * 8 + 0, _mm256_div_pd(v_acc_0, v_div));
        _mm256_store_pd(dstp + i * 8 + 4, _mm256_div_pd(v_acc_1, v_div));
    }
}

static void descale_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t src_stride,
    int src_w, int src_h, int dst_h, double start_h, double real_h, double lambda, kernel_t kernel
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    double factor = src_h / real_h;
    int min_h = (int)floor(start_h);
    int max_h = (int)ceil(real_h + start_h) - 1;
    int border = dst_h - 1;
    int step = (int)ceil(kernel.radius * 2) + 2;
    double *weights = malloc(sizeof(double) * step * src_h);
    int *col_idx = malloc(sizeof(int) * step * src_h);
    int *row_ptr = malloc(sizeof(int) * (src_h + 1));
    row_ptr[0] = 0;
    int nnz = 0;
    
    for (int y = 0; y < src_h; y++) {
        double center = (y + 0.5) / factor - 0.5 + start_h;
        int low = VSMAX((int)floor(center - kernel.radius), min_h);
        int high = VSMIN((int)ceil(center + kernel.radius), max_h);
        double norm = 0.0;
        for (int i = low; i <= high; i++) {
            double temp_val = kernel.f(i - center, kernel.ctx);
            if (temp_val == 0.0) continue;
            norm += temp_val;
            int temp_idx = CLAMP(i, 0, border);
            if (row_ptr[y] != nnz && temp_idx == col_idx[nnz - 1]) {
                weights[nnz - 1] += temp_val;
                continue;
            }
            weights[nnz] = temp_val;
            col_idx[nnz] = temp_idx;
            nnz++;
        }
        for (int i = row_ptr[y]; i < nnz; i++) {
            weights[i] /= norm;
        }
        row_ptr[y + 1] = nnz;
    }
    
    double *weights_tr = malloc(sizeof(double) * nnz);
    int *col_idx_tr = malloc(sizeof(int) * nnz);
    int *row_ptr_tr = malloc(sizeof(int) * (dst_h + 1));
    transpose_csr(weights, col_idx, row_ptr, weights_tr, col_idx_tr, row_ptr_tr, nnz, dst_h, src_h);
    
    int ku = get_ku_from_csr(col_idx, row_ptr, src_h);
    int kt = ku + 1;
    
    ptrdiff_t dst_stride = (dst_h + 7) / 8 * 8;
    
    double *matrix = (double *)calloc(dst_h * kt, sizeof(double));
    packed_banded_gramian_from_csr(matrix, weights, col_idx, row_ptr, dst_h, src_h, ku);
    
    for (int i = dst_h * ku; i < dst_h * kt; i++) matrix[i] += lambda;
    
    packed_banded_cholesky_from_gramian(matrix, dst_h, ku);
    
    double *restrict dst_buf = (double *)_mm_malloc(sizeof(double) * dst_stride * 8, 64);
    
    for (int x = 0; x < mod8_w; x += 8) {
        for (int y = 0; y < dst_h; y++) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = row_ptr_tr[y]; i < row_ptr_tr[y + 1]; i++) {
                __m256 pix = _mm256_load_ps(srcp + col_idx_tr[i] * src_stride);
                __m256d v_weight = _mm256_set1_pd(weights_tr[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_store_pd(dst_buf + y * 8 + 0, v_acc_0);
            _mm256_store_pd(dst_buf + y * 8 + 4, v_acc_1);
        }
        solve_packed_banded_cholesky_lane8(matrix, dst_buf, dst_h, ku);
        for (int y = 0; y < dst_h; y++) {
            __m128 pix0 = _mm256_cvtpd_ps(_mm256_load_pd(dst_buf + y * 8 + 0));
            __m128 pix1 = _mm256_cvtpd_ps(_mm256_load_pd(dst_buf + y * 8 + 4));
            _mm256_stream_ps(dstp + y * src_stride, _mm256_setr_m128(pix0, pix1));
        }
        srcp += 8;
        dstp += 8;
    }
    if (tail) {
        for (int y = 0; y < dst_h; y++) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = row_ptr_tr[y]; i < row_ptr_tr[y + 1]; i++) {
                __m256 pix = _mm256_maskload_ps(srcp + col_idx_tr[i] * src_stride, tail_mask);
                __m256d v_weight = _mm256_set1_pd(weights_tr[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_store_pd(dst_buf + y * 8 + 0, v_acc_0);
            _mm256_store_pd(dst_buf + y * 8 + 4, v_acc_1);
        }
        solve_packed_banded_cholesky_lane8(matrix, dst_buf, dst_h, ku);
        for (int y = 0; y < dst_h; y++) {
            __m128 pix0 = _mm256_cvtpd_ps(_mm256_load_pd(dst_buf + y * 8 + 0));
            __m128 pix1 = _mm256_cvtpd_ps(_mm256_load_pd(dst_buf + y * 8 + 4));
            _mm256_stream_ps(dstp + y * src_stride, _mm256_setr_m128(pix0, pix1));
        }
    }
    _mm_sfence();
    _mm_free(dst_buf);
    free(matrix);
    free(row_ptr_tr);
    free(col_idx_tr);
    free(weights_tr);
    free(row_ptr);
    free(col_idx);
    free(weights);
}

static const VSFrame *VS_CC DescaleGetFrame(
    int n, int activationReason, void *instanceData, void **frameData UNUSED,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    DescaleData *d = (DescaleData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        const VSMap *props = vsapi->getFramePropertiesRO(src);
        
        int err;
        int chromaloc = vsapi->mapGetIntSaturated(props, "_ChromaLocation", 0, &err);
        if (err || chromaloc < 0 || chromaloc > 5) {
            chromaloc = 0;
        }
        
        VSFrame *tmp = NULL;
        if (d->process_w && d->process_h) {
            tmp = vsapi->newVideoFrame(fi, d->dst_width, d->vi.height, NULL, core);
        }
        
        VSFrame *dst = vsapi->newVideoFrame(fi, d->dst_width, d->dst_height, src, core);
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float *restrict srcp = (const float *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            
            float *restrict dstp = (float *)vsapi->getWritePtr(dst, plane);
            ptrdiff_t dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            int dst_w = vsapi->getFrameWidth(dst, plane);
            int dst_h = vsapi->getFrameHeight(dst, plane);
            
            double start_w = d->start_w;
            double start_h = d->start_h;
            double real_w = d->real_w;
            double real_h = d->real_h;
            
            if (plane && (fi->colorFamily == cfYUV)) {
                if (fi->subSamplingW) {
                    start_w /= 1 << fi->subSamplingW;
                    real_w /= 1 << fi->subSamplingW;
                    if (~chromaloc & 1) {// left allign
                        double offset = ((1 << fi->subSamplingW) - 1) / 2.0;
                        start_w += offset / (1 << fi->subSamplingW) - offset * real_w / (src_w << fi->subSamplingW);
                    }
                }
                if (fi->subSamplingH) {
                    start_h /= 1 << fi->subSamplingH;
                    real_h /= 1 << fi->subSamplingH;
                    if (chromaloc & 2) {// top allign
                        double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                        start_h += offset / (1 << fi->subSamplingH) - offset * real_h / (src_h << fi->subSamplingH);
                    }
                    else if (chromaloc & 4) {// bottom allign
                        double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                        start_h -= offset / (1 << fi->subSamplingH) - offset * real_h / (src_h << fi->subSamplingH);
                    }
                }
            }
            
            if (d->process_w && d->process_h) {
                float *restrict tmpp = (float *)vsapi->getWritePtr(tmp, plane);
                descale_width(srcp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, start_w, real_w, d->lambda, d->kernel_w);
                descale_height(tmpp, dstp, dst_stride, dst_w, src_h, dst_h, start_h, real_h, d->lambda, d->kernel_h);
            }
            else if (d->process_w) {
                descale_width(srcp, dstp, src_stride, dst_stride, src_w, src_h, dst_w, start_w, real_w, d->lambda, d->kernel_w);
            }
            else if (d->process_h) {
                descale_height(srcp, dstp, dst_stride, dst_w, src_h, dst_h, start_h, real_h, d->lambda, d->kernel_h);
            }
            else {
                vsh_bitblt(dstp, sizeof(float) * dst_stride, srcp, sizeof(float) * src_stride, sizeof(float) * src_w, src_h);
            }
        }
        
        vsapi->freeFrame(tmp);
        vsapi->freeFrame(src);
        
        return dst;
    }
    return NULL;
}

static void VS_CC DescaleFree(void *instanceData, VSCore *core UNUSED, const VSAPI *vsapi) {
    DescaleData *d = (DescaleData *)instanceData;
    vsapi->freeNode(d->node);
    
    if (d->kernel_w.ctx == d->kernel_h.ctx) {
        free(d->kernel_w.ctx);
    }
    else {
        free(d->kernel_w.ctx);
        free(d->kernel_h.ctx);
    }
    
    free(d);
}

static void VS_CC DescaleCreate(const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi) {
    DescaleData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "Resize: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.dst_width = vsapi->mapGetIntSaturated(in, "width", 0, NULL);
    d.dst_height = vsapi->mapGetIntSaturated(in, "height", 0, NULL);
    
    if (d.dst_width <= 1 << d.vi.format.subSamplingW || d.dst_width > d.vi.width) {
        vsapi->mapSetError(out, "Resize: \"width\" any of the planes must be greater than 1 and less than or equal to source width");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_width % (1 << d.vi.format.subSamplingW)) {
        vsapi->mapSetError(out, "Resize: \"width\" must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > d.vi.height) {
        vsapi->mapSetError(out, "Resize: \"height\" any of the planes must be greater than 1 and less than or equal to source height");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height % (1 << d.vi.format.subSamplingH)) {
        vsapi->mapSetError(out, "Resize: \"height\" must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.start_w = vsapi->mapGetFloat(in, "src_left", 0, &err);
    if (err) {
        d.start_w = 0.0;
    }
    
    if (d.start_w <= -d.dst_width || d.start_w >= d.dst_width) {
        vsapi->mapSetError(out, "Resize: \"src_left\" must be between \"-width\" and \"width\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.start_h = vsapi->mapGetFloat(in, "src_top", 0, &err);
    if (err) {
        d.start_h = 0.0;
    }
    
    if (d.start_h <= -d.dst_height || d.start_h >= d.dst_height) {
        vsapi->mapSetError(out, "Resize: \"src_top\" must be between \"-height\" and \"height\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.real_w = vsapi->mapGetFloat(in, "src_width", 0, &err);
    if (err) {
        d.real_w = (double)d.dst_width;
    }
    
    if (d.real_w <= -d.dst_width + d.start_w || d.real_w >= d.dst_width * 2 - d.start_w) {
        vsapi->mapSetError(out, "Resize: \"src_width\" must be between \"-width + src_left\" and \"width * 2 - src_left\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.real_w <= 0.0) {
        d.real_w += d.dst_width - d.start_w;
    }
    
    d.real_h = vsapi->mapGetFloat(in, "src_height", 0, &err);
    if (err) {
        d.real_h = (double)d.dst_height;
    }
    
    if (d.real_h <= -d.dst_height + d.start_h || d.real_h >= d.dst_height * 2 - d.start_h) {
        vsapi->mapSetError(out, "Resize: \"src_height\" must be between \"-height + src_top\" and \"height * 2 - src_top\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.real_h <= 0.0) {
        d.real_h += d.dst_height - d.start_h;
    }
    
    d.lambda = vsapi->mapGetFloat(in, "lambda", 0, &err);
    if (err) {
        d.lambda = 1e-4;
    }
    
    if (d.lambda < 1e-12 || d.lambda >= 1.0) {
        vsapi->mapSetError(out, "Resize: \"lambda\" must be between 1e-12 and 1");
        vsapi->freeNode(d.node);
        return;
    }
    
    const char *kernel = vsapi->mapGetData(in, "kernel", 0, &err);
    if (err || !strcmp(kernel, "area")) {
        area_ctx *ar_w = (area_ctx *)malloc(sizeof(*ar_w));
        area_ctx *ar_h = (area_ctx *)malloc(sizeof(*ar_h));
        ar_w->scale = d.real_w / d.vi.width;
        ar_h->scale = d.real_h / d.vi.height;
        d.kernel_w = (kernel_t){area_kernel, 0.5 + ar_w->scale / 2.0, ar_w};
        d.kernel_h = (kernel_t){area_kernel, 0.5 + ar_h->scale / 2.0, ar_h};
    }
    else if (!strcmp(kernel, "magic")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel, 1.5, NULL};
    }
    else if (!strcmp(kernel, "magic13")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel_2013, 2.5, NULL};
    }
    else if (!strcmp(kernel, "magic21")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel_2021, 4.5, NULL};
    }
    else if (!strcmp(kernel, "bilinear")) {
        d.kernel_w = d.kernel_h = (kernel_t){bilinear_kernel, 1.0, NULL};
    }
    else if (!strcmp(kernel, "bicubic")) {
        bicubic_ctx *bc = (bicubic_ctx *)malloc(sizeof(*bc));
        bc->b = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
            bc->b = 1.0 / 3.0;
        }
        bc->c = vsapi->mapGetFloat(in, "c", 0, &err);
        if (err) {
            bc->c = 1.0 / 3.0;
        }
        d.kernel_w = d.kernel_h = (kernel_t){bicubic_kernel, 2.0, bc};
    }
    else if (!strcmp(kernel, "lanczos")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3;
        }
        if (sn->taps < 1 || sn->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){lanczos_kernel, (double)sn->taps, sn};
    }
    else if (!strcmp(kernel, "spline16")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline16_kernel, 2.0, NULL};
    }
    else if (!strcmp(kernel, "spline36")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline36_kernel, 3.0, NULL};
    }
    else if (!strcmp(kernel, "spline64")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline64_kernel, 4.0, NULL};
    }
    else if (!strcmp(kernel, "spline100")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline100_kernel, 5.0, NULL};
    }
    else if (!strcmp(kernel, "spline144")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline144_kernel, 6.0, NULL};
    }
    else if (!strcmp(kernel, "point")) {
        d.kernel_w = d.kernel_h = (kernel_t){point_kernel, 0.5, NULL};
    }
    else if (!strcmp(kernel, "blackman")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3;
        }
        if (sn->taps < 1 || sn->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){blackman_kernel, (double)sn->taps, sn};
    }
    else if (!strcmp(kernel, "nuttall")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3;
        }
        if (sn->taps < 1 || sn->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){nuttall_kernel, (double)sn->taps, sn};
    }
    else if (!strcmp(kernel, "kaiser")) {
        kaiser_ctx *ks = (kaiser_ctx *)malloc(sizeof(*ks));
        ks->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            ks->taps = 3;
        }
        if (ks->taps < 1 || ks->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(ks);
            return;
        }
        ks->beta = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           ks->beta = 4.0;
        }
        if (ks->beta <= 0.0 || ks->beta > 32.0) {
            vsapi->mapSetError(out, "Resize: beta must be between 0 and 32");
            vsapi->freeNode(d.node);
            free(ks);
            return;
        }
        ks->i0_beta = bessel_i0(ks->beta);
        d.kernel_w = d.kernel_h = (kernel_t){kaiser_kernel, (double)ks->taps, ks};
    }
    else if (!strcmp(kernel, "gauss")) {
        gauss_ctx *gs = (gauss_ctx *)malloc(sizeof(*gs));
        gs->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            gs->taps = 3;
        }
        if (gs->taps < 1 || gs->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           gs->p = 30.0;
        }
        if (gs->p < 1.0 || gs->p > 100.0) {
            vsapi->mapSetError(out, "Resize: p must be between 1 and 100");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p /= 10.0;
        d.kernel_w = d.kernel_h = (kernel_t){gauss_kernel, (double)gs->taps, gs};
    }
    else {
        vsapi->mapSetError(out, "Resize: invalid kernel specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.process_w = (d.dst_width == d.vi.width && d.real_w == d.dst_width && d.start_w == 0.0) ? false : true;
    d.process_h = (d.dst_height == d.vi.height && d.real_h == d.dst_height && d.start_h == 0.0) ? false : true;
    
    DescaleData *data = (DescaleData *)malloc(sizeof d);
    *data = d;
    
    d.vi.width = d.dst_width;
    d.vi.height = d.dst_height;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Descale", &d.vi, DescaleGetFrame, DescaleFree, fmParallel, deps, 1, data, core);
}

typedef struct {
    VSNode *node0;
    VSNode *node1;
} RelativeErrorData;

#if defined(__AVX2__) && defined(__FMA__)
static double get_relative_error(
    const float *restrict srcp0, const float *restrict srcp1, int src_w, int src_h, ptrdiff_t stride
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix0 = _mm256_load_ps(srcp0 + x);
            __m256 pix1 = _mm256_load_ps(srcp1 + x);
            __m256d pix0_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(pix0, 0));
            __m256d pix0_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(pix0, 1));
            __m256d sub0 = _mm256_sub_pd(pix0_0, _mm256_cvtps_pd(_mm256_extractf128_ps(pix1, 0)));
            __m256d sub1 = _mm256_sub_pd(pix0_1, _mm256_cvtps_pd(_mm256_extractf128_ps(pix1, 1)));
            acc0 = _mm256_fmadd_pd(sub0, sub0, acc0);
            acc1 = _mm256_fmadd_pd(sub1, sub1, acc1);
            acc2 = _mm256_fmadd_pd(pix0_0, pix0_0, acc2);
            acc3 = _mm256_fmadd_pd(pix0_1, pix0_1, acc3);
        }
        if (tail) {
            __m256 pix0 = _mm256_maskload_ps(srcp0 + x, tail_mask);
            __m256 pix1 = _mm256_maskload_ps(srcp1 + x, tail_mask);
            __m256d pix0_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(pix0, 0));
            __m256d pix0_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(pix0, 1));
            __m256d sub0 = _mm256_sub_pd(pix0_0, _mm256_cvtps_pd(_mm256_extractf128_ps(pix1, 0)));
            __m256d sub1 = _mm256_sub_pd(pix0_1, _mm256_cvtps_pd(_mm256_extractf128_ps(pix1, 1)));
            acc0 = _mm256_fmadd_pd(sub0, sub0, acc0);
            acc1 = _mm256_fmadd_pd(sub1, sub1, acc1);
            acc2 = _mm256_fmadd_pd(pix0_0, pix0_0, acc2);
            acc3 = _mm256_fmadd_pd(pix0_1, pix0_1, acc3);
        }
        srcp0 += stride;
        srcp1 += stride;
    }
    
    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    __m128d acc4 = _mm_add_pd(_mm256_extractf128_pd(acc0, 0), _mm256_extractf128_pd(acc0, 1));
    __m128d acc5 = _mm_add_pd(_mm256_extractf128_pd(acc2, 0), _mm256_extractf128_pd(acc2, 1));
    acc4 = _mm_add_sd(acc4, _mm_unpackhi_pd(acc4, acc4));
    acc5 = _mm_add_sd(acc5, _mm_unpackhi_pd(acc5, acc5));
    
    return sqrt(_mm_cvtsd_f64(acc4)) / fmax(sqrt(_mm_cvtsd_f64(acc5)), 1e-16);
}
#else
static double get_relative_error(
    const float *restrict srcp0, const float *restrict srcp1, int src_w, int src_h, ptrdiff_t stride
) {
    double acc0 = 0.0;
    double acc1 = 0.0;
    
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            acc0 += pow((double)srcp0[x] - (double)srcp1[x], 2);
            acc1 += pow((double)srcp0[x], 2);
        }
        srcp0 += stride;
        srcp1 += stride;
    }
    
    return sqrt(acc0) / fmax(sqrt(acc1), 1e-16);
}
#endif

static const VSFrame *VS_CC RelativeErrorGetFrame(
    int n, int activationReason, void *instanceData, void **frameData UNUSED,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    RelativeErrorData *d = (RelativeErrorData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node0, frameCtx);
        vsapi->requestFrameFilter(n, d->node1, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src0 = vsapi->getFrameFilter(n, d->node0, frameCtx);
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->node1, frameCtx);
        VSFrame *dst = vsapi->copyFrame(src0, core);
        
        const float *restrict srcp0 = (const float *)vsapi->getReadPtr(src0, 0);
        const float *restrict srcp1 = (const float *)vsapi->getReadPtr(src1, 0);
        ptrdiff_t src_stride = vsapi->getStride(src0, 0) / sizeof(float);
        int src_w = vsapi->getFrameWidth(src0, 0);
        int src_h = vsapi->getFrameHeight(src0, 0);
        
        double err = get_relative_error(srcp0, srcp1, src_w, src_h, src_stride);
        
        VSMap *props = vsapi->getFramePropertiesRW(dst);
        vsapi->mapSetFloat(props, "RelativeError", err, maReplace);
        
        vsapi->freeFrame(src0);
        vsapi->freeFrame(src1);
        return dst;
    }
    return NULL;
}

static void VS_CC RelativeErrorFree(void *instanceData, VSCore *core UNUSED, const VSAPI *vsapi) {
    RelativeErrorData *d = (RelativeErrorData *)instanceData;
    vsapi->freeNode(d->node0);
    vsapi->freeNode(d->node1);
    free(d);
}

static void VS_CC RelativeErrorCreate(
    const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi
) {
    RelativeErrorData d;
    d.node0 = vsapi->mapGetNode(in, "clip0", 0, NULL);
    d.node1 = vsapi->mapGetNode(in, "clip1", 0, NULL);
    VSVideoInfo vi0 = *vsapi->getVideoInfo(d.node0);
    VSVideoInfo vi1 = *vsapi->getVideoInfo(d.node1);
    
    if (!vsh_isConstantVideoFormat(&vi0) || !vsh_isSameVideoPresetFormat(pfGrayS, &vi0.format, core, vsapi)) {
        vsapi->mapSetError(out, "RelativeError: only GRAY constant format 32bit float input supported");
        vsapi->freeNode(d.node0);
        vsapi->freeNode(d.node1);
        return;
    }
    
    if (!vsh_isSameVideoInfo(&vi0, &vi1) || vi0.numFrames != vi1.numFrames) {
        vsapi->mapSetError(out, "RelativeError: clips must have the same format and number of frames");
        vsapi->freeNode(d.node0);
        vsapi->freeNode(d.node1);
        return;
    }
    
    RelativeErrorData *data = (RelativeErrorData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node0, rpStrictSpatial}, {d.node1, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "RelativeError", &vi0, RelativeErrorGetFrame, RelativeErrorFree, fmParallel, deps, 2, data, core);
}

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    GammaData gamma;
    bool process[3];
} LinearData;

static const VSFrame *VS_CC LinearizeGetFrame(
    int n, int activationReason, void *instanceData, void **frameData UNUSED,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    LinearData *d = (LinearData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        VSFrame *dst = vsapi->newVideoFrame(fi, d->vi.width, d->vi.height, src, core);
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float *restrict srcp = (const float *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            float *restrict dstp = (float *)vsapi->getWritePtr(dst, plane);
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            
            if (d->process[plane]) {
                to_linear(srcp, dstp, src_stride, src_w, src_h, d->gamma);
            }
            else {
                vsh_bitblt(dstp, sizeof(float) * src_stride, srcp, sizeof(float) * src_stride, sizeof(float) * src_w, src_h);
            }
        }
        vsapi->freeFrame(src);
        return dst;
    }
    return NULL;
}

static void VS_CC LinearizeFree(void *instanceData, VSCore *core UNUSED, const VSAPI *vsapi) {
    LinearData *d = (LinearData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC LinearizeCreate(
    const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi
) {
    LinearData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "Linearize: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    const char *gamma = vsapi->mapGetData(in, "gamma", 0, &err);
    if (err) {
        if (d.vi.format.colorFamily == cfRGB) {
            d.gamma = (GammaData){2.4f, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
        }
        else {
            d.gamma = (GammaData){1.0f / 0.45f, 0.081f, 0.018f, 0.099f, 4.5f, false};
        }
    }
    else if (!strcmp(gamma, "srgb")) {
        d.gamma = (GammaData){2.4f, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
    }
    else if (!strcmp(gamma, "smpte170m")) {
        d.gamma = (GammaData){1.0f / 0.45f, 0.081f, 0.018f, 0.099f, 4.5f, false};
    }
    else if (!strcmp(gamma, "adobe")) {
        d.gamma = (GammaData){2.19921875f, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "dcip3")) {
        d.gamma = (GammaData){2.6f, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "smpte240m")) {
        d.gamma = (GammaData){1.0f / 0.45f, 0.0913f, 0.0228f, 0.1115f, 4.0f, false};
    }
    else {
        vsapi->mapSetError(out, "Linearize: invalid gamma specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    const int m = vsapi->mapNumElements(in, "planes");
    
    for (int i = 0; i < 3; i++) {
        d.process[i] = (m <= 0);
    }
    
    for (int i = 0; i < m; i++) {
        const int n = vsapi->mapGetIntSaturated(in, "planes", i, NULL);
        
        if (n < 0 || n >= d.vi.format.numPlanes) {
            vsapi->mapSetError(out, "Linearize: plane index out of range");
            vsapi->freeNode(d.node);
            return;
        }
        
        if (d.process[n]) {
            vsapi->mapSetError(out, "Linearize: plane specified twice");
            vsapi->freeNode(d.node);
            return;
        }
         
        d.process[n] = true;
    }
    
    LinearData *data = (LinearData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Linearize", &d.vi, LinearizeGetFrame, LinearizeFree, fmParallel, deps, 1, data, core);
}

static const VSFrame *VS_CC GammaCorrGetFrame(
    int n, int activationReason, void *instanceData, void **frameData UNUSED,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    LinearData *d = (LinearData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        VSFrame *dst = vsapi->newVideoFrame(fi, d->vi.width, d->vi.height, src, core);
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float *restrict srcp = (const float *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            float *restrict dstp = (float *)vsapi->getWritePtr(dst, plane);
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            
            if (d->process[plane]) {
                from_linear(srcp, dstp, src_stride, src_w, src_h, d->gamma);
            }
            else {
                vsh_bitblt(dstp, sizeof(float) * src_stride, srcp, sizeof(float) * src_stride, sizeof(float) * src_w, src_h);
            }
        }
        vsapi->freeFrame(src);
        return dst;
    }
    return NULL;
}

static void VS_CC GammaCorrFree(void *instanceData, VSCore *core UNUSED, const VSAPI *vsapi) {
    LinearData *d = (LinearData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC GammaCorrCreate(
    const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi
) {
    LinearData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "GammaCorr: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    const char *gamma = vsapi->mapGetData(in, "gamma", 0, &err);
    if (err) {
        if (d.vi.format.colorFamily == cfRGB) {
            d.gamma = (GammaData){2.4f, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
        }
        else {
            d.gamma = (GammaData){1.0f / 0.45f, 0.081f, 0.018f, 0.099f, 4.5f, false};
        }
    }
    else if (!strcmp(gamma, "srgb")) {
        d.gamma = (GammaData){2.4f, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
    }
    else if (!strcmp(gamma, "smpte170m")) {
        d.gamma = (GammaData){1.0f / 0.45f, 0.081f, 0.018f, 0.099f, 4.5f, false};
    }
    else if (!strcmp(gamma, "adobe")) {
        d.gamma = (GammaData){2.19921875f, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "dcip3")) {
        d.gamma = (GammaData){2.6f, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "smpte240m")) {
        d.gamma = (GammaData){1.0f / 0.45f, 0.0913f, 0.0228f, 0.1115f, 4.0f, false};
    }
    else {
        vsapi->mapSetError(out, "GammaCorr: invalid gamma specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    const int m = vsapi->mapNumElements(in, "planes");
    
    for (int i = 0; i < 3; i++) {
        d.process[i] = (m <= 0);
    }
    
    for (int i = 0; i < m; i++) {
        const int n = vsapi->mapGetIntSaturated(in, "planes", i, NULL);
        
        if (n < 0 || n >= d.vi.format.numPlanes) {
            vsapi->mapSetError(out, "GammaCorr: plane index out of range");
            vsapi->freeNode(d.node);
            return;
        }
        
        if (d.process[n]) {
            vsapi->mapSetError(out, "GammaCorr: plane specified twice");
            vsapi->freeNode(d.node);
            return;
        }
        
        d.process[n] = true;
    }
    
    LinearData *data = (LinearData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "GammaCorr", &d.vi, GammaCorrGetFrame, GammaCorrFree, fmParallel, deps, 1, data, core);
}

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    bool direct;
} BitDepthData;

static const VSFrame *VS_CC BitDepthGetFrame(
    int n, int activationReason, void *instanceData, void **frameData UNUSED,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    BitDepthData *d = (BitDepthData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        VSFrame *dst = vsapi->newVideoFrame(&d->vi.format, d->vi.width, d->vi.height, src, core);
        
        const VSMap *props = vsapi->getFramePropertiesRO(src);
        
        int err;
        int range = vsapi->mapGetIntSaturated(props, "_ColorRange", 0, &err);
        
        if (d->direct) {
            range = 0;
        }
        else if (err || range < 0 || range > 1) {
            range = (fi->colorFamily == cfRGB) ? 0 : 1;
        }
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const void *restrict srcp = (const void *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / fi->bytesPerSample;
            void *restrict dstp = (void *)vsapi->getWritePtr(dst, plane);
            ptrdiff_t dst_stride = vsapi->getStride(dst, plane) / d->vi.format.bytesPerSample;
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            bool chroma = plane && (fi->colorFamily == cfYUV);
            
            if ((fi->bytesPerSample == 1) && (d->vi.format.bytesPerSample == 4)) {
                uint8_to_float(srcp, dstp, src_stride, dst_stride, src_w, src_h, chroma, range);
            }
            else if ((fi->bytesPerSample == 2) && (d->vi.format.bytesPerSample == 4)) {
                uint16_to_float(srcp, dstp, src_stride, dst_stride, src_w, src_h, chroma, range, fi->bitsPerSample);
            }
            else if ((fi->bytesPerSample == 4) && (d->vi.format.bytesPerSample == 1)) {
                float_to_uint8(srcp, dstp, src_stride, dst_stride, src_w, src_h, chroma, range);
            }
            else if ((fi->bytesPerSample == 4) && (d->vi.format.bytesPerSample == 2)) {
                float_to_uint16(srcp, dstp, src_stride, dst_stride, src_w, src_h, chroma, range, d->vi.format.bitsPerSample);
            }
            else if ((fi->bytesPerSample == 1) && (d->vi.format.bytesPerSample == 2)) {
                uint8_to_uint16(srcp, dstp, src_stride, dst_stride, src_w, src_h, d->vi.format.bitsPerSample);
            }
            else if ((fi->bytesPerSample == 2) && (d->vi.format.bytesPerSample == 1)) {
                uint16_to_uint8(srcp, dstp, src_stride, dst_stride, src_w, src_h, fi->bitsPerSample);
            }
            else {
                uint16_to_uint16(srcp, dstp, src_stride, dst_stride, src_w, src_h, fi->bitsPerSample, d->vi.format.bitsPerSample);
            }
        }
        vsapi->freeFrame(src);
        return dst;
    }
    return NULL;
}

static void VS_CC BitDepthFree(void *instanceData, VSCore *core UNUSED, const VSAPI *vsapi) {
    BitDepthData *d = (BitDepthData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC BitDepthCreate(
    const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi
) {
    BitDepthData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (
        !vsh_isConstantVideoFormat(&d.vi) ||
        (d.vi.format.sampleType == stInteger && (d.vi.format.bitsPerSample < 8 || d.vi.format.bitsPerSample > 16)) ||
        (d.vi.format.sampleType == stFloat && d.vi.format.bitsPerSample != 32)
    ) {
        vsapi->mapSetError(out, "BitDepth: only constant format 8-16bit integer or 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    int bits = vsapi->mapGetIntSaturated(in, "bits", 0, NULL);
    
    if (bits == d.vi.format.bitsPerSample) {
        vsapi->mapSetError(out, "BitDepth: same \"bits\" as input format is not allowed");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (bits < 8 || (bits > 16 && bits < 32) || bits > 32) {
        vsapi->mapSetError(out, "BitDepth: \"bits\" must be between 8 and 16 or 32");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    d.direct = !!vsapi->mapGetIntSaturated(in, "direct", 0, &err);
    
    if (err) {
        d.direct = false;
    }
    
    d.vi.format.bitsPerSample = bits;
    d.vi.format.bytesPerSample = (bits + 7) / 8;
    d.vi.format.sampleType = (bits == 32) ? stFloat : stInteger;
    
    BitDepthData *data = (BitDepthData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "BitDepth", &d.vi, BitDepthGetFrame, BitDepthFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("ru.artyfox.plugins", "artyfox", "A disjointed set of filters", VS_MAKE_VERSION(15, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Resize",
                             "clip:vnode;"
                             "width:int;"
                             "height:int;"
                             "src_left:float:opt;"
                             "src_top:float:opt;"
                             "src_width:float:opt;"
                             "src_height:float:opt;"
                             "kernel:data:opt;"
                             "b:float:opt;"
                             "c:float:opt;"
                             "taps:int:opt;"
                             "gamma:data:opt;"
                             "sharp:float:opt;",
                             "clip:vnode;",
                             ResizeCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("Descale",
                             "clip:vnode;"
                             "width:int;"
                             "height:int;"
                             "src_left:float:opt;"
                             "src_top:float:opt;"
                             "src_width:float:opt;"
                             "src_height:float:opt;"
                             "kernel:data:opt;"
                             "b:float:opt;"
                             "c:float:opt;"
                             "taps:int:opt;"
                             "lambda:float:opt;",
                             "clip:vnode;",
                             DescaleCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("RelativeError",
                             "clip0:vnode;"
                             "clip1:vnode;",
                             "clip:vnode;",
                             RelativeErrorCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("Linearize",
                             "clip:vnode;"
                             "gamma:data:opt;"
                             "planes:int[]:opt;",
                             "clip:vnode;",
                             LinearizeCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("GammaCorr",
                             "clip:vnode;"
                             "gamma:data:opt;"
                             "planes:int[]:opt;",
                             "clip:vnode;",
                             GammaCorrCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("BitDepth",
                             "clip:vnode;"
                             "bits:int;"
                             "direct:int:opt;",
                             "clip:vnode;",
                             BitDepthCreate,
                             NULL,
                             plugin);
}
