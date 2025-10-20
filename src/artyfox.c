#include <stdbool.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#define CLAMP(x, min, max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : (x))) 
#define M_PI 3.14159265358979323846

#ifdef __AVX2__
#define _MM256_TRANSPOSE8_PS(row0, row1, row2, row3, row4, row5, row6, row7) \
do { \
    __m256 __t0 = _mm256_unpacklo_ps((row0), (row2)); \
    __m256 __t1 = _mm256_unpackhi_ps((row0), (row2)); \
    __m256 __t2 = _mm256_unpacklo_ps((row1), (row3)); \
    __m256 __t3 = _mm256_unpackhi_ps((row1), (row3)); \
    __m256 __t4 = _mm256_unpacklo_ps((row4), (row6)); \
    __m256 __t5 = _mm256_unpackhi_ps((row4), (row6)); \
    __m256 __t6 = _mm256_unpacklo_ps((row5), (row7)); \
    __m256 __t7 = _mm256_unpackhi_ps((row5), (row7)); \
    __m256 __u0 = _mm256_unpacklo_ps(__t0, __t2); \
    __m256 __u1 = _mm256_unpackhi_ps(__t0, __t2); \
    __m256 __u2 = _mm256_unpacklo_ps(__t1, __t3); \
    __m256 __u3 = _mm256_unpackhi_ps(__t1, __t3); \
    __m256 __u4 = _mm256_unpacklo_ps(__t4, __t6); \
    __m256 __u5 = _mm256_unpackhi_ps(__t4, __t6); \
    __m256 __u6 = _mm256_unpacklo_ps(__t5, __t7); \
    __m256 __u7 = _mm256_unpackhi_ps(__t5, __t7); \
    (row0) = _mm256_permute2f128_ps(__u0, __u4, 0b00100000); \
    (row1) = _mm256_permute2f128_ps(__u1, __u5, 0b00100000); \
    (row2) = _mm256_permute2f128_ps(__u2, __u6, 0b00100000); \
    (row3) = _mm256_permute2f128_ps(__u3, __u7, 0b00100000); \
    (row4) = _mm256_permute2f128_ps(__u0, __u4, 0b00110001); \
    (row5) = _mm256_permute2f128_ps(__u1, __u5, 0b00110001); \
    (row6) = _mm256_permute2f128_ps(__u2, __u6, 0b00110001); \
    (row7) = _mm256_permute2f128_ps(__u3, __u7, 0b00110001); \
} while (0)
#endif

typedef double (*kernel_func)(double x, void *ctx);

typedef struct {
    kernel_func f;
    double radius;
    void *ctx;
} kernel_t;

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int dst_width, dst_height;
    double start_w, start_h, real_w, real_h;
    kernel_t kernel_w, kernel_h;
    float gamma, sharp;
    bool process_w, process_h;
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

static double magic_kernel(double x, void *ctx) {
    (void)ctx;
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

static double magic_kernel_2013(double x, void *ctx) {
    (void)ctx;
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

static double magic_kernel_2021(double x, void *ctx) {
    (void)ctx;
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

static double bilinear_kernel(double x, void *ctx) {
    (void)ctx;
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

static double spline16_kernel(double x, void *ctx) {
    (void)ctx;
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

static double spline36_kernel(double x, void *ctx) {
    (void)ctx;
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

static double spline64_kernel(double x, void *ctx) {
    (void)ctx;
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

static double spline100_kernel(double x, void *ctx) {
    (void)ctx;
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

static double spline144_kernel(double x, void *ctx) {
    (void)ctx;
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

static double point_kernel(double x, void *ctx) {
    (void)ctx;
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

static void rgb_to_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_0_04045 = _mm256_set1_ps(0.04045f);
    __m256 v_0_055 = _mm256_set1_ps(0.055f);
    __m256 v_1_055 = _mm256_set1_ps(1.055f);
    __m256 v_gamma = _mm256_set1_ps(gamma);
    __m256 v_12_92 = _mm256_set1_ps(12.92f);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_04045, _CMP_GT_OQ);
            __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_0_055), v_1_055), v_gamma);
            __m256 branch_1 = _mm256_div_ps(pix_abs, v_12_92);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_04045, _CMP_GT_OQ);
            __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_0_055), v_1_055), v_gamma);
            __m256 branch_1 = _mm256_div_ps(pix_abs, v_12_92);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        srcp += stride;
        dstp += stride;
    }
    _mm_sfence();
}

static void yuv_to_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_0_081 = _mm256_set1_ps(0.081f);
    __m256 v_0_099 = _mm256_set1_ps(0.099f);
    __m256 v_1_099 = _mm256_set1_ps(1.099f);
    __m256 v_gamma = _mm256_set1_ps(gamma);
    __m256 v_4_5 = _mm256_set1_ps(4.5f);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_081, _CMP_GE_OQ);
            __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_0_099), v_1_099), v_gamma);
            __m256 branch_1 = _mm256_div_ps(pix_abs, v_4_5);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_081, _CMP_GE_OQ);
            __m256 branch_0 = _mm256_pow_ps(_mm256_div_ps(_mm256_add_ps(pix_abs, v_0_099), v_1_099), v_gamma);
            __m256 branch_1 = _mm256_div_ps(pix_abs, v_4_5);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        srcp += stride;
        dstp += stride;
    }
    _mm_sfence();
}

static void linear_to_rgb(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_0_0031308 = _mm256_set1_ps(0.0031308f);
    __m256 v_0_055 = _mm256_set1_ps(0.055f);
    __m256 v_1_055 = _mm256_set1_ps(1.055f);
    __m256 v_gamma = _mm256_set1_ps(1.0f / gamma);
    __m256 v_12_92 = _mm256_set1_ps(12.92f);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_0031308, _CMP_GT_OQ);
            __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_1_055, v_0_055);
            __m256 branch_1 = _mm256_mul_ps(pix, v_12_92);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_0031308, _CMP_GT_OQ);
            __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_1_055, v_0_055);
            __m256 branch_1 = _mm256_mul_ps(pix, v_12_92);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        srcp += stride;
        dstp += stride;
    }
    _mm_sfence();
}

static void linear_to_yuv(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_0_018 = _mm256_set1_ps(0.018f);
    __m256 v_0_099 = _mm256_set1_ps(0.099f);
    __m256 v_1_099 = _mm256_set1_ps(1.099f);
    __m256 v_gamma = _mm256_set1_ps(1.0f / gamma);
    __m256 v_4_5 = _mm256_set1_ps(4.5f);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_018, _CMP_GE_OQ);
            __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_1_099, v_0_099);
            __m256 branch_1 = _mm256_mul_ps(pix, v_4_5);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 pix_abs = _mm256_and_ps(pix, v_abs);
            __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_0_018, _CMP_GE_OQ);
            __m256 branch_0 = _mm256_fmsub_ps(_mm256_pow_ps(pix_abs, v_gamma), v_1_099, v_0_099);
            __m256 branch_1 = _mm256_mul_ps(pix, v_4_5);
            __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
            _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
        }
        srcp += stride;
        dstp += stride;
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
    __m128i tale_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m256i pix = _mm256_cvtepu8_epi16(_mm_load_si128((__m128i *)(srcp + x)));
            __m256i branch = _mm256_slli_epi16(pix, count);
            _mm256_stream_si256((__m256i *)(dstp + x), branch);
        }
        if (tail) {
            __m256i pix = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tale_mask));
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
    __m128i tale_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(chroma ? 128.0f : 16.0f);
        v_high = _mm256_set1_ps(chroma ? 224.0f : 219.0f);
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 128.0f : 0.0f);
        v_high = _mm256_set1_ps(chroma ? 254.0f : 255.0f);
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m128i pix = _mm_load_si128((__m128i *)(srcp + x));
            __m256 pix_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pix));
            __m256 pix_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(pix, _MM_SHUFFLE(1, 0, 3, 2))));
            __m256 branch_0 = _mm256_div_ps(_mm256_sub_ps(pix_0, v_low), v_high);
            __m256 branch_1 = _mm256_div_ps(_mm256_sub_ps(pix_1, v_low), v_high);
            _mm256_stream_ps(dstp + x, branch_0);
            _mm256_stream_ps(dstp + x + 8, branch_1);
        }
        if (tail > 8) {
            __m128i pix = _mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tale_mask);
            __m256 pix_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pix));
            __m256 pix_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi32(pix, _MM_SHUFFLE(1, 0, 3, 2))));
            __m256 branch_0 = _mm256_div_ps(_mm256_sub_ps(pix_0, v_low), v_high);
            __m256 branch_1 = _mm256_div_ps(_mm256_sub_ps(pix_1, v_low), v_high);
            _mm256_stream_ps(dstp + x, branch_0);
            _mm256_stream_ps(dstp + x + 8, branch_1);
        }
        else if (tail) {
            __m128i pix = _mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tale_mask);
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
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_div = _mm256_set1_ps((float)(1 << (bits - 8)));
    __m256 v_half = _mm256_set1_ps(0.5f);
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m256i pix = _mm256_load_si256((__m256i *)(srcp + x));
            __m256 pix_f_0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 0)));
            __m256 pix_f_1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 1)));
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_0, v_div), v_half));
            __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_1, v_div), v_half));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            __m128i pix_u_1 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_1, 0), _mm256_extracti128_si256(pix_i_1, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, pix_u_1));
        }
        if (tail) {
            __m256i pix = _mm256_and_si256(_mm256_load_si256((__m256i *)(srcp + x)), tale_mask);
            __m256 pix_f_0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 0)));
            __m256 pix_f_1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 1)));
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_0, v_div), v_half));
            __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_1, v_div), v_half));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            __m128i pix_u_1 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_1, 0), _mm256_extracti128_si256(pix_i_1, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, pix_u_1));
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
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
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
                __m256i pix = _mm256_and_si256(_mm256_load_si256((__m256i *)(srcp + x)), tale_mask);
                __m256i branch = _mm256_slli_epi16(pix, count);
                _mm256_stream_si256((__m256i *)(dstp + x), branch);
            }
            srcp += src_stride;
            dstp += dst_stride;
        }
    }
    else {
        __m256 v_div = _mm256_set1_ps((float)(1 << (src_bits - dst_bits)));
        __m256 v_half = _mm256_set1_ps(0.5f);
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod16_w; x += 16) {
                __m256i pix = _mm256_load_si256((__m256i *)(srcp + x));
                __m256 pix_f_0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 0)));
                __m256 pix_f_1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 1)));
                __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_0, v_div), v_half));
                __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_1, v_div), v_half));
                __m256i branch = _mm256_permute4x64_epi64(_mm256_packus_epi32(pix_i_0, pix_i_1), _MM_SHUFFLE(3, 1, 2, 0));
                _mm256_stream_si256((__m256i *)(dstp + x), branch);
            }
            if (tail) {
                __m256i pix = _mm256_and_si256(_mm256_load_si256((__m256i *)(srcp + x)), tale_mask);
                __m256 pix_f_0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 0)));
                __m256 pix_f_1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(pix, 1)));
                __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_0, v_div), v_half));
                __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_div_ps(pix_f_1, v_div), v_half));
                __m256i branch = _mm256_permute4x64_epi64(_mm256_packus_epi32(pix_i_0, pix_i_1), _MM_SHUFFLE(3, 1, 2, 0));
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
    __m128i tale_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps((float)((chroma ? 128 : 16) << (bits - 8)));
        v_high = _mm256_set1_ps((float)((chroma ? 224 : 219) << (bits - 8)));
    }
    else {
        v_low = _mm256_set1_ps(chroma ? (float)(128 << (bits - 8)) : 0.0f);
        v_high = _mm256_set1_ps(chroma ? (float)((1 << bits) - 2) : (float)((1 << bits) - 1));
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
            __m128i pix = _mm_and_si128(_mm_load_si128((__m128i *)(srcp + x)), tale_mask);
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
    __m256i tale_mask_0 = _mm256_loadu_si256((__m256i *)mask_arr);
    __m256i tale_mask_1 = _mm256_loadu_si256((__m256i *)(mask_arr + 8));
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(chroma ? 128.5f : 16.5f);
        v_high = _mm256_set1_ps(chroma ? 224.0f : 219.0f);
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 128.5f : 0.5f);
        v_high = _mm256_set1_ps(chroma ? 254.0f : 255.0f);
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod16_w; x += 16) {
            __m256 pix_f_0 = _mm256_load_ps(srcp + x);
            __m256 pix_f_1 = _mm256_load_ps(srcp + x + 8);
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_0, v_high, v_low));
            __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_1, v_high, v_low));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            __m128i pix_u_1 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_1, 0), _mm256_extracti128_si256(pix_i_1, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, pix_u_1));
        }
        if (tail > 8) {
            __m256 pix_f_0 = _mm256_load_ps(srcp + x);
            __m256 pix_f_1 = _mm256_maskload_ps(srcp + x + 8, tale_mask_1);
            __m256i pix_i_0 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_0, v_high, v_low));
            __m256i pix_i_1 = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f_1, v_high, v_low));
            __m128i pix_u_0 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_0, 0), _mm256_extracti128_si256(pix_i_0, 1));
            __m128i pix_u_1 = _mm_packus_epi32(_mm256_extracti128_si256(pix_i_1, 0), _mm256_extracti128_si256(pix_i_1, 1));
            _mm_stream_si128((__m128i *)(dstp + x), _mm_packus_epi16(pix_u_0, pix_u_1));
        }
        else if (tail) {
            __m256 pix_f_0 = _mm256_maskload_ps(srcp + x, tale_mask_0);
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
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(0.5f + (float)((chroma ? 128 : 16) << (bits - 8)));
        v_high = _mm256_set1_ps((float)((chroma ? 224 : 219) << (bits - 8)));
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 0.5f + (float)(128 << (bits - 8)) : 0.5f);
        v_high = _mm256_set1_ps(chroma ? (float)((1 << bits) - 2) : (float)((1 << bits) - 1));
    }
    
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix_f = _mm256_load_ps(srcp + x);
            __m256i pix_i = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f, v_high, v_low));
            __m128i pix_u = _mm_packus_epi32(_mm256_extracti128_si256(pix_i, 0), _mm256_extracti128_si256(pix_i, 1));
            _mm_stream_si128((__m128i *)(dstp + x), pix_u);
        }
        if (tail) {
            __m256 pix_f = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256i pix_i = _mm256_cvttps_epi32(_mm256_fmadd_ps(pix_f, v_high, v_low));
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
    __m256i tale_mask_0 = _mm256_loadu_si256((__m256i *)mask_arr);
    __m256i tale_mask_1 = _mm256_loadu_si256((__m256i *)(mask_arr + 1));
    
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
        v_0 = _mm256_maskload_ps(srcp + x - 1, tale_mask_0);
        v_1 = _mm256_maskload_ps(srcp + x, tale_mask_1);
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
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
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
        __m256 v_1 = _mm256_maskload_ps(srcp + x, tale_mask);
        __m256 v_2 = _mm256_maskload_ps(srcp + x + stride, tale_mask);
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
            __m256 v_0 = _mm256_maskload_ps(srcp + x - stride, tale_mask);
            __m256 v_1 = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 v_2 = _mm256_maskload_ps(srcp + x + stride, tale_mask);
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
        __m256 v_0 = _mm256_maskload_ps(srcp + x - stride, tale_mask);
        __m256 v_1 = _mm256_maskload_ps(srcp + x, tale_mask);
        __m256 v_avg = _mm256_mul_ps(_mm256_add_ps(_mm256_add_ps(v_0, v_1), v_1), v_mul);
        _mm256_store_ps(dstp + x, _mm256_fmadd_ps(v_1, v_sharp, v_avg));
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
    
    float *restrict src_buf = (float *)_mm_malloc(sizeof(float) * src_stride * 8, 64);
    float *restrict dst_buf = (float *)_mm_malloc(sizeof(float) * dst_stride * 8, 64);
    
    int tail = src_h % 8;
    int mod8_h = src_h - tail;
    
    for (int y = 0; y < mod8_h; y += 8) {
        for (int x = 0; x < src_w; x += 8) {
            __m256 line_0 = _mm256_load_ps(srcp + x);
            __m256 line_1 = _mm256_load_ps(srcp + x + src_stride);
            __m256 line_2 = _mm256_load_ps(srcp + x + src_stride * 2);
            __m256 line_3 = _mm256_load_ps(srcp + x + src_stride * 3);
            __m256 line_4 = _mm256_load_ps(srcp + x + src_stride * 4);
            __m256 line_5 = _mm256_load_ps(srcp + x + src_stride * 5);
            __m256 line_6 = _mm256_load_ps(srcp + x + src_stride * 6);
            __m256 line_7 = _mm256_load_ps(srcp + x + src_stride * 7);
            _MM256_TRANSPOSE8_PS(line_0, line_1, line_2, line_3, line_4, line_5, line_6, line_7);
            _mm256_store_ps(src_buf + x * 8, line_0);
            _mm256_store_ps(src_buf + x * 8 + 8, line_1);
            _mm256_store_ps(src_buf + x * 8 + 16, line_2);
            _mm256_store_ps(src_buf + x * 8 + 24, line_3);
            _mm256_store_ps(src_buf + x * 8 + 32, line_4);
            _mm256_store_ps(src_buf + x * 8 + 40, line_5);
            _mm256_store_ps(src_buf + x * 8 + 48, line_6);
            _mm256_store_ps(src_buf + x * 8 + 56, line_7);
        }
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = 0; i < lengths[x]; i++) {
                __m256 pix = _mm256_load_ps(src_buf + counts[x * step + i] * 8);
                __m256d v_weight = _mm256_set1_pd(weights[x * step + i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_store_ps(dst_buf + x * 8, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        for (int x = 0; x < dst_w; x += 8) {
            __m256 line_0 = _mm256_load_ps(dst_buf + x * 8);
            __m256 line_1 = _mm256_load_ps(dst_buf + x * 8 + 8);
            __m256 line_2 = _mm256_load_ps(dst_buf + x * 8 + 16);
            __m256 line_3 = _mm256_load_ps(dst_buf + x * 8 + 24);
            __m256 line_4 = _mm256_load_ps(dst_buf + x * 8 + 32);
            __m256 line_5 = _mm256_load_ps(dst_buf + x * 8 + 40);
            __m256 line_6 = _mm256_load_ps(dst_buf + x * 8 + 48);
            __m256 line_7 = _mm256_load_ps(dst_buf + x * 8 + 56);
            _MM256_TRANSPOSE8_PS(line_0, line_1, line_2, line_3, line_4, line_5, line_6, line_7);
            _mm256_stream_ps(dstp + x, line_0);
            _mm256_stream_ps(dstp + x + dst_stride, line_1);
            _mm256_stream_ps(dstp + x + dst_stride * 2, line_2);
            _mm256_stream_ps(dstp + x + dst_stride * 3, line_3);
            _mm256_stream_ps(dstp + x + dst_stride * 4, line_4);
            _mm256_stream_ps(dstp + x + dst_stride * 5, line_5);
            _mm256_stream_ps(dstp + x + dst_stride * 6, line_6);
            _mm256_stream_ps(dstp + x + dst_stride * 7, line_7);
        }
        dstp += dst_stride * 8;
        srcp += src_stride * 8;
    }
    if (tail) {
        for (int x = 0; x < src_w; x += 8) {
            __m256 line_0 = _mm256_load_ps(srcp + x);
            __m256 line_1 = (tail > 1) ? _mm256_load_ps(srcp + x + src_stride) : _mm256_setzero_ps();
            __m256 line_2 = (tail > 2) ? _mm256_load_ps(srcp + x + src_stride * 2) : _mm256_setzero_ps();
            __m256 line_3 = (tail > 3) ? _mm256_load_ps(srcp + x + src_stride * 3) : _mm256_setzero_ps();
            __m256 line_4 = (tail > 4) ? _mm256_load_ps(srcp + x + src_stride * 4) : _mm256_setzero_ps();
            __m256 line_5 = (tail > 5) ? _mm256_load_ps(srcp + x + src_stride * 5) : _mm256_setzero_ps();
            __m256 line_6 = (tail > 6) ? _mm256_load_ps(srcp + x + src_stride * 6) : _mm256_setzero_ps();
            __m256 line_7 = _mm256_setzero_ps();
            _MM256_TRANSPOSE8_PS(line_0, line_1, line_2, line_3, line_4, line_5, line_6, line_7);
            _mm256_store_ps(src_buf + x * 8, line_0);
            _mm256_store_ps(src_buf + x * 8 + 8, line_1);
            _mm256_store_ps(src_buf + x * 8 + 16, line_2);
            _mm256_store_ps(src_buf + x * 8 + 24, line_3);
            _mm256_store_ps(src_buf + x * 8 + 32, line_4);
            _mm256_store_ps(src_buf + x * 8 + 40, line_5);
            _mm256_store_ps(src_buf + x * 8 + 48, line_6);
            _mm256_store_ps(src_buf + x * 8 + 56, line_7);
        }
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = 0; i < lengths[x]; i++) {
                __m256 pix = _mm256_load_ps(src_buf + counts[x * step + i] * 8);
                __m256d v_weight = _mm256_set1_pd(weights[x * step + i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_store_ps(dst_buf + x * 8, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        for (int x = 0; x < dst_w; x += 8) {
            __m256 line_0 = _mm256_load_ps(dst_buf + x * 8);
            __m256 line_1 = _mm256_load_ps(dst_buf + x * 8 + 8);
            __m256 line_2 = _mm256_load_ps(dst_buf + x * 8 + 16);
            __m256 line_3 = _mm256_load_ps(dst_buf + x * 8 + 24);
            __m256 line_4 = _mm256_load_ps(dst_buf + x * 8 + 32);
            __m256 line_5 = _mm256_load_ps(dst_buf + x * 8 + 40);
            __m256 line_6 = _mm256_load_ps(dst_buf + x * 8 + 48);
            __m256 line_7 = _mm256_load_ps(dst_buf + x * 8 + 56);
            _MM256_TRANSPOSE8_PS(line_0, line_1, line_2, line_3, line_4, line_5, line_6, line_7);
            _mm256_stream_ps(dstp + x, line_0);
            if (tail > 1) _mm256_stream_ps(dstp + x + dst_stride, line_1);
            if (tail > 2) _mm256_stream_ps(dstp + x + dst_stride * 2, line_2);
            if (tail > 3) _mm256_stream_ps(dstp + x + dst_stride * 3, line_3);
            if (tail > 4) _mm256_stream_ps(dstp + x + dst_stride * 4, line_4);
            if (tail > 5) _mm256_stream_ps(dstp + x + dst_stride * 5, line_5);
            if (tail > 6) _mm256_stream_ps(dstp + x + dst_stride * 6, line_6);
        }
    }
    _mm_free(src_buf);
    _mm_free(dst_buf);
    free(counts);
    free(weights);
    free(lengths);
    _mm_sfence();
}

static void resize_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t dst_stride,
    int src_w, int src_h, int dst_h, double start_h, double real_h, kernel_t kernel
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
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
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = 0; i < length; i++) {
                __m256 pix = _mm256_load_ps(srcp + counts[i] + x);
                __m256d v_weight = _mm256_set1_pd(weights[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_stream_ps(dstp + x, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        if (tail) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = 0; i < length; i++) {
                __m256 pix = _mm256_maskload_ps(srcp + counts[i] + x, tale_mask);
                __m256d v_weight = _mm256_set1_pd(weights[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_stream_ps(dstp + x, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        dstp += dst_stride;
    }
    free(counts);
    free(weights);
    _mm_sfence();
}

#else

static void rgb_to_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            if (srcp[x] > 0.04045f) {
                dstp[x] = powf((srcp[x] + 0.055f) / 1.055f, gamma);
            }
            else if (srcp[x] < -0.04045f) {
                dstp[x] = -powf((-srcp[x] + 0.055f) / 1.055f, gamma);
            }
            else {
                dstp[x] = srcp[x] / 12.92f;
            }
        }
        srcp += stride;
        dstp += stride;
    }
}

static void yuv_to_linear(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            if (srcp[x] >= 0.081f) {
                dstp[x] = powf((srcp[x] + 0.099f) / 1.099f, gamma);
            }
            else if (srcp[x] <= -0.081f) {
                dstp[x] = -powf((-srcp[x] + 0.099f) / 1.099f, gamma);
            }
            else {
                dstp[x] = srcp[x] / 4.5f;
            }
        }
        srcp += stride;
        dstp += stride;
    }
}

static void linear_to_rgb(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            if (srcp[x] > 0.0031308f) {
                dstp[x] = powf(srcp[x], 1.0f / gamma) * 1.055f - 0.055f;
            }
            else if (srcp[x] < -0.0031308f) {
                dstp[x] = powf(-srcp[x], 1.0f / gamma) * -1.055f + 0.055f;
            }
            else {
                dstp[x] = srcp[x] * 12.92f;
            }
        }
        srcp += stride;
        dstp += stride;
    }
}

static void linear_to_yuv(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t stride, int src_w, int src_h, float gamma
) {
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            if (srcp[x] >= 0.018f) {
                dstp[x] = powf(srcp[x], 1.0f / gamma) * 1.099f - 0.099f;
            }
            else if (srcp[x] <= -0.018f) {
                dstp[x] = powf(-srcp[x], 1.0f / gamma) * -1.099f + 0.099f;
            }
            else {
                dstp[x] = srcp[x] * 4.5f;
            }
        }
        srcp += stride;
        dstp += stride;
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
        high = (chroma ? 254.0f : 255.0f);
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
        high = (chroma ? (1 << bits) - 2 : (1 << bits) - 1);
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
        high = (chroma ? 254.0f : 255.0f);
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
    float low, high;
    if (range) {
        low = (chroma ? 128 : 16) << (bits - 8);
        high = (chroma ? 224 : 219) << (bits - 8);
    }
    else {
        low = (chroma ? 128 << (bits - 8) : 0);
        high = (chroma ? (1 << bits) - 2 : (1 << bits) - 1);
    }
    float full = (1 << bits) - 1;
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
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    (void)frameData;
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
        if (d->gamma != 1.0f) {
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
            
            if (d->gamma != 1.0f) {
                float *restrict linp = (float *)vsapi->getWritePtr(lin, plane);
                if (fi->colorFamily == cfRGB) {
                    rgb_to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                }
                else {
                    yuv_to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                }
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
                memcpy(dstp, srcp, sizeof(float) * src_stride * src_h);
            }
            
            if (d->sharp != 1.0f) {
                float *restrict shrp = (float *)vsapi->getWritePtr(shr, plane);
                sharp_width(dstp, shrp, dst_stride, dst_w, dst_h, d->sharp);
                sharp_height(shrp, dstp, dst_stride, dst_w, dst_h, d->sharp);
            }
            
            if (d->gamma != 1.0f) {
                float *restrict gcrp = dstp;
                dstp = bit_convert ? (void *)vsapi->getWritePtr(bcd, plane) : (void *)vsapi->getWritePtr(dst, plane);
                if (fi->colorFamily == cfRGB) {
                    linear_to_rgb(gcrp, dstp, dst_stride, dst_w, dst_h, d->gamma);
                }
                else {
                    linear_to_yuv(gcrp, dstp, dst_stride, dst_w, dst_h, d->gamma);
                }
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

static void VS_CC ResizeFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;
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

static void VS_CC ResizeCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;
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
        vsapi->mapSetError(out, "Resize: width any of the planes must be greater than 1 and less than or equal to 65535");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_width % (1 << d.vi.format.subSamplingW)) {
        vsapi->mapSetError(out, "Resize: width must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > 65535) {
        vsapi->mapSetError(out, "Resize: height any of the planes must be greater than 1 and less than or equal to 65535");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height % (1 << d.vi.format.subSamplingH)) {
        vsapi->mapSetError(out, "Resize: height must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.start_w = vsapi->mapGetFloat(in, "src_left", 0, &err);
    if (err) {
        d.start_w = 0.0;
    }
    
    if (d.start_w <= -d.vi.width || d.start_w >= d.vi.width) {
        vsapi->mapSetError(out, "Resize: \"src_left\" must be between -clip.width and clip.width");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.start_h = vsapi->mapGetFloat(in, "src_top", 0, &err);
    if (err) {
        d.start_h = 0.0;
    }
    
    if (d.start_h <= -d.vi.height || d.start_h >= d.vi.height) {
        vsapi->mapSetError(out, "Resize: \"src_top\" must be between -clip.height and clip.height");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.real_w = vsapi->mapGetFloat(in, "src_width", 0, &err);
    if (err) {
        d.real_w = (double)d.vi.width;
    }
    
    if (d.real_w <= -d.vi.width || d.real_w >= d.vi.width * 2) {
        vsapi->mapSetError(out, "Resize: \"src_width\" must be between -clip.width and clip.width * 2");
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
    
    if (d.real_h <= -d.vi.height || d.real_h >= d.vi.height * 2) {
        vsapi->mapSetError(out, "Resize: \"src_height\" must be between -clip.height and clip.height * 2");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.real_h <= 0.0) {
        d.real_h += d.vi.height - d.start_h;
    }
    
    d.gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
    if (err) {
        d.gamma = (d.vi.format.colorFamily == cfRGB) ? 2.4f : 1.0f / 0.45f;
    }
    
    if (d.gamma < 0.1f || d.gamma > 5.0f) {
        vsapi->mapSetError(out, "Resize: gamma must be between 0.1 and 5");
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
            gs->taps = 4;
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
    float gamma;
    bool process[3];
} GammaData;

static const VSFrame *VS_CC LinearizeGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    (void)frameData;
    GammaData *d = (GammaData *)instanceData;
    
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
                if (fi->colorFamily == cfRGB) {
                    rgb_to_linear(srcp, dstp, src_stride, src_w, src_h, d->gamma);
                }
                else {
                    yuv_to_linear(srcp, dstp, src_stride, src_w, src_h, d->gamma);
                }
            }
            else {
                memcpy(dstp, srcp, sizeof(float) * src_stride * src_h);
            }
        }
        vsapi->freeFrame(src);
        return dst;
    }
    return NULL;
}

static void VS_CC LinearizeFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;
    GammaData *d = (GammaData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC LinearizeCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;
    GammaData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "Linearize: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
    if (err) {
        d.gamma = (d.vi.format.colorFamily == cfRGB) ? 2.4f : 1.0f / 0.45f;
    }
    
    if (d.gamma < 0.1f || d.gamma > 5.0f) {
        vsapi->mapSetError(out, "Linearize: gamma must be between 0.1 and 5");
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
    
    GammaData *data = (GammaData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "Linearize", &d.vi, LinearizeGetFrame, LinearizeFree, fmParallel, deps, 1, data, core);
}

static const VSFrame *VS_CC GammaCorrGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    (void)frameData;
    GammaData *d = (GammaData *)instanceData;
    
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
                if (fi->colorFamily == cfRGB) {
                    linear_to_rgb(srcp, dstp, src_stride, src_w, src_h, d->gamma);
                }
                else {
                    linear_to_yuv(srcp, dstp, src_stride, src_w, src_h, d->gamma);
                }
            }
            else {
                memcpy(dstp, srcp, sizeof(float) * src_stride * src_h);
            }
        }
        vsapi->freeFrame(src);
        return dst;
    }
    return NULL;
}

static void VS_CC GammaCorrFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;
    GammaData *d = (GammaData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC GammaCorrCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;
    GammaData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "GammaCorr: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
    if (err) {
        d.gamma = (d.vi.format.colorFamily == cfRGB) ? 2.4f : 1.0f / 0.45f;
    }
    
    if (d.gamma < 0.1f || d.gamma > 5.0f) {
        vsapi->mapSetError(out, "GammaCorr: gamma must be between 0.1 and 5");
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
    
    GammaData *data = (GammaData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "GammaCorr", &d.vi, GammaCorrGetFrame, GammaCorrFree, fmParallel, deps, 1, data, core);
}

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
} BitDepthData;

static const VSFrame *VS_CC BitDepthGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) {
    (void)frameData;
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
        if (err || range < 0 || range > 1) {
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

static void VS_CC BitDepthFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    (void)core;
    BitDepthData *d = (BitDepthData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC BitDepthCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    (void)userData;
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
    
    d.vi.format.bitsPerSample = bits;
    d.vi.format.bytesPerSample = (bits + 7) / 8;
    d.vi.format.sampleType = (bits == 32) ? stFloat : stInteger;
    
    BitDepthData *data = (BitDepthData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "BitDepth", &d.vi, BitDepthGetFrame, BitDepthFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("ru.artyfox.plugins", "artyfox", "A disjointed set of filters", VS_MAKE_VERSION(10, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
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
                             "gamma:float:opt;"
                             "sharp:float:opt;",
                             "clip:vnode;",
                             ResizeCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("Linearize",
                             "clip:vnode;"
                             "gamma:float:opt;"
                             "planes:int[]:opt;",
                             "clip:vnode;",
                             LinearizeCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("GammaCorr",
                             "clip:vnode;"
                             "gamma:float:opt;"
                             "planes:int[]:opt;",
                             "clip:vnode;",
                             GammaCorrCreate,
                             NULL,
                             plugin);
    vspapi->registerFunction("BitDepth",
                             "clip:vnode;"
                             "bits:int;",
                             "clip:vnode;",
                             BitDepthCreate,
                             NULL,
                             plugin);
}
