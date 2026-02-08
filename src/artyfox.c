#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <immintrin.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#define ALWAYS_INLINE __attribute__((always_inline))
#define UNUSED __attribute__((unused))
#define CLAMP(x, min, max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : (x))) 

typedef double (*kernel_func)(double x, void *ctx);

typedef struct {
    kernel_func f;
    double radius;
    void *ctx;
} kernel_t;

typedef struct {
    int col_n, row_n, nnz;
    double *values;
    int *col_idx, *row_ptr;
} csr_t;

typedef struct {
    int col_n, row_n, ku;
    double *values;
} banded_t;

typedef struct {
    double gamma;
    float thr_to, thr_from, corr, div;
    bool strict;
} GammaData;

typedef void (*convert_func)(
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits, bool range, bool chroma
);

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int dst_width, dst_height;
    double start_w, start_h, real_w, real_h;
    kernel_t kernel_w, kernel_h;
    GammaData gamma;
    float sharp;
    bool linear, process_w, process_h;
    csr_t luma_w, luma_h;
    convert_func conv_up, conv_down;
} ResizeData;

typedef struct {
    double scale;
} area_ctx;

// Based on: https://entropymine.com/imageworsener/pixelmixing/
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

// Based on: https://johncostella.com/magic/
static double magic_kernel(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 3.0 / 4.0 - x * x;
    }
    if (x < 1.5) {
        x -= 1.5;
        return 1.0 / 2.0 * (x * x);
    }
    return 0.0;
}

static double magic_kernel_2013(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 17.0 / 16.0 - 7.0 / 4.0 * (x * x);
    }
    if (x < 1.5) {
        return (1.0 - x) * (7.0 / 4.0 - x);
    }
    if (x < 2.5) {
        x -= 2.5;
        return -1.0 / 8.0 * (x * x);
    }
    return 0.0;
}

static double magic_kernel_2021(double x, void *ctx UNUSED) {
    if (x < 0.0) {
        x = -x;
    }
    if (x < 0.5) {
        return 577.0 / 576.0 - 239.0 / 144.0 * (x * x);
    }
    if (x < 1.5) {
        return 35.0 / 36.0 * (x - 1.0) * (x - 239.0 / 140.0);
    }
    if (x < 2.5) {
        return 1.0 / 6.0 * (x - 2.0) * (65.0 / 24.0 - x);
    }
    if (x < 3.5) {
        return 1.0 / 36.0 * (x - 3.0) * (x - 15.0 / 4.0);
    }
    if (x < 4.5) {
        x -= 4.5;
        return -1.0 / 288.0 * (x * x);
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

// Based on: https://entropymine.com/imageworsener/bicubic/
static double bicubic_kernel(double x, void *ctx) {
    bicubic_ctx *bc = (bicubic_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x < 1.0) {
        double u = (12.0 - 9.0 * bc->b - 6.0 * bc->c) * x;
        u = (u + (-18.0 + 12.0 * bc->b + 6.0 * bc->c)) * (x * x);
        return (u + (6.0 - 2.0 * bc->b)) / 6.0;
    }
    if (x < 2.0) {
        double u = (-bc->b - 6.0 * bc->c) * x;
        u = (u + (6.0 * bc->b + 30.0 * bc->c)) * x;
        u = (u + (-12.0 * bc->b - 48.0 * bc->c)) * x;
        return (u + (8.0 * bc->b + 24.0 * bc->c)) / 6.0;
    }
    return 0.0;
}

typedef struct {
    double taps;
} sinc_ctx;

static inline double sinc_function(double x) {
    if (x == 0.0) {
        return 1.0;
    }
    x *= M_PI;
    return sin(x) / x;
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
        return ((x - 9.0 / 5.0) * x - 1.0 / 5.0) * x + 1.0;
    }
    if (x < 2.0) {
        x -= 1.0;
        return ((-1.0 / 3.0 * x + 4.0 / 5.0) * x - 7.0 / 15.0) * x;
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
    double taps, beta, i0_beta;
} kaiser_ctx;

// Based on: https://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision/
// Works correctly for x < 500.
// For x >= 500 it is necessary to split exp(x), but since in this case x <= 32, this branch is not implemented.
static inline double bessel_i0(double x) {
    // 2.38eps
    if (x < 7.75) {
        x /= 2.0;
        x *= x;
        double u = 1.1497640034400735733456400e-29;
        u = u * x + 2.0732014503197852176921968e-27;
        u = u * x + 5.9203280572170548134753422e-25;
        u = u * x + 1.3141332422663039834197910e-22;
        u = u * x + 2.5791926805873898803749321e-20;
        u = u * x + 4.3583591008893599099577755e-18;
        u = u * x + 6.2760839879536225394314453e-16;
        u = u * x + 7.5940582595094190098755663e-14;
        u = u * x + 7.5940584360755226536109511e-12;
        u = u * x + 6.1511873265092916275099070e-10;
        u = u * x + 3.9367598891475388547279760e-08;
        u = u * x + 1.9290123456788994104574754e-06;
        u = u * x + 6.9444444444444568581891535e-05;
        u = u * x + 1.7361111111111110294015271e-03;
        u = u * x + 2.7777777777777777805664954e-02;
        u = u * x + 2.4999999999999999999629693e-01;
        u = u * x + 1.0000000000000000000000801e+00;
        return u * x + 1.0;
    }
    // 0.72eps
    double x_rec = 1.0 / x;
    double u = 1.6069467093441596329340754e+16;
    u = u * x_rec + -2.1363029690365351606041265e+16;
    u = u * x_rec + 1.3012646806421079076251950e+16;
    u = u * x_rec + -4.8049082153027457378879746e+15;
    u = u * x_rec + 1.1989242681178569338129044e+15;
    u = u * x_rec + -2.1323049786724612220362154e+14;
    u = u * x_rec + 2.7752144774934763122129261e+13;
    u = u * x_rec + -2.6632742974569782078420204e+12;
    u = u * x_rec + 1.8592340458074104721496236e+11;
    u = u * x_rec + -8.9270060370015930749184222e+09;
    u = u * x_rec + 2.3518420447411254516178388e+08;
    u = u * x_rec + 2.6092888649549172879282592e+06;
    u = u * x_rec + -5.9355022509673600842060002e+05;
    u = u * x_rec + 3.1275740782277570164423916e+04;
    u = u * x_rec + -1.0026890180180668595066918e+03;
    u = u * x_rec + 2.2725199603010833194037016e+01;
    u = u * x_rec + -1.0699095472110916094973951e-01;
    u = u * x_rec + 9.4085204199017869159183831e-02;
    u = u * x_rec + 4.4718622769244715693031735e-02;
    u = u * x_rec + 2.9219501690198775910219311e-02;
    u = u * x_rec + 2.8050628884163787533196746e-02;
    u = u * x_rec + 4.9867785050353992900698488e-02;
    u = u * x_rec + 3.9894228040143265335649948e-01;
    return exp(x) * u / sqrt(x);
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
    double b, p, taps;
} gauss_ctx;

static double gauss_kernel(double x, void *ctx) {
    gauss_ctx *gs = (gauss_ctx *)ctx;
    if (x < 0.0) {
        x = -x;
    }
    if (x <= gs->taps) {
        return pow(gs->b, -gs->p * (x * x));
    }
    return 0.0;
}

// exp2(log2(x) * y); 0.5 ulp
// Based on: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
// All checks are removed because ffast-math is used.
// x = 0 returns 0, which satisfies the use case.
// x < 0 and y <= 0 contradict the conditions of the use case and are not processed.
static __m256 ffast_pow(__m256 x, __m256d y) {
    __m256i i = _mm256_castps_si256(x);
    __m256i exp = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_and_si256(i, _mm256_set1_epi32(0x7f800000)), 23), _mm256_set1_epi32(127));
    __m256 mant = _mm256_or_ps(_mm256_castsi256_ps(_mm256_and_si256(i, _mm256_set1_epi32(0x007fffff))), _mm256_set1_ps(1.0f));
    __m256d m0 = _mm256_cvtps_pd(_mm256_extractf128_ps(mant, 0));
    __m256d m1 = _mm256_cvtps_pd(_mm256_extractf128_ps(mant, 1));
    __m256d temp = _mm256_set1_pd(0.0029290200848161477), p0 = temp, p1 = temp;
    temp = _mm256_set1_pd(-0.048633183278885168), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(0.36573319077073368), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(-1.6454916622598008), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(4.9271656462993851), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(-10.331415641012363), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(15.540128043817649), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(-16.905304327891724), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(13.296487881230806), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(-7.6638533210657123), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(3.9049493931218771), p0 = _mm256_fmadd_pd(p0, m0, temp), p1 = _mm256_fmadd_pd(p1, m1, temp);
    temp = _mm256_set1_pd(1.0);
    p0 = _mm256_fmadd_pd(p0, _mm256_sub_pd(m0, temp), _mm256_cvtepi32_pd(_mm256_extractf128_ps(exp, 0)));
    p1 = _mm256_fmadd_pd(p1, _mm256_sub_pd(m1, temp), _mm256_cvtepi32_pd(_mm256_extractf128_ps(exp, 1)));
    p0 = _mm256_mul_pd(p0, y), p1 = _mm256_mul_pd(p1, y);
    temp = _mm256_set1_pd(127.0), p0 = _mm256_min_pd(p0, temp), p1 = _mm256_min_pd(p1, temp);
    temp = _mm256_set1_pd(-127.0), p0 = _mm256_max_pd(p0, temp), p1 = _mm256_max_pd(p1, temp);
    __m256d i0 = _mm256_floor_pd(p0), i1 = _mm256_floor_pd(p1);
    __m256d f0 = _mm256_sub_pd(p0, i0), f1 = _mm256_sub_pd(p1, i1);
    __m128i bias = _mm_set1_epi32(127);
    __m256d ei0 = _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm256_cvttpd_epi32(i0), bias), 23)));
    __m256d ei1 = _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm256_cvttpd_epi32(i1), bias), 23)));
    temp = _mm256_set1_pd(1.0150336705309649e-07), p0 = temp, p1 = temp;
    temp = _mm256_set1_pd(1.3259405609345135e-06), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(1.5252984838653427e-05), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(0.00015403434948071791), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(0.0013333557617604443), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(0.0096181291920672454), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(0.05550410866868561), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(0.24022650695649653), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(0.69314718055987101), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    temp = _mm256_set1_pd(1.0000000000000127), p0 = _mm256_fmadd_pd(p0, f0, temp), p1 = _mm256_fmadd_pd(p1, f1, temp);
    p0 = _mm256_mul_pd(ei0, p0), p1 = _mm256_mul_pd(ei1, p1);
    return _mm256_setr_m128(_mm256_cvtpd_ps(p0), _mm256_cvtpd_ps(p1));
}

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
    __m256d v_gamma = _mm256_set1_pd(d.gamma);
    __m256 v_div = _mm256_set1_ps(d.div);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    
    if (d.strict) {
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod8_w; x += 8) {
                __m256 pix = _mm256_load_ps(srcp + x);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = ffast_pow(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
                __m256 branch_1 = _mm256_div_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = ffast_pow(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
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
                __m256 branch_0 = ffast_pow(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
                __m256 branch_1 = _mm256_div_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GE_OQ);
                __m256 branch_0 = ffast_pow(_mm256_div_ps(_mm256_add_ps(pix_abs, v_corr), v_corr_one), v_gamma);
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
    __m256d v_gamma = _mm256_set1_pd(1.0 / d.gamma);
    __m256 v_div = _mm256_set1_ps(d.div);
    __m256 v_abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    
    if (d.strict) {
        for (int y = 0; y < src_h; y++) {
            int x = 0;
            for (; x < mod8_w; x += 8) {
                __m256 pix = _mm256_load_ps(srcp + x);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(ffast_pow(pix_abs, v_gamma), v_corr_one, v_corr);
                __m256 branch_1 = _mm256_mul_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GT_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(ffast_pow(pix_abs, v_gamma), v_corr_one, v_corr);
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
                __m256 branch_0 = _mm256_fmsub_ps(ffast_pow(pix_abs, v_gamma), v_corr_one, v_corr);
                __m256 branch_1 = _mm256_mul_ps(pix_abs, v_div);
                __m256 branch = _mm256_blendv_ps(branch_1, branch_0, mask_abs);
                _mm256_stream_ps(dstp + x, _mm256_or_ps(_mm256_andnot_ps(v_abs, pix), branch));
            }
            if (tail) {
                __m256 pix = _mm256_maskload_ps(srcp + x, tail_mask);
                __m256 pix_abs = _mm256_and_ps(pix, v_abs);
                __m256 mask_abs = _mm256_cmp_ps(pix_abs, v_thr, _CMP_GE_OQ);
                __m256 branch_0 = _mm256_fmsub_ps(ffast_pow(pix_abs, v_gamma), v_corr_one, v_corr);
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits, bool range UNUSED, bool chroma UNUSED
) {
    const uint8_t *restrict srcp = ptrs;
    uint16_t *restrict dstp = ptrd;
    int count = dst_bits - src_bits;
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits UNUSED, int dst_bits UNUSED, bool range, bool chroma
) {
    const uint8_t *restrict srcp = ptrs;
    float *restrict dstp = ptrd;
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits, bool range UNUSED, bool chroma UNUSED
) {
    const uint16_t *restrict srcp = ptrs;
    uint8_t *restrict dstp = ptrd;
    int tail = src_w % 16;
    int mod16_w = src_w - tail;
    
    int16_t mask_arr[16] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    int count = src_bits - dst_bits;
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits, bool range UNUSED, bool chroma UNUSED
) {
    const uint16_t *restrict srcp = ptrs;
    uint16_t *restrict dstp = ptrd;
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits, int dst_bits UNUSED, bool range, bool chroma
) {
    const uint16_t *restrict srcp = ptrs;
    float *restrict dstp = ptrd;
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int16_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m128i tail_mask = _mm_loadu_si128((__m128i *)mask_arr);
    
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps((float)((chroma ? 128 : 16) << (src_bits - 8)));
        v_high = _mm256_set1_ps((float)((chroma ? 224 : 219) << (src_bits - 8)));
    }
    else {
        v_low = _mm256_set1_ps(chroma ? (float)(128 << (src_bits - 8)) : 0.0f);
        v_high = _mm256_set1_ps(chroma ? (float)(1 << src_bits) : (float)((1 << src_bits) - 1));
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits UNUSED, int dst_bits UNUSED, bool range, bool chroma
) {
    const float *restrict srcp = ptrs;
    uint8_t *restrict dstp = ptrd;
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
    const void *restrict ptrs, void *restrict ptrd, ptrdiff_t src_stride, ptrdiff_t dst_stride,
    int src_w, int src_h, int src_bits UNUSED, int dst_bits, bool range, bool chroma
) {
    const float *restrict srcp = ptrs;
    uint16_t *restrict dstp = ptrd;
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256i v_max = _mm256_set1_epi32((1 << dst_bits) - 1);
    __m256 v_low, v_high;
    if (range) {
        v_low = _mm256_set1_ps(0.5f + (float)((chroma ? 128 : 16) << (dst_bits - 8)));
        v_high = _mm256_set1_ps((float)((chroma ? 224 : 219) << (dst_bits - 8)));
    }
    else {
        v_low = _mm256_set1_ps(chroma ? 0.5f + (float)(128 << (dst_bits - 8)) : 0.5f);
        v_high = _mm256_set1_ps(chroma ? (float)(1 << dst_bits) : (float)((1 << dst_bits) - 1));
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

static csr_t csr_get_weights(kernel_t kernel, int src_n, int dst_n, double start_n, double real_n) {
    double factor = dst_n / real_n;
    double scale = fmin(factor, 1.0);
    int min_n = (int)floor(start_n);
    int max_n = (int)ceil(real_n + start_n) - 1;
    int border = src_n - 1;
    double radius = kernel.radius / scale;
    int step = (int)ceil(radius * 2.0) + 2;
    double *weights = (double *)malloc(sizeof(double) * step * dst_n);
    int *col_idx = (int *)malloc(sizeof(int) * step * dst_n);
    int *row_ptr = (int *)malloc(sizeof(int) * (dst_n + 1));
    row_ptr[0] = 0;
    int nnz = 0;
    
    for (int i = 0; i < dst_n; i++) {
        double center = (i + 0.5) / factor - 0.5 + start_n;
        int low = VSMAX((int)floor(center - radius), min_n);
        int high = VSMIN((int)ceil(center + radius), max_n);
        double norm = 0.0;
        for (int j = low; j <= high; j++) {
            double temp_val = kernel.f((j - center) * scale, kernel.ctx);
            if (temp_val == 0.0) continue;
            norm += temp_val;
            int temp_idx = CLAMP(j, 0, border);
            if (row_ptr[i] != nnz && temp_idx == col_idx[nnz - 1]) {
                weights[nnz - 1] += temp_val;
                continue;
            }
            weights[nnz] = temp_val;
            col_idx[nnz] = temp_idx;
            nnz++;
        }
        for (int j = row_ptr[i]; j < nnz; j++) {
            weights[j] /= norm;
        }
        row_ptr[i + 1] = nnz;
    }
    
    return (csr_t){src_n, dst_n, nnz, weights, col_idx, row_ptr};
}

static void csr_free(csr_t csr) {
    free(csr.row_ptr);
    free(csr.col_idx);
    free(csr.values);
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
    int src_w, int src_h, int dst_w, csr_t weights
) {
    float *restrict src_buf = (float *)_mm_malloc(sizeof(float) * src_stride * 4, 64);
    float *restrict dst_buf = (float *)_mm_malloc(sizeof(float) * dst_stride * 4, 64);
    
    int tail = src_h % 4;
    int mod4_h = src_h - tail;
    
    for (int y = 0; y < mod4_h; y += 4) {
        transpose_block_into_buf(srcp, src_buf, src_stride, src_w);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = weights.row_ptr[x]; i < weights.row_ptr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + weights.col_idx[i] * 4);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
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
            for (int i = weights.row_ptr[x]; i < weights.row_ptr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + weights.col_idx[i] * 4);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm_store_ps(dst_buf + x * 4, _mm256_cvtpd_ps(v_acc));
        }
        transpose_block_from_buf_with_tail(dst_buf, dstp, dst_stride, dst_w, tail);
    }
    _mm_sfence();
    _mm_free(dst_buf);
    _mm_free(src_buf);
}

static void resize_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t dst_stride,
    int src_w, int src_h UNUSED, int dst_h, csr_t weights
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    for (int y = 0; y < dst_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = weights.row_ptr[y]; i < weights.row_ptr[y + 1]; i++) {
                __m256 pix = _mm256_load_ps(srcp + weights.col_idx[i] * dst_stride + x);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_stream_ps(dstp + x, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        if (tail) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = weights.row_ptr[y]; i < weights.row_ptr[y + 1]; i++) {
                __m256 pix = _mm256_maskload_ps(srcp + weights.col_idx[i] * dst_stride + x, tail_mask);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_stream_ps(dstp + x, _mm256_setr_m128(_mm256_cvtpd_ps(v_acc_0), _mm256_cvtpd_ps(v_acc_1)));
        }
        dstp += dst_stride;
    }
    _mm_sfence();
}

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
        
        bool range = !!vsapi->mapGetIntSaturated(props, "_ColorRange", 0, &err);
        if (bit_convert && !d->linear) {
            range = false;
        }
        else if (err) {
            range = (fi->colorFamily == cfRGB) ? false : true;
        }
        
        csr_t chroma_w, chroma_h;
        if (d->process_w && fi->subSamplingW) {
            int chroma_src_w = d->vi.width >> fi->subSamplingW;
            int chroma_dst_w = d->dst_width >> fi->subSamplingW;
            double start_w = d->start_w / (1 << fi->subSamplingW);
            double real_w = d->real_w / (1 << fi->subSamplingW);
            if (~chromaloc & 1) {// left allign
                double offset = ((1 << fi->subSamplingW) - 1) / 2.0;
                start_w += offset / (1 << fi->subSamplingW) - offset * real_w / d->dst_width;
            }
            chroma_w = csr_get_weights(d->kernel_w, chroma_src_w, chroma_dst_w, start_w, real_w);
        }
        else {
            chroma_w = (csr_t){0, 0, 0, NULL, NULL, NULL};
        }
        
        if (d->process_h && fi->subSamplingH) {
            int chroma_src_h = d->vi.height >> fi->subSamplingH;
            int chroma_dst_h = d->dst_height >> fi->subSamplingH;
            double start_h = d->start_h / (1 << fi->subSamplingH);
            double real_h = d->real_h / (1 << fi->subSamplingH);
            if (chromaloc & 2) {// top allign
                double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                start_h += offset / (1 << fi->subSamplingH) - offset * real_h / d->dst_height;
            }
            else if (chromaloc & 4) {// bottom allign
                double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                start_h -= offset / (1 << fi->subSamplingH) - offset * real_h / d->dst_height;
            }
            chroma_h = csr_get_weights(d->kernel_h, chroma_src_h, chroma_dst_h, start_h, real_h);
        }
        else {
            chroma_h = (csr_t){0, 0, 0, NULL, NULL, NULL};
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
            bool chroma = plane && (fi->colorFamily == cfYUV);
            bool sub_w = plane && fi->subSamplingW;
            bool sub_h = plane && fi->subSamplingH;
            
            if (bit_convert) {
                float *restrict bcup = (float *)vsapi->getWritePtr(bcu, plane);
                ptrdiff_t bcu_stride = vsapi->getStride(bcu, plane) / sizeof(float);
                d->conv_up(srcp, bcup, src_stride, bcu_stride, src_w, src_h, fi->bitsPerSample, 32, range, chroma);
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
                resize_width(srcp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, sub_w ? chroma_w : d->luma_w);
                resize_height(tmpp, dstp, dst_stride, dst_w, src_h, dst_h, sub_h ? chroma_h : d->luma_h);
            }
            else if (d->process_w) {
                resize_width(srcp, dstp, src_stride, dst_stride, src_w, src_h, dst_w, sub_w ? chroma_w : d->luma_w);
            }
            else if (d->process_h) {
                resize_height(srcp, dstp, dst_stride, dst_w, src_h, dst_h, sub_h ? chroma_h : d->luma_h);
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
                d->conv_down(bcdp, dstp, bcd_stride, dst_stride, dst_w, dst_h, 32, fi->bitsPerSample, range, chroma);
            }
        }
        vsapi->freeFrame(shr);
        vsapi->freeFrame(tmp);
        vsapi->freeFrame(gcr);
        vsapi->freeFrame(lin);
        vsapi->freeFrame(bcd);
        vsapi->freeFrame(bcu);
        vsapi->freeFrame(src);
        
        csr_free(chroma_h);
        csr_free(chroma_w);
        
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
    
    csr_free(d->luma_w);
    csr_free(d->luma_h);
    
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
            d.gamma = (GammaData){2.4, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
        }
        else {
            d.gamma = (GammaData){1.0 / 0.45, 0.081f, 0.018f, 0.099f, 4.5f, false};
        }
    }
    else if (!strcmp(gamma, "srgb")) {
        d.gamma = (GammaData){2.4, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
    }
    else if (!strcmp(gamma, "smpte170m")) {
        d.gamma = (GammaData){1.0 / 0.45, 0.081f, 0.018f, 0.099f, 4.5f, false};
    }
    else if (!strcmp(gamma, "adobe")) {
        d.gamma = (GammaData){2.19921875, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "dcip3")) {
        d.gamma = (GammaData){2.6, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "smpte240m")) {
        d.gamma = (GammaData){1.0 / 0.45, 0.0913f, 0.0228f, 0.1115f, 4.0f, false};
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
        sn->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3.0;
        }
        if (sn->taps < 1.0 || sn->taps > 128.0) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){lanczos_kernel, sn->taps, sn};
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
        sn->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3.0;
        }
        if (sn->taps < 1.0 || sn->taps > 128.0) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){blackman_kernel, sn->taps, sn};
    }
    else if (!strcmp(kernel, "nuttall")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3.0;
        }
        if (sn->taps < 1.0 || sn->taps > 128.0) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){nuttall_kernel, sn->taps, sn};
    }
    else if (!strcmp(kernel, "kaiser")) {
        kaiser_ctx *ks = (kaiser_ctx *)malloc(sizeof(*ks));
        ks->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            ks->taps = 3.0;
        }
        if (ks->taps < 1.0 || ks->taps > 128.0) {
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
        d.kernel_w = d.kernel_h = (kernel_t){kaiser_kernel, ks->taps, ks};
    }
    else if (!strcmp(kernel, "gauss")) {
        gauss_ctx *gs = (gauss_ctx *)malloc(sizeof(*gs));
        gs->b = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           gs->b = 2.0;
        }
        if (gs->b < 1.5 || gs->b > 3.5) {
            vsapi->mapSetError(out, "Resize: b must be between 1.5 and 3.5");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p = vsapi->mapGetFloat(in, "c", 0, &err);
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
        gs->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            gs->taps = 4.0;
        }
        if (gs->taps == 0.0) {
            gs->taps = sqrt(4.6 / (gs->p * log(gs->b)));
        }
        if (gs->taps < 0.6 || gs->taps > 128.0) {
            vsapi->mapSetError(out, "Resize: taps must be between 0.6 and 128 or 0 for automatic calculation");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){gauss_kernel, gs->taps, gs};
    }
    else {
        vsapi->mapSetError(out, "Resize: invalid kernel specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.process_w = (d.dst_width == d.vi.width && d.real_w == d.vi.width && d.start_w == 0.0) ? false : true;
    d.process_h = (d.dst_height == d.vi.height && d.real_h == d.vi.height && d.start_h == 0.0) ? false : true;
    
    if (d.process_w) {
        d.luma_w = csr_get_weights(d.kernel_w, d.vi.width, d.dst_width, d.start_w, d.real_w);
    }
    else {
        d.luma_w = (csr_t){0, 0, 0, NULL, NULL, NULL};
    }
    
    if (d.process_h) {
        d.luma_h = csr_get_weights(d.kernel_h, d.vi.height, d.dst_height, d.start_h, d.real_h);
    }
    else {
        d.luma_h = (csr_t){0, 0, 0, NULL, NULL, NULL};
    }
    
    if (d.vi.format.bytesPerSample == 1) {
        d.conv_up = uint8_to_float;
        d.conv_down = float_to_uint8;
    }
    else if (d.vi.format.bytesPerSample == 2) {
        d.conv_up = uint16_to_float;
        d.conv_down = float_to_uint16;
    }
    else {
        d.conv_up = NULL;
        d.conv_down = NULL;
    }
    
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
    csr_t luma_w, luma_h;
    banded_t luma_b_w, luma_b_h;
} DescaleData;

static csr_t csr_transpose(csr_t csr) {
    double *values = (double *)malloc(sizeof(double) * csr.nnz);
    int *col_idx = (int *)malloc(sizeof(int) * csr.nnz);
    int *row_ptr = (int *)malloc(sizeof(int) * (csr.col_n + 1));
    int *col_count = calloc(csr.col_n, sizeof(int));
    
    for (int i = 0; i < csr.nnz; i++) col_count[csr.col_idx[i]]++;
    
    row_ptr[0] = 0;
    for (int i = 0; i < csr.col_n; i++) row_ptr[i + 1] = row_ptr[i] + col_count[i];
    
    int *next = malloc(sizeof(int) * csr.col_n);
    memcpy(next, row_ptr, sizeof(int) * csr.col_n);
    
    for (int i = 0; i < csr.row_n; i++) {
        for (int j = csr.row_ptr[i]; j < csr.row_ptr[i + 1]; j++) {
            int dst = next[csr.col_idx[j]]++;
            col_idx[dst] = i;
            values[dst] = csr.values[j];
        }
    }
    
    free(next);
    free(col_count);
    
    return (csr_t){csr.row_n, csr.col_n, csr.nnz, values, col_idx, row_ptr};
}

static int get_ku_from_csr(csr_t csr) {
    int ku = 0;
    
    for (int i = 0; i < csr.row_n; i++) {
        int p0 = csr.row_ptr[i];
        int p1 = csr.row_ptr[i + 1];
        int c_min = csr.col_idx[p0];
        int c_max = csr.col_idx[p1 - 1];
        int w = c_max - c_min;
        if (w > ku) ku = w;
    }
    
    return ku;
}

static banded_t banded_gramian_from_csr(csr_t csr, double lambda) {
    int ku = get_ku_from_csr(csr);
    int kf = ku + 1;
    
    double *banded = (double *)calloc(csr.col_n * kf, sizeof(double));
    
    for (int i = 0; i < csr.row_n; i++) {
        int p0 = csr.row_ptr[i];
        int p1 = csr.row_ptr[i + 1];
        
        for (int j = p0; j < p1; j++) {
            int cj = csr.col_idx[j];
            double vj = csr.values[j];
            
            for (int k = j; k < p1; k++) {
                int ck = csr.col_idx[k];
                double vk = csr.values[k];
                int col = ku + cj - ck;
                banded[col * csr.col_n + ck] += vj * vk;
            }
        }
    }
    
    for (int i = csr.col_n * ku; i < csr.col_n * kf; i++) {
        banded[i] += lambda;
    }
    
    return (banded_t){csr.col_n, kf, ku, banded};
}

static void banded_free(banded_t banded) {
    free(banded.values);
}

static void banded_cholesky_from_gramian(banded_t banded) {
    for (int i = 0; i < banded.col_n; i++) {
        int j_start = VSMAX(i - banded.ku, 0);
        for (int j = j_start; j < i; j++) {
            double acc = 0.0;
            int k_start = VSMAX(j - banded.ku, j_start);
            for (int k = k_start; k < j; k++) {
                int idx_jk = (banded.ku + k - j) * banded.col_n + j;
                int idx_ik = (banded.ku + k - i) * banded.col_n + i;
                acc += banded.values[idx_jk] * banded.values[idx_ik];
            }
            int idx_ij = (banded.ku + j - i) * banded.col_n + i;
            int idx_jj = banded.ku * banded.col_n + j;
            banded.values[idx_ij] = (banded.values[idx_ij] - acc) / banded.values[idx_jj];
        }
        double acc = 0.0;
        for (int j = j_start; j < i; j++) {
            int idx_ik = (banded.ku + j - i) * banded.col_n + i;
            double lik = banded.values[idx_ik];
            acc += lik * lik;
        }
        int idx_ii = banded.ku * banded.col_n + i;
        banded.values[idx_ii] = sqrt(banded.values[idx_ii] - acc);
    }
}

static void solve_banded_cholesky_lane4(banded_t srcp, double *dstp) {
    for (int i = 0; i < srcp.col_n; i++) {
        int start = VSMAX(0, i - srcp.ku);
        __m256d v_acc = _mm256_load_pd(dstp + i * 4);
        for (int j = start; j < i; j++) {
            int row_in_src = srcp.ku + j - i;
            if (row_in_src >= 0) {
                __m256d pix = _mm256_load_pd(dstp + j * 4);
                __m256d v_weight = _mm256_set1_pd(srcp.values[row_in_src * srcp.col_n + i]);
                v_acc = _mm256_fnmadd_pd(pix, v_weight, v_acc);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp.values[srcp.ku * srcp.col_n + i]);
        _mm256_store_pd(dstp + i * 4, _mm256_div_pd(v_acc, v_div));
    }
    for (int i = srcp.col_n - 1; i >= 0; i--) {
        int end = VSMIN(srcp.col_n - 1, i + srcp.ku);
        __m256d v_acc = _mm256_load_pd(dstp + i * 4);
        for (int j = i + 1; j <= end; j++) {
            int row_in_src = srcp.ku + i - j;
            if (row_in_src >= 0) {
                __m256d pix = _mm256_load_pd(dstp + j * 4);
                __m256d v_weight = _mm256_set1_pd(srcp.values[row_in_src * srcp.col_n + j]);
                v_acc = _mm256_fnmadd_pd(pix, v_weight, v_acc);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp.values[srcp.ku * srcp.col_n + i]);
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
    int src_w, int src_h, int dst_w, csr_t weights, banded_t banded
) {
    int tail = src_h % 4;
    int mod4_h = src_h - tail;
    
    float *restrict src_buf = (float *)_mm_malloc(sizeof(float) * src_stride * 4, 64);
    double *restrict dst_buf = (double *)_mm_malloc(sizeof(double) * dst_stride * 4, 64);
    
    for (int y = 0; y < mod4_h; y += 4) {
        transpose_block_into_buf(srcp, src_buf, src_stride, src_w);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = weights.row_ptr[x]; i < weights.row_ptr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + weights.col_idx[i] * 4);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm256_store_pd(dst_buf + x * 4, v_acc);
        }
        solve_banded_cholesky_lane4(banded, dst_buf);
        transpose_double_block_from_buf(dst_buf, dstp, dst_stride, dst_w);
        dstp += dst_stride * 4;
        srcp += src_stride * 4;
    }
    if (tail) {
        transpose_block_into_buf_with_tail(srcp, src_buf, src_stride, src_w, tail);
        for (int x = 0; x < dst_w; x++) {
            __m256d v_acc = _mm256_setzero_pd();
            for (int i = weights.row_ptr[x]; i < weights.row_ptr[x + 1]; i++) {
                __m128 pix = _mm_load_ps(src_buf + weights.col_idx[i] * 4);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc = _mm256_fmadd_pd(_mm256_cvtps_pd(pix), v_weight, v_acc);
            }
            _mm256_store_pd(dst_buf + x * 4, v_acc);
        }
        solve_banded_cholesky_lane4(banded, dst_buf);
        transpose_double_block_from_buf_with_tail(dst_buf, dstp, dst_stride, dst_w, tail);
    }
    _mm_sfence();
    _mm_free(dst_buf);
    _mm_free(src_buf);
}

static void solve_banded_cholesky_lane8(banded_t srcp, double *dstp) {
    for (int i = 0; i < srcp.col_n; i++) {
        int start = VSMAX(0, i - srcp.ku);
        __m256d v_acc_0 = _mm256_load_pd(dstp + i * 8 + 0);
        __m256d v_acc_1 = _mm256_load_pd(dstp + i * 8 + 4);
        for (int j = start; j < i; j++) {
            int row_in_src = srcp.ku + j - i;
            if (row_in_src >= 0) {
                __m256d pix_0 = _mm256_load_pd(dstp + j * 8 + 0);
                __m256d pix_1 = _mm256_load_pd(dstp + j * 8 + 4);
                __m256d v_weight = _mm256_set1_pd(srcp.values[row_in_src * srcp.col_n + i]);
                v_acc_0 = _mm256_fnmadd_pd(pix_0, v_weight, v_acc_0);
                v_acc_1 = _mm256_fnmadd_pd(pix_1, v_weight, v_acc_1);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp.values[srcp.ku * srcp.col_n + i]);
        _mm256_store_pd(dstp + i * 8 + 0, _mm256_div_pd(v_acc_0, v_div));
        _mm256_store_pd(dstp + i * 8 + 4, _mm256_div_pd(v_acc_1, v_div));
    }
    for (int i = srcp.col_n - 1; i >= 0; i--) {
        int end = VSMIN(srcp.col_n - 1, i + srcp.ku);
        __m256d v_acc_0 = _mm256_load_pd(dstp + i * 8 + 0);
        __m256d v_acc_1 = _mm256_load_pd(dstp + i * 8 + 4);
        for (int j = i + 1; j <= end; j++) {
            int row_in_src = srcp.ku + i - j;
            if (row_in_src >= 0) {
                __m256d pix_0 = _mm256_load_pd(dstp + j * 8 + 0);
                __m256d pix_1 = _mm256_load_pd(dstp + j * 8 + 4);
                __m256d v_weight = _mm256_set1_pd(srcp.values[row_in_src * srcp.col_n + j]);
                v_acc_0 = _mm256_fnmadd_pd(pix_0, v_weight, v_acc_0);
                v_acc_1 = _mm256_fnmadd_pd(pix_1, v_weight, v_acc_1);
            }
        }
        __m256d v_div = _mm256_set1_pd(srcp.values[srcp.ku * srcp.col_n + i]);
        _mm256_store_pd(dstp + i * 8 + 0, _mm256_div_pd(v_acc_0, v_div));
        _mm256_store_pd(dstp + i * 8 + 4, _mm256_div_pd(v_acc_1, v_div));
    }
}

static void descale_height(
    const float *restrict srcp, float *restrict dstp, ptrdiff_t src_stride,
    int src_w, int src_h UNUSED, int dst_h, csr_t weights, banded_t banded
) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tail_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    double *restrict dst_buf = (double *)_mm_malloc(sizeof(double) * dst_h * 8, 64);
    
    for (int x = 0; x < mod8_w; x += 8) {
        for (int y = 0; y < dst_h; y++) {
            __m256d v_acc_0 = _mm256_setzero_pd();
            __m256d v_acc_1 = _mm256_setzero_pd();
            for (int i = weights.row_ptr[y]; i < weights.row_ptr[y + 1]; i++) {
                __m256 pix = _mm256_load_ps(srcp + weights.col_idx[i] * src_stride);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_store_pd(dst_buf + y * 8 + 0, v_acc_0);
            _mm256_store_pd(dst_buf + y * 8 + 4, v_acc_1);
        }
        solve_banded_cholesky_lane8(banded, dst_buf);
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
            for (int i = weights.row_ptr[y]; i < weights.row_ptr[y + 1]; i++) {
                __m256 pix = _mm256_maskload_ps(srcp + weights.col_idx[i] * src_stride, tail_mask);
                __m256d v_weight = _mm256_set1_pd(weights.values[i]);
                v_acc_0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 0)), v_weight, v_acc_0);
                v_acc_1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(pix, 1)), v_weight, v_acc_1);
            }
            _mm256_store_pd(dst_buf + y * 8 + 0, v_acc_0);
            _mm256_store_pd(dst_buf + y * 8 + 4, v_acc_1);
        }
        solve_banded_cholesky_lane8(banded, dst_buf);
        for (int y = 0; y < dst_h; y++) {
            __m128 pix0 = _mm256_cvtpd_ps(_mm256_load_pd(dst_buf + y * 8 + 0));
            __m128 pix1 = _mm256_cvtpd_ps(_mm256_load_pd(dst_buf + y * 8 + 4));
            _mm256_stream_ps(dstp + y * src_stride, _mm256_setr_m128(pix0, pix1));
        }
    }
    _mm_sfence();
    _mm_free(dst_buf);
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
        
        csr_t chroma_w, chroma_h;
        banded_t chroma_b_w, chroma_b_h;
        if (d->process_w && fi->subSamplingW) {
            int chroma_src_w = d->vi.width >> fi->subSamplingW;
            int chroma_dst_w = d->dst_width >> fi->subSamplingW;
            double start_w = d->start_w / (1 << fi->subSamplingW);
            double real_w = d->real_w / (1 << fi->subSamplingW);
            if (~chromaloc & 1) {// left allign
                double offset = ((1 << fi->subSamplingW) - 1) / 2.0;
                start_w += offset / (1 << fi->subSamplingW) - offset * real_w / d->vi.width;
            }
            csr_t temp = csr_get_weights(d->kernel_w, chroma_dst_w, chroma_src_w, start_w, real_w);
            chroma_w = csr_transpose(temp);
            chroma_b_w = banded_gramian_from_csr(temp, d->lambda);
            banded_cholesky_from_gramian(chroma_b_w);
            csr_free(temp);
        }
        else {
            chroma_w = (csr_t){0, 0, 0, NULL, NULL, NULL};
            chroma_b_w = (banded_t){0, 0, 0, NULL};
        }
        
        if (d->process_h && fi->subSamplingH) {
            int chroma_src_h = d->vi.height >> fi->subSamplingH;
            int chroma_dst_h = d->dst_height >> fi->subSamplingH;
            double start_h = d->start_h / (1 << fi->subSamplingH);
            double real_h = d->real_h / (1 << fi->subSamplingH);
            if (chromaloc & 2) {// top allign
                double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                start_h += offset / (1 << fi->subSamplingH) - offset * real_h / d->vi.height;
            }
            else if (chromaloc & 4) {// bottom allign
                double offset = ((1 << fi->subSamplingH) - 1) / 2.0;
                start_h -= offset / (1 << fi->subSamplingH) - offset * real_h / d->vi.height;
            }
            csr_t temp = csr_get_weights(d->kernel_h, chroma_dst_h, chroma_src_h, start_h, real_h);
            chroma_h = csr_transpose(temp);
            chroma_b_h = banded_gramian_from_csr(temp, d->lambda);
            banded_cholesky_from_gramian(chroma_b_h);
            csr_free(temp);
        }
        else {
            chroma_h = (csr_t){0, 0, 0, NULL, NULL, NULL};
            chroma_b_h = (banded_t){0, 0, 0, NULL};
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
            bool sub_w = plane && fi->subSamplingW;
            bool sub_h = plane && fi->subSamplingH;
            
            if (d->process_w && d->process_h) {
                float *restrict tmpp = (float *)vsapi->getWritePtr(tmp, plane);
                descale_width(srcp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, sub_w ? chroma_w : d->luma_w, sub_w ? chroma_b_w : d->luma_b_w);
                descale_height(tmpp, dstp, dst_stride, dst_w, src_h, dst_h, sub_h ? chroma_h : d->luma_h, sub_h ? chroma_b_h : d->luma_b_h);
            }
            else if (d->process_w) {
                descale_width(srcp, dstp, src_stride, dst_stride, src_w, src_h, dst_w, sub_w ? chroma_w : d->luma_w, sub_w ? chroma_b_w : d->luma_b_w);
            }
            else if (d->process_h) {
                descale_height(srcp, dstp, dst_stride, dst_w, src_h, dst_h, sub_h ? chroma_h : d->luma_h, sub_h ? chroma_b_h : d->luma_b_h);
            }
            else {
                vsh_bitblt(dstp, sizeof(float) * dst_stride, srcp, sizeof(float) * src_stride, sizeof(float) * src_w, src_h);
            }
        }
        
        vsapi->freeFrame(tmp);
        vsapi->freeFrame(src);
        
        banded_free(chroma_b_h);
        banded_free(chroma_b_w);
        csr_free(chroma_h);
        csr_free(chroma_w);
        
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
    
    csr_free(d->luma_w);
    csr_free(d->luma_h);
    banded_free(d->luma_b_w);
    banded_free(d->luma_b_h);
    
    free(d);
}

static void VS_CC DescaleCreate(const VSMap *in, VSMap *out, void *userData UNUSED, VSCore *core, const VSAPI *vsapi) {
    DescaleData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "Descale: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.dst_width = vsapi->mapGetIntSaturated(in, "width", 0, NULL);
    d.dst_height = vsapi->mapGetIntSaturated(in, "height", 0, NULL);
    
    if (d.dst_width <= 1 << d.vi.format.subSamplingW || d.dst_width > d.vi.width) {
        vsapi->mapSetError(out, "Descale: \"width\" any of the planes must be greater than 1 and less than or equal to source width");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_width % (1 << d.vi.format.subSamplingW)) {
        vsapi->mapSetError(out, "Descale: \"width\" must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > d.vi.height) {
        vsapi->mapSetError(out, "Descale: \"height\" any of the planes must be greater than 1 and less than or equal to source height");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height % (1 << d.vi.format.subSamplingH)) {
        vsapi->mapSetError(out, "Descale: \"height\" must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.start_w = vsapi->mapGetFloat(in, "src_left", 0, &err);
    if (err) {
        d.start_w = 0.0;
    }
    
    if (d.start_w <= -d.dst_width || d.start_w >= d.dst_width) {
        vsapi->mapSetError(out, "Descale: \"src_left\" must be between \"-width\" and \"width\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.start_h = vsapi->mapGetFloat(in, "src_top", 0, &err);
    if (err) {
        d.start_h = 0.0;
    }
    
    if (d.start_h <= -d.dst_height || d.start_h >= d.dst_height) {
        vsapi->mapSetError(out, "Descale: \"src_top\" must be between \"-height\" and \"height\"");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.real_w = vsapi->mapGetFloat(in, "src_width", 0, &err);
    if (err) {
        d.real_w = (double)d.dst_width;
    }
    
    if (d.real_w <= -d.dst_width + d.start_w || d.real_w >= d.dst_width * 2 - d.start_w) {
        vsapi->mapSetError(out, "Descale: \"src_width\" must be between \"-width + src_left\" and \"width * 2 - src_left\"");
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
        vsapi->mapSetError(out, "Descale: \"src_height\" must be between \"-height + src_top\" and \"height * 2 - src_top\"");
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
    
    if (d.lambda < 1e-16 || d.lambda >= 1.0) {
        vsapi->mapSetError(out, "Descale: \"lambda\" must be between 1e-16 and 1");
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
        sn->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3.0;
        }
        if (sn->taps < 1.0 || sn->taps > 128.0) {
            vsapi->mapSetError(out, "Descale: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){lanczos_kernel, sn->taps, sn};
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
        sn->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3.0;
        }
        if (sn->taps < 1.0 || sn->taps > 128.0) {
            vsapi->mapSetError(out, "Descale: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){blackman_kernel, sn->taps, sn};
    }
    else if (!strcmp(kernel, "nuttall")) {
        sinc_ctx *sn = (sinc_ctx *)malloc(sizeof(*sn));
        sn->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            sn->taps = 3.0;
        }
        if (sn->taps < 1.0 || sn->taps > 128.0) {
            vsapi->mapSetError(out, "Descale: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(sn);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){nuttall_kernel, sn->taps, sn};
    }
    else if (!strcmp(kernel, "kaiser")) {
        kaiser_ctx *ks = (kaiser_ctx *)malloc(sizeof(*ks));
        ks->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            ks->taps = 3.0;
        }
        if (ks->taps < 1.0 || ks->taps > 128.0) {
            vsapi->mapSetError(out, "Descale: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(ks);
            return;
        }
        ks->beta = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           ks->beta = 4.0;
        }
        if (ks->beta <= 0.0 || ks->beta > 32.0) {
            vsapi->mapSetError(out, "Descale: beta must be between 0 and 32");
            vsapi->freeNode(d.node);
            free(ks);
            return;
        }
        ks->i0_beta = bessel_i0(ks->beta);
        d.kernel_w = d.kernel_h = (kernel_t){kaiser_kernel, ks->taps, ks};
    }
    else if (!strcmp(kernel, "gauss")) {
        gauss_ctx *gs = (gauss_ctx *)malloc(sizeof(*gs));
        gs->b = vsapi->mapGetFloat(in, "b", 0, &err);
        if (err) {
           gs->b = 2.0;
        }
        if (gs->b < 1.5 || gs->b > 3.5) {
            vsapi->mapSetError(out, "Descale: b must be between 1.5 and 3.5");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p = vsapi->mapGetFloat(in, "c", 0, &err);
        if (err) {
           gs->p = 30.0;
        }
        if (gs->p < 1.0 || gs->p > 100.0) {
            vsapi->mapSetError(out, "Descale: p must be between 1 and 100");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        gs->p /= 10.0;
        gs->taps = vsapi->mapGetFloat(in, "taps", 0, &err);
        if (err) {
            gs->taps = 4.0;
        }
        if (gs->taps == 0.0) {
            gs->taps = sqrt(4.6 / (gs->p * log(gs->b)));
        }
        if (gs->taps < 0.6 || gs->taps > 128.0) {
            vsapi->mapSetError(out, "Descale: taps must be between 0.6 and 128 or 0 for automatic calculation");
            vsapi->freeNode(d.node);
            free(gs);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){gauss_kernel, gs->taps, gs};
    }
    else {
        vsapi->mapSetError(out, "Descale: invalid kernel specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.process_w = (d.dst_width == d.vi.width && d.real_w == d.dst_width && d.start_w == 0.0) ? false : true;
    d.process_h = (d.dst_height == d.vi.height && d.real_h == d.dst_height && d.start_h == 0.0) ? false : true;
    
    if (d.process_w) {
        csr_t temp = csr_get_weights(d.kernel_w, d.dst_width, d.vi.width, d.start_w, d.real_w);
        d.luma_w = csr_transpose(temp);
        d.luma_b_w = banded_gramian_from_csr(temp, d.lambda);
        banded_cholesky_from_gramian(d.luma_b_w);
        csr_free(temp);
    }
    else {
        d.luma_w = (csr_t){0, 0, 0, NULL, NULL, NULL};
        d.luma_b_w = (banded_t){0, 0, 0, NULL};
    }
    
    if (d.process_h) {
        csr_t temp = csr_get_weights(d.kernel_h, d.dst_height, d.vi.height, d.start_h, d.real_h);
        d.luma_h = csr_transpose(temp);
        d.luma_b_h = banded_gramian_from_csr(temp, d.lambda);
        banded_cholesky_from_gramian(d.luma_b_h);
        csr_free(temp);
    }
    else {
        d.luma_h = (csr_t){0, 0, 0, NULL, NULL, NULL};
        d.luma_b_h = (banded_t){0, 0, 0, NULL};
    }
    
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
            d.gamma = (GammaData){2.4, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
        }
        else {
            d.gamma = (GammaData){1.0 / 0.45, 0.081f, 0.018f, 0.099f, 4.5f, false};
        }
    }
    else if (!strcmp(gamma, "srgb")) {
        d.gamma = (GammaData){2.4, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
    }
    else if (!strcmp(gamma, "smpte170m")) {
        d.gamma = (GammaData){1.0 / 0.45, 0.081f, 0.018f, 0.099f, 4.5f, false};
    }
    else if (!strcmp(gamma, "adobe")) {
        d.gamma = (GammaData){2.19921875, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "dcip3")) {
        d.gamma = (GammaData){2.6, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "smpte240m")) {
        d.gamma = (GammaData){1.0 / 0.45, 0.0913f, 0.0228f, 0.1115f, 4.0f, false};
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
            d.gamma = (GammaData){2.4, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
        }
        else {
            d.gamma = (GammaData){1.0 / 0.45, 0.081f, 0.018f, 0.099f, 4.5f, false};
        }
    }
    else if (!strcmp(gamma, "srgb")) {
        d.gamma = (GammaData){2.4, 0.04045f, 0.0031308f, 0.055f, 12.92f, true};
    }
    else if (!strcmp(gamma, "smpte170m")) {
        d.gamma = (GammaData){1.0 / 0.45, 0.081f, 0.018f, 0.099f, 4.5f, false};
    }
    else if (!strcmp(gamma, "adobe")) {
        d.gamma = (GammaData){2.19921875, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "dcip3")) {
        d.gamma = (GammaData){2.6, 0.0f, 0.0f, 0.0f, 1.0f, false};
    }
    else if (!strcmp(gamma, "smpte240m")) {
        d.gamma = (GammaData){1.0 / 0.45, 0.0913f, 0.0228f, 0.1115f, 4.0f, false};
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
    convert_func f;
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
        bool range = !!vsapi->mapGetIntSaturated(props, "_ColorRange", 0, &err);
        if (d->direct) {
            range = false;
        }
        else if (err) {
            range = (fi->colorFamily == cfRGB) ? false : true;
        }
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const void *restrict srcp = (const void *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / fi->bytesPerSample;
            void *restrict dstp = (void *)vsapi->getWritePtr(dst, plane);
            ptrdiff_t dst_stride = vsapi->getStride(dst, plane) / d->vi.format.bytesPerSample;
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            bool chroma = plane && (fi->colorFamily == cfYUV);
            
            d->f(srcp, dstp, src_stride, dst_stride, src_w, src_h, fi->bitsPerSample, d->vi.format.bitsPerSample, range, chroma);
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
    
    int bytes = (bits + 7) / 8;
    
    if ((d.vi.format.bytesPerSample == 1) && (bytes == 4)) {
        d.f = uint8_to_float;
    }
    else if ((d.vi.format.bytesPerSample == 2) && (bytes == 4)) {
        d.f = uint16_to_float;
    }
    else if ((d.vi.format.bytesPerSample == 4) && (bytes == 1)) {
        d.f = float_to_uint8;
    }
    else if ((d.vi.format.bytesPerSample == 4) && (bytes == 2)) {
        d.f = float_to_uint16;
    }
    else if ((d.vi.format.bytesPerSample == 1) && (bytes == 2)) {
        d.f = uint8_to_uint16;
    }
    else if ((d.vi.format.bytesPerSample == 2) && (bytes == 1)) {
        d.f = uint16_to_uint8;
    }
    else {
        d.f = uint16_to_uint16;
    }
    
    int err;
    d.direct = !!vsapi->mapGetIntSaturated(in, "direct", 0, &err);
    
    if (err) {
        d.direct = false;
    }
    
    d.vi.format.bitsPerSample = bits;
    d.vi.format.bytesPerSample = bytes;
    d.vi.format.sampleType = (bits == 32) ? stFloat : stInteger;
    
    BitDepthData *data = (BitDepthData *)malloc(sizeof d);
    *data = d;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "BitDepth", &d.vi, BitDepthGetFrame, BitDepthFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("ru.artyfox.plugins", "artyfox", "A disjointed set of filters", VS_MAKE_VERSION(16, 1), VAPOURSYNTH_API_VERSION, 0, plugin);
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
                             "taps:float:opt;"
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
                             "taps:float:opt;"
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
