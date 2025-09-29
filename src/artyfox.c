#include <stdbool.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#define CLAMP(x, min, max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : (x))) 
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define M_PI 3.14159265358979323846

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

static void rgb_to_linear(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_p_0_04045 = _mm256_set1_ps(0.04045f);
    __m256 v_n_0_04045 = _mm256_set1_ps(-0.04045f);
    __m256 v_p_0_055 = _mm256_set1_ps(0.055f);
    __m256 v_p_1_055 = _mm256_set1_ps(1.055f);
    __m256 v_gamma = _mm256_set1_ps(gamma);
    __m256 v_p_12_92 = _mm256_set1_ps(12.92f);
    __m256 mask_p = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 mask_n = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_04045, _CMP_GT_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_04045, _CMP_LT_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 base = _mm256_div_ps(_mm256_add_ps(abs_x, v_p_0_055), v_p_1_055);
            __m256 branch_0 = _mm256_pow_ps(base, v_gamma);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_div_ps(pix, v_p_12_92);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_stream_ps(dstp + x, branch_0_1_2);
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_04045, _CMP_GT_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_04045, _CMP_LT_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 base = _mm256_div_ps(_mm256_add_ps(abs_x, v_p_0_055), v_p_1_055);
            __m256 branch_0 = _mm256_pow_ps(base, v_gamma);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_div_ps(pix, v_p_12_92);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_stream_ps(dstp + x, branch_0_1_2);
        }
        // for (; x < src_w; x++) {
        //     if (srcp[x] > 0.04045f) {
        //         dstp[x] = powf((srcp[x] + 0.055f) / 1.055f, gamma);
        //     }
        //     else if (srcp[x] < -0.04045f) {
        //         dstp[x] = -powf((-srcp[x] + 0.055f) / 1.055f, gamma);
        //     }
        //     else {
        //         dstp[x] = srcp[x] / 12.92f;
        //     }
        // }
        srcp += stride;
        dstp += stride;
    }
}

static void yuv_to_linear(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_p_0_081 = _mm256_set1_ps(0.081f);
    __m256 v_n_0_081 = _mm256_set1_ps(-0.081f);
    __m256 v_p_0_099 = _mm256_set1_ps(0.099f);
    __m256 v_p_1_099 = _mm256_set1_ps(1.099f);
    __m256 v_gamma = _mm256_set1_ps(gamma);
    __m256 v_p_4_5 = _mm256_set1_ps(4.5f);
    __m256 mask_p = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 mask_n = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_081, _CMP_GE_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_081, _CMP_LE_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 base = _mm256_div_ps(_mm256_add_ps(abs_x, v_p_0_099), v_p_1_099);
            __m256 branch_0 = _mm256_pow_ps(base, v_gamma);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_div_ps(pix, v_p_4_5);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_stream_ps(dstp + x, branch_0_1_2);
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_081, _CMP_GE_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_081, _CMP_LE_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 base = _mm256_div_ps(_mm256_add_ps(abs_x, v_p_0_099), v_p_1_099);
            __m256 branch_0 = _mm256_pow_ps(base, v_gamma);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_div_ps(pix, v_p_4_5);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_stream_ps(dstp + x, branch_0_1_2);
        }
        // for (; x < src_w; x++) {
        //     if (srcp[x] >= 0.081f) {
        //         dstp[x] = powf((srcp[x] + 0.099f) / 1.099f, gamma);
        //     }
        //     else if (srcp[x] <= -0.081f) {
        //         dstp[x] = -powf((-srcp[x] + 0.099f) / 1.099f, gamma);
        //     }
        //     else {
        //         dstp[x] = srcp[x] / 4.5f;
        //     }
        // }
        srcp += stride;
        dstp += stride;
    }
}

static void resize_width(
    const float *srcp, float *dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride, int src_w, int src_h, int dst_w,
    double start_w, double real_w, kernel_t kernel
) {
    double factor = dst_w / real_w;
    double scale = (factor < 1.0) ? factor : 1.0;
    double **weights = (double **)malloc(sizeof(double *) * dst_w);
    int *low = (int *)malloc(sizeof(int) * dst_w);
    int *high = (int *)malloc(sizeof(int) * dst_w);
    int min_w = (int)floor(start_w);
    int max_w = (int)ceil(real_w + start_w) - 1;
    int border = src_w - 1;
    double radius = kernel.radius / scale;
    
    for (int x = 0; x < dst_w; x++) {
        double center = (x + 0.5) / factor - 0.5 + start_w;
        low[x] = MAX((int)floor(center - radius), min_w);
        high[x] = MIN((int)ceil(center + radius), max_w);
        weights[x] = (double *)malloc(sizeof(double) * (high[x] - low[x] + 1));
        double norm = 0.0;
        for (int i = low[x]; i <= high[x]; i++) {
            weights[x][i - low[x]] = kernel.f((i - center) * scale, kernel.ctx);
            norm += weights[x][i - low[x]];
        }
        for (int i = low[x]; i <= high[x]; i++) {
            weights[x][i - low[x]] /= norm;
        }
    }
    
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            double acc = 0.0;
            for (int i = low[x]; i <= high[x]; i++) {
                acc += srcp[CLAMP(i, 0, border)] * weights[x][i - low[x]];
            }
            dstp[x] = (float)acc;
        }
        dstp += dst_stride;
        srcp += src_stride;
    }
    
    for (int x = 0; x < dst_w; x++) {
        free(weights[x]);
    }
    free(weights);
    free(low);
    free(high);
}

static void resize_height(
    const float *srcp, float *dstp, ptrdiff_t dst_stride, int src_w, int src_h, int dst_h,
    double start_h, double real_h, kernel_t kernel
) {
    double factor = dst_h / real_h;
    double scale = (factor < 1.0) ? factor : 1.0;
    int min_h = (int)floor(start_h);
    int max_h = (int)ceil(real_h + start_h) - 1;
    int border = src_h - 1;
    double radius = kernel.radius / scale;
    
    for (int y = 0; y < dst_h; y++) {
        double center = (y + 0.5) / factor - 0.5 + start_h;
        int low = MAX((int)floor(center - radius), min_h);
        int high = MIN((int)ceil(center + radius), max_h);
        double *weights = (double *)malloc(sizeof(double) * (high - low + 1));
        double norm = 0.0;
        for (int i = low; i <= high; i++) {
            weights[i - low] = kernel.f((i - center) * scale, kernel.ctx);
            norm += weights[i - low];
        }
        for (int i = low; i <= high; i++) {
            weights[i - low] /= norm;
        }
        for (int x = 0; x < src_w; x++) {
            double acc = 0.0;
            for (int i = low; i <= high; i++) {
                acc += srcp[CLAMP(i, 0, border) * dst_stride + x] * weights[i - low];
            }
            dstp[x] = (float)acc;
        }
        dstp += dst_stride;
        free(weights);
    }
}

static void linear_to_rgb(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_p_0_031308 = _mm256_set1_ps(0.0031308f);
    __m256 v_n_0_031308 = _mm256_set1_ps(-0.0031308f);
    __m256 v_p_0_055 = _mm256_set1_ps(0.055f);
    __m256 v_p_1_055 = _mm256_set1_ps(1.055f);
    __m256 v_gamma = _mm256_set1_ps(1.0f / gamma);
    __m256 v_p_12_92 = _mm256_set1_ps(12.92f);
    __m256 mask_p = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 mask_n = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_031308, _CMP_GT_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_031308, _CMP_LT_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 powered = _mm256_pow_ps(abs_x, v_gamma);
            __m256 branch_0 = _mm256_sub_ps(_mm256_mul_ps(powered, v_p_1_055), v_p_0_055);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_mul_ps(pix, v_p_12_92);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_store_ps(dstp + x, branch_0_1_2);
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_031308, _CMP_GT_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_031308, _CMP_LT_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 powered = _mm256_pow_ps(abs_x, v_gamma);
            __m256 branch_0 = _mm256_sub_ps(_mm256_mul_ps(powered, v_p_1_055), v_p_0_055);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_mul_ps(pix, v_p_12_92);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_store_ps(dstp + x, branch_0_1_2);
        }
        // for (; x < src_w; x++) {
        //     if (srcp[x] > 0.0031308f) {
        //         dstp[x] = powf(srcp[x], 1.0f / gamma) * 1.055f - 0.055f;
        //     }
        //     else if (srcp[x] < -0.0031308f) {
        //         dstp[x] = powf(-srcp[x], 1.0f / gamma) * -1.055f + 0.055f;
        //     }
        //     else {
        //         dstp[x] = srcp[x] * 12.92f;
        //     }
        // }
        srcp += stride;
        dstp += stride;
    }
}

static void linear_to_yuv(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
    int tail = src_w % 8;
    int mod8_w = src_w - tail;
    
    int32_t mask_arr[8] = {0};
    for (int i = 0; i < tail; i++) mask_arr[i] = -1;
    __m256i tale_mask = _mm256_loadu_si256((__m256i *)mask_arr);
    
    __m256 v_p_0_018 = _mm256_set1_ps(0.018f);
    __m256 v_n_0_018 = _mm256_set1_ps(-0.018f);
    __m256 v_p_0_099 = _mm256_set1_ps(0.099f);
    __m256 v_p_1_099 = _mm256_set1_ps(1.099f);
    __m256 v_gamma = _mm256_set1_ps(1.0f / gamma);
    __m256 v_p_4_5 = _mm256_set1_ps(4.5f);
    __m256 mask_p = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 mask_n = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    for (int y = 0; y < src_h; y++) {
        int x = 0;
        for (; x < mod8_w; x += 8) {
            __m256 pix = _mm256_load_ps(srcp + x);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_018, _CMP_GE_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_018, _CMP_LE_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 powered = _mm256_pow_ps(abs_x, v_gamma);
            __m256 branch_0 = _mm256_sub_ps(_mm256_mul_ps(powered, v_p_1_099), v_p_0_099);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_mul_ps(pix, v_p_4_5);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_store_ps(dstp + x, branch_0_1_2);
        }
        if (tail) {
            __m256 pix = _mm256_maskload_ps(srcp + x, tale_mask);
            __m256 mask_0 = _mm256_cmp_ps(pix, v_p_0_018, _CMP_GE_OQ);
            __m256 mask_1 = _mm256_cmp_ps(pix, v_n_0_018, _CMP_LE_OQ);
            __m256 abs_x = _mm256_and_ps(pix, mask_p);
            __m256 powered = _mm256_pow_ps(abs_x, v_gamma);
            __m256 branch_0 = _mm256_sub_ps(_mm256_mul_ps(powered, v_p_1_099), v_p_0_099);
            __m256 branch_1 = _mm256_or_ps(branch_0, mask_n);
            __m256 branch_2 = _mm256_mul_ps(pix, v_p_4_5);
            __m256 branch_0_2 = _mm256_blendv_ps(branch_2, branch_0, mask_0);
            __m256 branch_0_1_2 = _mm256_blendv_ps(branch_0_2, branch_1, mask_1);
            _mm256_store_ps(dstp + x, branch_0_1_2);
        }
        // for (; x < src_w; x++) {
        //     if (srcp[x] >= 0.018f) {
        //         dstp[x] = powf(srcp[x], 1.0f / gamma) * 1.099f - 0.099f;
        //     }
        //     else if (srcp[x] <= -0.018f) {
        //         dstp[x] = powf(-srcp[x], 1.0f / gamma) * -1.099f + 0.099f;
        //     }
        //     else {
        //         dstp[x] = srcp[x] * 4.5f;
        //     }
        // }
        srcp += stride;
        dstp += stride;
    }
}

static void sharp_width(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float sharp) {
    int border = src_w - 1;
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = srcp[x] * sharp + (1.0f - sharp) * (srcp[MAX(x - 1, 0)] + srcp[x] + srcp[MIN(x + 1, border)]) / 3.0f;
        }
        dstp += stride;
        srcp += stride;
    }
}

static void sharp_height(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float sharp) {
    int border = src_h - 1;
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = srcp[x] * sharp + (1.0f - sharp) * (srcp[(y > 0) ? (x - stride) : x] + srcp[x] + srcp[(y < border) ? (x + stride) : x]) / 3.0f;
        }
        dstp += stride;
        srcp += stride;
    }
}

static const VSFrame *VS_CC ResizeGetFrame(
    int n, int activationReason, void *instanceData, void **frameData,
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
        
        int err;
        int chromaloc = vsapi->mapGetIntSaturated(props, "_ChromaLocation", 0, &err);
        if (err || chromaloc < 0 || chromaloc > 5) {
            chromaloc = 0;
        }
        
        VSFrame *lin = NULL;
        if (d->gamma != 1.0f) {
            lin = vsapi->newVideoFrame(fi, d->vi.width, d->vi.height, NULL, core);
        }
        
        VSFrame *tmp = NULL;
        if (d->process_w && d->process_h) {
            tmp = vsapi->newVideoFrame(fi, d->dst_width, d->vi.height, NULL, core);
        }
        
        VSFrame *dst = vsapi->newVideoFrame(fi, d->dst_width, d->dst_height, src, core);
        
        VSFrame *shr = NULL;
        if (d->sharp != 1.0f) {
            shr = vsapi->newVideoFrame(fi, d->dst_width, d->dst_height, NULL, core);
        }
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float * VS_RESTRICT srcp = (const float *)vsapi->getReadPtr(src, plane);
            float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            ptrdiff_t dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            int dst_w = vsapi->getFrameWidth(dst, plane);
            int dst_h = vsapi->getFrameHeight(dst, plane);
            
            double start_w = d->start_w;
            double start_h = d->start_h;
            double real_w = d->real_w;
            double real_h = d->real_h;
            
            // Для люмы и RGB/YUV444 сдвиги считаются напрямую, а для хромы - с учётом сабсэмплинга и выравнивания
            if (plane && (fi->subSamplingW | fi->subSamplingH)) {
                start_w /= 1 << fi->subSamplingW;
                real_w /= 1 << fi->subSamplingW;
                start_h /= 1 << fi->subSamplingH;
                real_h /= 1 << fi->subSamplingH;
                // Находим значение коррекции для правильного выравнивания хромы по горизонтали
                if (~chromaloc & 1) {// только для чётных
                    start_w += 0.5 / (1 << fi->subSamplingW) - 0.5 * real_w / (dst_w << fi->subSamplingW);// left allign
                }
                // Находим значение коррекции для правильного выравнивания хромы по вертикали
                if (chromaloc & 2) {// 2 и 3
                    start_h += 0.5 / (1 << fi->subSamplingH) - 0.5 * real_h / (dst_h << fi->subSamplingH);// top allign
                }
                else if (chromaloc & 4) {// 4 и 5
                    start_h -= 0.5 / (1 << fi->subSamplingH) - 0.5 * real_h / (dst_h << fi->subSamplingH);// bottom allign
                }
            }
            
            if (d->gamma != 1.0f) {
                float * VS_RESTRICT linp = (float *)vsapi->getWritePtr(lin, plane);
                if (fi->colorFamily == cfRGB) {
                    rgb_to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                }
                else {
                    yuv_to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                }
                if (d->process_w && d->process_h) {
                    float * VS_RESTRICT tmpp = (float *)vsapi->getWritePtr(tmp, plane);
                    resize_width(linp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, start_w, real_w, d->kernel_w);
                    resize_height(tmpp, dstp, dst_stride, dst_w, src_h, dst_h, start_h, real_h, d->kernel_h);
                }
                else if (d->process_w) {
                    resize_width(linp, dstp, src_stride, dst_stride, src_w, src_h, dst_w, start_w, real_w, d->kernel_w);
                }
                else if (d->process_h) {
                    resize_height(linp, dstp, dst_stride, dst_w, src_h, dst_h, start_h, real_h, d->kernel_h);
                }
                else {
                    memcpy(dstp, linp, sizeof(float) * src_stride * src_h);
                }
            }
            else if (d->process_w && d->process_h) {
                float * VS_RESTRICT tmpp = (float *)vsapi->getWritePtr(tmp, plane);
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
                float * VS_RESTRICT shrp = (float *)vsapi->getWritePtr(shr, plane);
                sharp_width(dstp, shrp, dst_stride, dst_w, dst_h, d->sharp);
                sharp_height(shrp, dstp, dst_stride, dst_w, dst_h, d->sharp);
            }
            
            if (d->gamma != 1.0f) {
                if (fi->colorFamily == cfRGB) {
                    linear_to_rgb(dstp, dstp, dst_stride, dst_w, dst_h, d->gamma);
                }
                else {
                    linear_to_yuv(dstp, dstp, dst_stride, dst_w, dst_h, d->gamma);
                }
            }
        }
        
        vsapi->freeFrame(shr);
        vsapi->freeFrame(tmp);
        vsapi->freeFrame(lin);
        vsapi->freeFrame(src);
        
        return dst;
    }
    return NULL;
}

static void VS_CC ResizeFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
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
    ResizeData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "Resize: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.vi.format.colorFamily == cfUndefined) {
        vsapi->mapSetError(out, "Resize: undefined color family");
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
    
    if (d.dst_width % (1 << d.vi.format.subSamplingW) != 0) {
        vsapi->mapSetError(out, "Resize: width must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > 65535) {
        vsapi->mapSetError(out, "Resize: height any of the planes must be greater than 1 and less than or equal to 65535");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height % (1 << d.vi.format.subSamplingH) != 0) {
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
    else {
        vsapi->mapSetError(out, "Resize: invalid kernel specified");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.process_w = (d.dst_width == d.vi.width && d.real_w == d.vi.width && d.start_w == 0.0) ? false : true;
    d.process_h = (d.dst_height == d.vi.height && d.real_h == d.vi.height && d.start_h == 0.0) ? false : true;
    
    ResizeData *data = (ResizeData *)malloc(sizeof d);
    *data = d;
    
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
    GammaData *d = (GammaData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        VSFrame *dst = vsapi->newVideoFrame(fi, d->vi.width, d->vi.height, src, core);
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float * VS_RESTRICT srcp = (const float *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);
            
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
    GammaData *d = (GammaData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC LinearizeCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    GammaData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "Linearize: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.vi.format.colorFamily == cfUndefined) {
        vsapi->mapSetError(out, "Linearize: undefined color family");
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
    GammaData *d = (GammaData *)instanceData;
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        VSFrame *dst = vsapi->newVideoFrame(fi, d->vi.width, d->vi.height, src, core);
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float * VS_RESTRICT srcp = (const float *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);
            
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
    GammaData *d = (GammaData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC GammaCorrCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    GammaData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "GammaCorr: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.vi.format.colorFamily == cfUndefined) {
        vsapi->mapSetError(out, "GammaCorr: undefined color family");
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
    vsapi->createVideoFilter(out, "GammaCorr", &d.vi, GammaCorrGetFrame, GammaCorrFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("ru.artyfox.plugins", "artyfox", "A disjointed set of filters", VS_MAKE_VERSION(7, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
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
}
