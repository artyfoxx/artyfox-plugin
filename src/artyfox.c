#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#define CLAMP(x, min, max) ((x) > (max) ? (max) : ((x) < (min) ? (min) : (x))) 
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define PI_FLOAT 3.1415927f

typedef float (*kernel_func)(float x, void *ctx);

typedef struct {
    kernel_func f;
    float tale;
    void *ctx;
} kernel_t;

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int dst_width;
    int dst_height;
    float start_w;
    float start_h;
    float real_w;
    float real_h;
    kernel_t kernel_w;
    kernel_t kernel_h;
    float gamma;
    float sharp;
    bool process_w;
    bool process_h;
} ResizeData;

static void rgb_to_linear(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
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

static void yuv_to_linear(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
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

typedef struct {
    float scale;
} area_ctx;

static float area_kernel(float x, void *ctx) {
    area_ctx *ar = (area_ctx *)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 0.5f - ar->scale / 2.0f) {
        return 1.0f;
    }
    if (x < 0.5f + ar->scale / 2.0f) {
        return 0.5f - (x - 0.5f) / ar->scale;
    }
    return 0.0f;
}

static float magic_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 0.5f) {
        return 0.75f - powf(x, 2.0f);
    }
    if (x < 1.5f) {
        return 0.5f * powf(x - 1.5f, 2.0f);
    }
    return 0.0f;
}

static float magic_kernel_2013(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 0.5f) {
        return 1.0625f - 1.75f * powf(x, 2.0f);
    }
    if (x < 1.5f) {
        return (1.0f - x) * (1.75f - x);
    }
    if (x < 2.5f) {
        return -0.125f * powf(x - 2.5f, 2.0f);
    }
    return 0.0f;
}

static float magic_kernel_2021(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 0.5f) {
        return 577.0f / 576.0f - 239.0f / 144.0f * powf(x, 2.0f);
    }
    if (x < 1.5f) {
        return 35.0f / 36.0f * (x - 1.0f) * (x - 239.0f / 140.0f);
    }
    if (x < 2.5f) {
        return 1.0f / 6.0f * (x - 2.0f) * (65.0f / 24.0f - x);
    }
    if (x < 3.5f) {
        return 1.0f / 36.0f * (x - 3.0f) * (x - 3.75f);
    }
    if (x < 4.5f) {
        return -1.0f / 288.0f * powf(x - 4.5f, 2.0f);
    }
    return 0.0f;
}

static float bilinear_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return 1.0f - x;
    }
    return 0.0f;
}

typedef struct {
    float b, c;
} bicubic_ctx;

static float bicubic_kernel(float x, void *ctx) {
    bicubic_ctx *bc = (bicubic_ctx *)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return (
            ((12.0f - 9.0f * bc->b - 6.0f * bc->c) * x +
            (-18.0f + 12.0f * bc->b + 6.0f * bc->c)) * x * x +
            (6.0f - 2.0f * bc->b)
        ) / 6.0f;
    }
    if (x < 2.0f) {
        return (
            (((-bc->b - 6.0f * bc->c) * x +
            (6.0f * bc->b + 30.0f * bc->c)) * x +
            (-12.0f * bc->b - 48.0f * bc->c)) * x +
            (8.0f * bc->b + 24.0f * bc->c)
        ) / 6.0f;
    }
    return 0.0f;
}

static inline float sinc_function(float x) {
    if (x == 0.0f) {
        return 1.0f;
    }
    x *= PI_FLOAT;
    return sinf(x) / (x);
}

typedef struct {
    int taps;
} lanczos_ctx;

static float lanczos_kernel(float x, void *ctx) {
    lanczos_ctx *lc = (lanczos_ctx *)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < lc->taps) {
        return sinc_function(x) * sinc_function(x / lc->taps);
    }
    return 0.0f;
}

static float spline16_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return ((x - 1.8f) * x - 0.2f) * x + 1.0f;
    }
    if (x < 2.0f) {
        x -= 1.0f;
        return ((-1.0f / 3.0f * x + 0.8f) * x - 7.0f / 15.0f) * x;
    }
    return 0.0f;
}

static float spline36_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return ((13.0f / 11.0f * x - 453.0f / 209.0f) * x - 3.0f / 209.0f) * x + 1.0f;
    }
    if (x < 2.0f) {
        x -= 1.0f;
        return ((-6.0f / 11.0f * x + 270.0f / 209.0f) * x - 156.0f / 209.0f) * x;
    }
    if (x < 3.0f) {
        x -= 2.0f;
        return ((1.0f / 11.0f * x - 45.0f / 209.0f) * x + 26.0f / 209.0f) * x;
    }
    return 0.0f;
}

static float spline64_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return ((49.0f / 41.0f * x - 6387.0f / 2911.0f) * x - 3.0f / 2911.0f) * x + 1.0f;
    }
    if (x < 2.0f) {
        x -= 1.0f;
        return ((-24.0f / 41.0f * x + 4032.0f / 2911.0f) * x - 2328.0f / 2911.0f) * x;
    }
    if (x < 3.0f) {
        x -= 2.0f;
        return ((6.0f / 41.0f * x - 1008.0f / 2911.0f) * x + 582.0f / 2911.0f) * x;
    }
    if (x < 4.0f) {
        x -= 3.0f;
        return ((-1.0f / 41.0f * x + 168.0f / 2911.0f) * x - 97.0f / 2911.0f) * x;
    }
    return 0.0f;
}

static float spline100_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return ((61.0f / 51.0f * x - 9893.0f / 4505.0f) * x - 1.0f / 13515.0f) * x + 1.0f;
    }
    if (x < 2.0f) {
        x -= 1.0f;
        return ((-10.0f / 17.0f * x +  1254.0f / 901.0f) * x - 724.0f / 901.0f) * x;
    }
    if (x < 3.0f) {
        x -= 2.0f;
        return ((8.0f / 51.0f * x - 1672.0f / 4505.0f) * x + 2896.0f / 13515.0f) * x;
    }
    if (x < 4.0f) {
        x -= 3.0f;
        return ((-2.0f / 51.0f * x +  418.0f / 4505.0f) * x -  724.0f / 13515.0f) * x;
    }
    if (x < 5.0f) {
        x -= 4.0f;
        return ((1.0f / 153.0f * x - 209.0f / 13515.0f) * x +  362.0f / 40545.0f) * x;
    }
    return 0.0f;
}

static float spline144_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 1.0f) {
        return ((683.0f / 571.0f * x - 1240203.0f / 564719.0f) * x - 3.0f / 564719.0f) * x + 1.0f;
    }
    if (x < 2.0f) {
        x -= 1.0f;
        return ((-336.0f / 571.0f * x + 786240.0f / 564719.0f) * x - 453936.0f / 564719.0f) * x;
    }
    if (x < 3.0f) {
        x -= 2.0f;
        return ((90.0f / 571.0f * x - 210600.0f / 564719.0f) * x + 121590.0f / 564719.0f) * x;
    }
    if (x < 4.0f) {
        x -= 3.0f;
        return ((-24.0f / 571.0f * x + 56160.0f / 564719.0f) * x - 32424.0f / 564719.0f) * x;
    }
    if (x < 5.0f) {
        x -= 4.0f;
        return ((6.0f / 571.0f * x - 14040.0f / 564719.0f) * x + 8106.0f / 564719.0f) * x;
    }
    if (x < 6.0f) {
        x -= 5.0f;
        return ((-1.0f / 571.0f * x + 2340.0f / 564719.0f) * x - 1351.0f / 564719.0f) * x;
    }
    return 0.0f;
}

static float point_kernel(float x, void *ctx) {
    (void)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < 0.5f) {
        return 1.0f;
    }
    return 0.0f;
}

static float blackman_kernel(float x, void *ctx) {
    lanczos_ctx *lc = (lanczos_ctx *)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < lc->taps) {
        float nx = x * PI_FLOAT / lc->taps;
        return sinc_function(x) * (0.42f + 0.5f * cosf(nx) + 0.08f * cosf(nx * 2.0f));
    }
    return 0.0f;
}

static float nuttall_kernel(float x, void *ctx) {
    lanczos_ctx *lc = (lanczos_ctx *)ctx;
    if (x < 0.0f) {
        x = -x;
    }
    if (x < lc->taps) {
        float nx = x * PI_FLOAT / lc->taps;
        return sinc_function(x) * (0.355768f + 0.487396f * cosf(nx) + 0.144232f * cosf(nx * 2.0f) + 0.012604f * cosf(nx * 3.0f));
    }
    return 0.0f;
}

static void resize_width(
    const float *srcp, float *dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride, int src_w, int src_h, int dst_w,
    float start_w, float real_w, kernel_t kernel
) {
    float factor = dst_w / real_w;
    float scale = (factor < 1.0f) ? factor : 1.0f;
    float **weights = (float **)malloc(sizeof(float *) * dst_w);
    int *low = (int *)malloc(sizeof(int) * dst_w);
    int *high = (int *)malloc(sizeof(int) * dst_w);
    int min_w = (int)floorf(start_w);
    int max_w = (int)ceilf(real_w + start_w) - 1;
    int border = src_w - 1;
    float tale = kernel.tale / scale;
    
    for (int x = 0; x < dst_w; x++) {
        float center = (x + 0.5f) / factor - 0.5f + start_w;
        low[x] = MAX((int)floorf(center - tale), min_w);
        high[x] = MIN((int)ceilf(center + tale), max_w);
        weights[x] = (float *)malloc(sizeof(float) * (high[x] - low[x] + 1));
        float norm = 0.0f;
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
            dstp[x] = 0.0f;
            for (int i = low[x]; i <= high[x]; i++) {
                dstp[x] += srcp[CLAMP(i, 0, border)] * weights[x][i - low[x]];
            }
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
    float start_h, float real_h, kernel_t kernel
) {
    float factor = dst_h / real_h;
    float scale = (factor < 1.0f) ? factor : 1.0f;
    int min_h = (int)floorf(start_h);
    int max_h = (int)ceilf(real_h + start_h) - 1;
    int border = src_h - 1;
    float tale = kernel.tale / scale;
    
    for (int y = 0; y < dst_h; y++) {
        float center = (y + 0.5f) / factor - 0.5f + start_h;
        int low = MAX((int)floorf(center - tale), min_h);
        int high = MIN((int)ceilf(center + tale), max_h);
        float *weights = (float *)malloc(sizeof(float) * (high - low + 1));
        float norm = 0.0f;
        for (int i = low; i <= high; i++) {
            weights[i - low] = kernel.f((i - center) * scale, kernel.ctx);
            norm += weights[i - low];
        }
        for (int i = low; i <= high; i++) {
            weights[i - low] /= norm;
        }
        for (int x = 0; x < src_w; x++) {
            dstp[x] = 0.0f;
            for (int i = low; i <= high; i++) {
                dstp[x] += srcp[CLAMP(i, 0, border) * src_w + x] * weights[i - low];
            }
        }
        dstp += dst_stride;
        free(weights);
    }
}

static void linear_to_rgb(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
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

static void linear_to_yuv(const float *srcp, float *dstp, ptrdiff_t stride, int src_w, int src_h, float gamma) {
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
            dstp[x] = srcp[x] * sharp + (1.0f - sharp) * (srcp[(y > 0) ? (x - src_w) : x] + srcp[x] + srcp[(y < border) ? (x + src_w) : x]) / 3.0f;
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
            
            float start_w = d->start_w;
            float start_h = d->start_h;
            float real_w = d->real_w;
            float real_h = d->real_h;
            
            // Для люмы и RGB/YUV444 сдвиги считаются напрямую, а для хромы - с учётом сабсэмплинга и выравнивания
            if (plane && (fi->subSamplingW | fi->subSamplingH)) {
                start_w /= 1 << fi->subSamplingW;
                real_w /= 1 << fi->subSamplingW;
                start_h /= 1 << fi->subSamplingH;
                real_h /= 1 << fi->subSamplingH;
                // Находим значение коррекции для правильного выравнивания хромы по горизонтали
                if (~chromaloc & 1) {// только для чётных
                    start_w += 0.5f / (1 << fi->subSamplingW) - 0.5f * real_w / (dst_w << fi->subSamplingW);// left allign
                }
                // Находим значение коррекции для правильного выравнивания хромы по вертикали
                if (chromaloc & 2) {// 2 и 3
                    start_h += 0.5f / (1 << fi->subSamplingH) - 0.5f * real_h / (dst_h << fi->subSamplingH);// top allign
                }
                else if (chromaloc & 4) {// 4 и 5
                    start_h -= 0.5f / (1 << fi->subSamplingH) - 0.5f * real_h / (dst_h << fi->subSamplingH);// bottom allign
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
                    memcpy(dstp, linp, sizeof(float) * src_w * src_h);
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
                memcpy(dstp, srcp, sizeof(float) * src_w * src_h);
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
    
    if (d.dst_width <= 1 << d.vi.format.subSamplingW || d.dst_width > 32767) {
        vsapi->mapSetError(out, "Resize: width any of the planes must be greater than 1 and less than or equal to 32767");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_width >> d.vi.format.subSamplingW << d.vi.format.subSamplingW != d.dst_width) {
        vsapi->mapSetError(out, "Resize: width must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > 32767) {
        vsapi->mapSetError(out, "Resize: height any of the planes must be greater than 1 and less than or equal to 32767");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height >> d.vi.format.subSamplingH << d.vi.format.subSamplingH != d.dst_height) {
        vsapi->mapSetError(out, "Resize: height must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    int err;
    
    d.start_w = vsapi->mapGetFloatSaturated(in, "src_left", 0, &err);
    if (err) {
        d.start_w = 0.0f;
    }
    
    d.start_h = vsapi->mapGetFloatSaturated(in, "src_top", 0, &err);
    if (err) {
        d.start_h = 0.0f;
    }
    
    d.real_w = vsapi->mapGetFloatSaturated(in, "src_width", 0, &err);
    if (err) {
        d.real_w = (float)d.vi.width;
    }
    
    if (d.real_w <= 0.0f) {
        d.real_w += d.vi.width - d.start_w;
    }
    
    d.real_h = vsapi->mapGetFloatSaturated(in, "src_height", 0, &err);
    if (err) {
        d.real_h = (float)d.vi.height;
    }
    
    if (d.real_h <= 0.0f) {
        d.real_h += d.vi.height - d.start_h;
    }
    
    const char *kernel = vsapi->mapGetData(in, "kernel", 0, &err);
    if (err || !strcmp(kernel, "area")) {
        area_ctx *ar_w = (area_ctx *)malloc(sizeof(*ar_w));
        area_ctx *ar_h = (area_ctx *)malloc(sizeof(*ar_h));
        ar_w->scale = (d.dst_width < d.real_w) ? (d.dst_width / d.real_w) : (d.real_w / d.dst_width);
        ar_h->scale = (d.dst_height < d.real_h) ? (d.dst_height / d.real_h) : (d.real_h / d.dst_height);
        d.kernel_w = (kernel_t){area_kernel, 0.5f + ar_w->scale / 2.0f, ar_w};
        d.kernel_h = (kernel_t){area_kernel, 0.5f + ar_h->scale / 2.0f, ar_h};
    }
    else if (!strcmp(kernel, "magic")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel, 1.5f, NULL};
    }
    else if (!strcmp(kernel, "magic13")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel_2013, 2.5f, NULL};
    }
    else if (!strcmp(kernel, "magic21")) {
        d.kernel_w = d.kernel_h = (kernel_t){magic_kernel_2021, 4.5f, NULL};
    }
    else if (!strcmp(kernel, "bilinear")) {
        d.kernel_w = d.kernel_h = (kernel_t){bilinear_kernel, 1.0f, NULL};
    }
    else if (!strcmp(kernel, "bicubic")) {
        bicubic_ctx *bc = (bicubic_ctx *)malloc(sizeof(*bc));
        bc->b = vsapi->mapGetFloatSaturated(in, "b", 0, &err);
        if (err) {
            bc->b = 1.0f / 3.0f;
        }
        bc->c = vsapi->mapGetFloatSaturated(in, "c", 0, &err);
        if (err) {
            bc->c = 1.0f / 3.0f;
        }
        d.kernel_w = d.kernel_h = (kernel_t){bicubic_kernel, 2.0f, bc};
    }
    else if (!strcmp(kernel, "lanczos")) {
        lanczos_ctx *lc = (lanczos_ctx *)malloc(sizeof(*lc));
        lc->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            lc->taps = 3;
        }
        if (lc->taps < 1 || lc->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(lc);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){lanczos_kernel, (float)lc->taps, lc};
    }
    else if (!strcmp(kernel, "spline16")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline16_kernel, 2.0f, NULL};
    }
    else if (!strcmp(kernel, "spline36")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline36_kernel, 3.0f, NULL};
    }
    else if (!strcmp(kernel, "spline64")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline64_kernel, 4.0f, NULL};
    }
    else if (!strcmp(kernel, "spline100")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline100_kernel, 5.0f, NULL};
    }
    else if (!strcmp(kernel, "spline144")) {
        d.kernel_w = d.kernel_h = (kernel_t){spline144_kernel, 6.0f, NULL};
    }
    else if (!strcmp(kernel, "point")) {
        d.kernel_w = d.kernel_h = (kernel_t){point_kernel, 0.5f, NULL};
    }
    else if (!strcmp(kernel, "blackman")) {
        lanczos_ctx *lc = (lanczos_ctx *)malloc(sizeof(*lc));
        lc->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            lc->taps = 3;
        }
        if (lc->taps < 1 || lc->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(lc);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){blackman_kernel, (float)lc->taps, lc};
    }
    else if (!strcmp(kernel, "nuttall")) {
        lanczos_ctx *lc = (lanczos_ctx *)malloc(sizeof(*lc));
        lc->taps = vsapi->mapGetIntSaturated(in, "taps", 0, &err);
        if (err) {
            lc->taps = 3;
        }
        if (lc->taps < 1 || lc->taps > 128) {
            vsapi->mapSetError(out, "Resize: taps must be between 1 and 128");
            vsapi->freeNode(d.node);
            free(lc);
            return;
        }
        d.kernel_w = d.kernel_h = (kernel_t){nuttall_kernel, (float)lc->taps, lc};
    }
    else {
        vsapi->mapSetError(out, "Resize: invalid kernel specified");
        vsapi->freeNode(d.node);
        return;
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
    
    d.process_w = (d.dst_width == d.vi.width && d.real_w == d.vi.width && d.start_w == 0.0f) ? false : true;
    d.process_h = (d.dst_height == d.vi.height && d.real_h == d.vi.height && d.start_h == 0.0f) ? false : true;
    
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
    vspapi->configPlugin("ru.artyfox.plugins", "artyfox", "A disjointed set of filters", VS_MAKE_VERSION(6, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
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
