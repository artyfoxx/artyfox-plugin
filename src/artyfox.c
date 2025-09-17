#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

#define CLAMP(x, min, max) (x > max ? max : (x < min ? min : x)) 
#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    int dst_width;
    int dst_height;
    float start_w;
    float start_h;
    float real_w;
    float real_h;
    float gamma;
    float sharp;
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

static void resize_width(const float *srcp, float *dstp, ptrdiff_t src_stride, ptrdiff_t dst_stride, int src_w, int src_h, int dst_w, bool chroma, int subsampling_w, float start_w, float real_w, int chromaloc) {
    // Для люмы и RGB сдвиги считаются напрямую, а для хромы - с учётом сабсэмплинга и выравнивания
    if (chroma) {
        start_w /= 1 << subsampling_w;
        real_w /= 1 << subsampling_w;
        // Находим значение коррекции для правильного выравнивания хромы по горизонтали
        if (~chromaloc & 1) {// только для чётных
            start_w += 0.5f / (1 << subsampling_w) - 0.5f * real_w / (dst_w << subsampling_w);// left allign
        }
    }
    
    int low_w = floorf(start_w);
    float shift_w = start_w - low_w;
    int high_w = ceilf(real_w + start_w);
    int max_w = src_w - 1;
    
    for (int y = 0; y < src_h; y++) {
        memset(dstp, 0, dst_stride * sizeof(float));
        float dist = real_w - dst_w + shift_w * dst_w;
        int k = 0;
        dstp[k] -= srcp[CLAMP(low_w, 0, max_w)] * shift_w * dst_w;
        for (int x = low_w; x < high_w; x++) {
            if (dist > 0.0f) {
                dstp[k] += srcp[CLAMP(x, 0, max_w)] * dst_w;
                dist -= dst_w;
            }
            else if (dist == 0.0f) {
                dstp[k++] += srcp[CLAMP(x, 0, max_w)] * dst_w;
                dist = real_w - dst_w;
            }
            else {
                dstp[k++] += srcp[CLAMP(x, 0, max_w)] * (dst_w + dist);
                if (k < dst_w) {
                    dstp[k] -= srcp[CLAMP(x, 0, max_w)] * dist;
                }
                dist += real_w - dst_w;
            }
        }
        for (int x = 0; x < dst_w; x++) {
            dstp[x] /= real_w;
        }
        dstp += dst_stride;
        srcp += src_stride;
    }
}

static void resize_height(const float *srcp, float *dstp, int src_w, int src_h, int dst_h, bool chroma, int subsampling_h, float start_h, float real_h, int chromaloc) {
    // Для люмы и RGB сдвиги считаются напрямую, а для хромы - с учётом сабсэмплинга и выравнивания
    if (chroma) {
        start_h /= 1 << subsampling_h;
        real_h /= 1 << subsampling_h;
        // Находим значение коррекции для правильного выравнивания хромы по вертикали
        if (chromaloc & 2) {// 2 и 3
            start_h += 0.5f / (1 << subsampling_h) - 0.5f * real_h / (dst_h << subsampling_h);// top allign
        }
        else if (chromaloc & 4) {// 4 и 5
            start_h -= 0.5f / (1 << subsampling_h) - 0.5f * real_h / (dst_h << subsampling_h);// bottom allign
        }
    }
    
    int low_h = floorf(start_h);
    float shift_h = start_h - low_h;
    int high_h = ceilf(real_h + start_h);
    int max_h = src_h - 1;
    
    for (int y = 0; y < src_w; y++) {
        for (int x = 0; x < dst_h; x++) {
            dstp[x * src_w] = 0.0f;
        }
        float dist = real_h - dst_h + shift_h * dst_h;
        int k = 0;
        dstp[k] -= srcp[CLAMP(low_h, 0, max_h) * src_w] * shift_h * dst_h;
        for (int x = low_h; x < high_h; x++) {
            if (dist > 0.0f) {
                dstp[k * src_w] += srcp[CLAMP(x, 0, max_h) * src_w] * dst_h;
                dist -= dst_h;
            }
            else if (dist == 0.0f) {
                dstp[k++ * src_w] += srcp[CLAMP(x, 0, max_h) * src_w] * dst_h;
                dist = real_h - dst_h;
            }
            else {
                dstp[k++ * src_w] += srcp[CLAMP(x, 0, max_h) * src_w] * (dst_h + dist);
                if (k < dst_h) {
                    dstp[k * src_w] -= srcp[CLAMP(x, 0, max_h) * src_w] * dist;
                }
                dist += real_h - dst_h;
            }
        }
        for (int x = 0; x < dst_h; x++) {
            dstp[x * src_w] /= real_h;
        }
        dstp++;
        srcp++;
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
    int max_w = src_w - 1;
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            dstp[x] = srcp[x] * sharp + (1.0f - sharp) * (srcp[MAX(x - 1, 0)] + srcp[x] + srcp[MIN(x + 1, max_w)]) / 3.0f;
        }
        dstp += stride;
        srcp += stride;
    }
}

static void sharp_height(const float *srcp, float *dstp, int src_w, int src_h, float sharp) {
    int max_h = src_h - 1;
    for (int y = 0; y < src_w; y++) {
        for (int x = 0; x < src_h; x++) {
            dstp[x * src_w] = srcp[x * src_w] * sharp + (1.0f - sharp) * (srcp[MAX(x - 1, 0) * src_w] + srcp[x * src_w] + srcp[MIN(x + 1, max_h) * src_w]) / 3.0f;
        }
        dstp++;
        srcp++;
    }
}

static const VSFrame *VS_CC AreaResizeGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
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
        if (err) {
            chromaloc = 0;
        }
        if (chromaloc < 0 || chromaloc > 5) {
            chromaloc = 0;
        }
        
        VSFrame *lin = NULL;
        if (d->gamma != 1.0f) {
            lin = vsapi->newVideoFrame(fi, d->vi.width, d->vi.height, NULL, core);
        }
        
        VSFrame *tmp = vsapi->newVideoFrame(fi, d->dst_width, d->vi.height, NULL, core);
        VSFrame *dst = vsapi->newVideoFrame(fi, d->dst_width, d->dst_height, src, core);
        
        VSFrame *shr = NULL;
        if (d->sharp != 1.0f) {
            shr = vsapi->newVideoFrame(fi, d->dst_width, d->dst_height, NULL, core);
        }
        
        for (int plane = 0; plane < fi->numPlanes; plane++) {
            const float * VS_RESTRICT srcp = (const float *)vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane) / sizeof(float);
            float * VS_RESTRICT tmpp = (float *)vsapi->getWritePtr(tmp, plane);
            float * VS_RESTRICT dstp = (float *)vsapi->getWritePtr(dst, plane);
            ptrdiff_t dst_stride = vsapi->getStride(dst, plane) / sizeof(float);
            
            int src_w = vsapi->getFrameWidth(src, plane);
            int src_h = vsapi->getFrameHeight(src, plane);
            int dst_w = vsapi->getFrameWidth(dst, plane);
            int dst_h = vsapi->getFrameHeight(dst, plane);
            bool chroma = (plane != 0) && (fi->colorFamily != cfRGB);
            
            if (d->gamma != 1.0f) {
                float * VS_RESTRICT linp = (float *)vsapi->getWritePtr(lin, plane);
                if (fi->colorFamily == cfRGB) {
                    rgb_to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                }
                else {
                    yuv_to_linear(srcp, linp, src_stride, src_w, src_h, d->gamma);
                }
                resize_width(linp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, chroma, fi->subSamplingW, d->start_w, d->real_w, chromaloc);
            }
            else {
                resize_width(srcp, tmpp, src_stride, dst_stride, src_w, src_h, dst_w, chroma, fi->subSamplingW, d->start_w, d->real_w, chromaloc);
            }
            
            resize_height(tmpp, dstp, dst_w, src_h, dst_h, chroma, fi->subSamplingH, d->start_h, d->real_h, chromaloc);
            
            if (d->sharp != 1.0f) {
                float * VS_RESTRICT shrp = (float *)vsapi->getWritePtr(shr, plane);
                sharp_width(dstp, shrp, dst_stride, dst_w, dst_h, d->sharp);
                sharp_height(shrp, dstp, dst_w, dst_h, d->sharp);
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

static void VS_CC AreaResizeFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ResizeData *d = (ResizeData *)instanceData;
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC AreaResizeCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    ResizeData d;
    d.node = vsapi->mapGetNode(in, "clip", 0, NULL);
    d.vi = *vsapi->getVideoInfo(d.node);
    
    if (!vsh_isConstantVideoFormat(&d.vi) || d.vi.format.sampleType != stFloat || d.vi.format.bitsPerSample != 32) {
        vsapi->mapSetError(out, "AreaResize: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.vi.format.colorFamily == cfUndefined) {
        vsapi->mapSetError(out, "AreaResize: undefined color family");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.dst_width = vsapi->mapGetIntSaturated(in, "width", 0, NULL);
    d.dst_height = vsapi->mapGetIntSaturated(in, "height", 0, NULL);
    
    if (d.dst_width <= 1 << d.vi.format.subSamplingW || d.dst_width > d.vi.width) {
        vsapi->mapSetError(out, "AreaResize: width any of the planes must be greater than 1 and less than or equal to clip width");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_width >> d.vi.format.subSamplingW << d.vi.format.subSamplingW != d.dst_width) {
        vsapi->mapSetError(out, "AreaResize: width must be a multiple of the subsampling");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height <= 1 << d.vi.format.subSamplingH || d.dst_height > d.vi.height) {
        vsapi->mapSetError(out, "AreaResize: height any of the planes must be greater than 1 and less than or equal to clip height");
        vsapi->freeNode(d.node);
        return;
    }
    
    if (d.dst_height >> d.vi.format.subSamplingH << d.vi.format.subSamplingH != d.dst_height) {
        vsapi->mapSetError(out, "AreaResize: height must be a multiple of the subsampling");
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
    
    if (d.dst_width > d.real_w) {
        vsapi->mapSetError(out, "AreaResize: width must be less than or equal to src_width");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.real_h = vsapi->mapGetFloatSaturated(in, "src_height", 0, &err);
    if (err) {
        d.real_h = (float)d.vi.height;
    }
    
    if (d.real_h <= 0.0f) {
        d.real_h += d.vi.height - d.start_h;
    }
    
    if (d.dst_height > d.real_h) {
        vsapi->mapSetError(out, "AreaResize: height must be less than or equal to src_height");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.gamma = vsapi->mapGetFloatSaturated(in, "gamma", 0, &err);
    if (err) {
        d.gamma = (d.vi.format.colorFamily == cfRGB) ? 2.4f : 1.0f / 0.45f;
    }
    
    if (d.gamma < 0.1f || d.gamma > 5.0f) {
        vsapi->mapSetError(out, "AreaResize: gamma must be between 0.1 and 5");
        vsapi->freeNode(d.node);
        return;
    }
    
    d.sharp = vsapi->mapGetFloatSaturated(in, "sharp", 0, &err);
    if (err) {
        d.sharp = 1.0f;
    }
    
    if (d.sharp < 0.1f || d.sharp > 5.0f) {
        vsapi->mapSetError(out, "AreaResize: sharp must be between 0.1 and 5");
        vsapi->freeNode(d.node);
        return;
    }
    
    ResizeData *data = (ResizeData *)malloc(sizeof d);
    *data = d;
    
    d.vi.width = d.dst_width;
    d.vi.height = d.dst_height;
    
    VSFilterDependency deps[] = {{d.node, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "AreaResize", &d.vi, AreaResizeGetFrame, AreaResizeFree, fmParallel, deps, 1, data, core);
}

typedef struct {
    VSNode *node;
    VSVideoInfo vi;
    float gamma;
    bool process[3];
} GammaData;

static const VSFrame *VS_CC LinearizeGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
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
                memcpy(dstp, srcp, src_stride * src_h * sizeof(float));
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

static const VSFrame *VS_CC GammaCorrGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
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
                memcpy(dstp, srcp, src_stride * src_h * sizeof(float));
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
    vspapi->configPlugin("ru.artyfox.plugins", "artyfox", "A disjointed set of filters", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("AreaResize",
                             "clip:vnode;"
                             "width:int;"
                             "height:int;"
                             "src_left:float:opt;"
                             "src_top:float:opt;"
                             "src_width:float:opt;"
                             "src_height:float:opt;"
                             "gamma:float:opt;"
                             "sharp:float:opt;",
                             "clip:vnode;",
                             AreaResizeCreate,
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
