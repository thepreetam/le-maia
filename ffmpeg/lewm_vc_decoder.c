/*
 * LeWM Video Codec - Decoder Implementation
 * FFmpeg plugin for LeWM-VC
 */

#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
#include <avcodec.h>
#include "lewm_vc_common.h"

typedef struct LeWMVCDecoderContext {
    LeWMVCContext *common;
    int width;
    int height;
    int initialized;
} LeWMVCDecoderContext;

static av_cold int lewmvc_decoder_init(AVCodecContext *avctx) {
    LeWMVCDecoderContext *ctx = av_mallocz(sizeof(LeWMVCDecoderContext));
    if (!ctx)
        return AVERROR(ENOMEM);

    avctx->priv_data = ctx;

    if (lewmvc_init_python() < 0) {
        av_free(ctx);
        return AVERROR(EINVAL);
    }

    ctx->common = lewmvc_create_context();
    if (!ctx->common) {
        av_free(ctx);
        return AVERROR(EINVAL);
    }

    if (lewmvc_get_decoder(ctx->common, "lewmvc") < 0) {
        lewmvc_free_context(ctx->common);
        av_free(ctx);
        return AVERROR(ENOSYS);
    }

    ctx->width = avctx->width;
    ctx->height = avctx->height;
    ctx->initialized = 1;

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    avctx->codec_type = AVMEDIA_TYPE_VIDEO;

    av_log(avctx, AV_LOG_INFO, "LeWM-VC Decoder initialized: %dx%d\n",
           avctx->width, avctx->height);

    return 0;
}

static int lewmvc_decoder_decode(AVCodecContext *avctx, AVFrame *frame,
                                 int *got_frame, AVPacket *pkt) {
    LeWMVCDecoderContext *ctx = avctx->priv_data;
    
    if (!ctx || !ctx->initialized || !ctx->common) {
        *got_frame = 0;
        return AVERROR(EINVAL);
    }

    if (!pkt->data || pkt->size == 0) {
        *got_frame = 0;
        return 0;
    }

    if (!ctx->common->decoder) {
        *got_frame = 0;
        return AVERROR(ENOSYS);
    }

    PyObject *decode_func = PyObject_GetAttrString(ctx->common->decoder, "decode");
    if (!decode_func) {
        PyErr_Print();
        *got_frame = 0;
        return AVERROR(ENOSYS);
    }

    PyObject *args = Py_BuildValue("(y#)", pkt->data, pkt->size);
    PyObject *result = PyObject_CallObject(decode_func, args);
    Py_DECREF(args);
    Py_DECREF(decode_func);

    if (!result) {
        PyErr_Print();
        *got_frame = 0;
        return AVERROR(EINVAL);
    }

    if (!PyBytes_Check(result)) {
        Py_DECREF(result);
        *got_frame = 0;
        return 0;
    }

    char *output_data = PyBytes_AsString(result);
    Py_ssize_t output_size = PyBytes_Size(result);

    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = ctx->width;
    frame->height = ctx->height;

    int ret = av_frame_get_buffer(frame, 32);
    if (ret < 0) {
        Py_DECREF(result);
        *got_frame = 0;
        return ret;
    }

    memcpy(frame->data[0], output_data, output_size);
    frame->linesize[0] = ctx->width;
    frame->linesize[1] = ctx->width / 2;
    frame->linesize[2] = ctx->width / 2;

    frame->key_frame = 1;
    frame->pict_type = AV_PICTURE_TYPE_I;

    Py_DECREF(result);
    *got_frame = 1;

    return pkt->size;
}

static av_cold int lewmvc_decoder_close(AVCodecContext *avctx) {
    LeWMVCDecoderContext *ctx = avctx->priv_data;

    if (ctx) {
        if (ctx->common) {
            lewmvc_free_context(ctx->common);
        }
        av_free(ctx);
    }

    av_log(avctx, AV_LOG_INFO, "LeWM-VC Decoder closed\n");
    return 0;
}

static const AVCodecDefault lewmvc_decoder_defaults[] = {
    { "thread_type", "0" },
    { NULL },
};

AVCodec ff_lewmvc_decoder = {
    .name           = "lewmvc",
    .long_name      = "LeWM Video Codec (Lightweight Efficient Wavelet Motion)",
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_NONE,
    .priv_data_size = sizeof(LeWMVCDecoderContext),
    .init           = lewmvc_decoder_init,
    .decode         = lewmvc_decoder_decode,
    .close          = lewmvc_decoder_close,
    .capabilities   = AV_CODEC_CAP_DR1,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
    .defaults       = lewmvc_decoder_defaults,
    .wrapper_name   = "lewm_vc",
};