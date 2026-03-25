/*
 * LeWM Video Codec - Encoder Implementation
 * FFmpeg plugin for LeWM-VC
 */

#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <avcodec.h>
#include "lewm_vc_common.h"

typedef struct LeWMVCEncoderContext {
    LeWMVCContext *common;
    int width;
    int height;
    int qp;
    int initialized;
} LeWMVCEncoderContext;

static av_cold int lewmvc_encoder_init(AVCodecContext *avctx) {
    LeWMVCEncoderContext *ctx = av_mallocz(sizeof(LeWMVCEncoderContext));
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

    if (lewmvc_get_encoder(ctx->common, "lewmvc") < 0) {
        lewmvc_free_context(ctx->common);
        av_free(ctx);
        return AVERROR(ENOSYS);
    }

    ctx->width = avctx->width;
    ctx->height = avctx->height;
    ctx->qp = 32;

    if (avctx->bit_rate > 0) {
        ctx->qp = (int)(avctx->bit_rate / 1000);
    }

    ctx->initialized = 1;

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    avctx->codec_type = AVMEDIA_TYPE_VIDEO;
    avctx->codec_id = AV_CODEC_ID_NONE;
    avctx->gop_size = 30;
    avctx->max_b_frames = 0;

    av_log(avctx, AV_LOG_INFO, "LeWM-VC Encoder initialized: %dx%d, QP=%d\n",
           avctx->width, avctx->height, ctx->qp);

    return 0;
}

static int lewmvc_encoder_encode(AVCodecContext *avctx, AVPacket *pkt,
                                const AVFrame *frame, int *got_packet) {
    LeWMVCEncoderContext *ctx = avctx->priv_data;
    
    if (!ctx || !ctx->initialized || !ctx->common) {
        *got_packet = 0;
        return AVERROR(EINVAL);
    }

    if (!frame) {
        *got_packet = 0;
        return 0;
    }

    if (!ctx->common->encoder) {
        *got_packet = 0;
        return AVERROR(ENOSYS);
    }

    PyObject *encode_func = PyObject_GetAttrString(ctx->common->encoder, "encode");
    if (!encode_func) {
        PyErr_Print();
        *got_packet = 0;
        return AVERROR(ENOSYS);
    }

    PyObject *y_data = PyBytes_FromStringAndSize(
        (const char *)frame->data[0], frame->linesize[0] * ctx->height);
    PyObject *u_data = PyBytes_FromStringAndSize(
        (const char *)frame->data[1], frame->linesize[1] * (ctx->height / 2));
    PyObject *v_data = PyBytes_FromStringAndSize(
        (const char *)frame->data[2], frame->linesize[2] * (ctx->height / 2));

    if (!y_data || !u_data || !v_data) {
        Py_XDECREF(y_data);
        Py_XDECREF(u_data);
        Py_XDECREF(v_data);
        Py_DECREF(encode_func);
        *got_packet = 0;
        return AVERROR(ENOMEM);
    }

    memcpy(PyBytes_AsString(y_data), frame->data[0], 
            frame->linesize[0] * ctx->height);
    memcpy(PyBytes_AsString(u_data), frame->data[1], 
            frame->linesize[1] * (ctx->height / 2));
    memcpy(PyBytes_AsString(v_data), frame->data[2], 
            frame->linesize[2] * (ctx->height / 2));

    PyObject *args = Py_BuildValue("((NNN)ii)", y_data, u_data, v_data,
                                   ctx->width, ctx->height);
    PyObject *result = PyObject_CallObject(encode_func, args);
    Py_DECREF(args);
    Py_DECREF(encode_func);

    if (!result) {
        PyErr_Print();
        *got_packet = 0;
        return AVERROR(EINVAL);
    }

    if (!PyBytes_Check(result)) {
        Py_DECREF(result);
        *got_packet = 0;
        return 0;
    }

    char *encoded_data = PyBytes_AsString(result);
    Py_ssize_t encoded_size = PyBytes_Size(result);

    int ret = av_new_packet(pkt, encoded_size);
    if (ret < 0) {
        Py_DECREF(result);
        *got_packet = 0;
        return ret;
    }

    memcpy(pkt->data, encoded_data, encoded_size);
    pkt->pts = frame->pts;
    pkt->dts = frame->pts;
    pkt->flags = AV_PKT_FLAG_KEY;

    Py_DECREF(result);
    *got_packet = 1;

    return 0;
}

static av_cold int lewmvc_encoder_close(AVCodecContext *avctx) {
    LeWMVCEncoderContext *ctx = avctx->priv_data;

    if (ctx) {
        if (ctx->common) {
            lewmvc_free_context(ctx->common);
        }
        av_free(ctx);
    }

    av_log(avctx, AV_LOG_INFO, "LeWM-VC Encoder closed\n");
    return 0;
}

static const AVCodecDefault lewmvc_encoder_defaults[] = {
    { "qp", "32" },
    { "b", "0" },
    { "thread_type", "0" },
    { NULL },
};

static const AVOption lewmvc_encoder_options[] = {
    { "qp", "Quantization parameter (0-51)", offsetof(LeWMVCEncoderContext, qp),
      AV_OPT_TYPE_INT, { .i64 = 32 }, 0, 51, AV_OPT_FLAG_ENCODING_PARAM },
    { NULL },
};

AVCodec ff_lewmvc_encoder = {
    .name           = "lewmvc",
    .long_name      = "LeWM Video Codec (Lightweight Efficient Wavelet Motion)",
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_NONE,
    .priv_data_size = sizeof(LeWMVCEncoderContext),
    .init           = lewmvc_encoder_init,
    .encode         = lewmvc_encoder_encode,
    .close          = lewmvc_encoder_close,
    .capabilities   = AV_CODEC_CAP_AUTO_THREADS,
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
    .defaults       = lewmvc_encoder_defaults,
    .priv_class     = &(const AVClass){ .class_name = "lewmvc encoder",
                                         .option = lewmvc_encoder_options },
    .wrapper_name   = "lewm_vc",
};