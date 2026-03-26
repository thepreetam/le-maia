/*
 * LeWM Video Codec - Decoder Implementation
 * FFmpeg 8.x compatible plugin for LeWM-VC
 *
 * NOTE: FFmpeg 8.x removed callback fields from AVCodec struct.
 * The callbacks (init, decode, close) are now in internal FFCodec.
 * For FFmpeg 8.x, use Python-based encoding/decoding directly.
 * Callbacks are available when compiled with FFmpeg 7.x headers.
 */

#include "lewm_vc_common.h"

/* Check FFmpeg version for callback support */
/* FFmpeg 8.x (major 62) removed callbacks from public API */
/* FFmpeg 7.x (major 60-61) has callbacks */
#if LIBAVCODEC_VERSION_MAJOR < 62
#define FFmpeg_HAS_CALLBACKS 1
#else
#define FFmpeg_HAS_CALLBACKS 0
#endif

/* Decoder private context */
typedef struct LeWMVCDecoderContext {
    LeWMVCContext *common;
    int width;
    int height;
    int initialized;
    int frames_decoded;
} LeWMVCDecoderContext;

#if FFmpeg_HAS_CALLBACKS
/* Decoder initialization - FFmpeg 7.x and earlier */
static av_cold int lewmvc_decoder_init(AVCodecContext *avctx) {
    LeWMVCDecoderContext *ctx;
    
    ctx = av_mallocz(sizeof(LeWMVCDecoderContext));
    if (!ctx) {
        return AVERROR(ENOMEM);
    }
    avctx->priv_data = ctx;
    
    /* Initialize Python and LeWM-VC decoder */
    if (lewmvc_init_python() < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to initialize Python\n");
        av_free(ctx);
        return AVERROR(EINVAL);
    }
    
    ctx->common = lewmvc_create_context();
    if (!ctx->common) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create LeWM context\n");
        av_free(ctx);
        return AVERROR(EINVAL);
    }
    
    if (lewmvc_get_decoder(ctx->common, "lewmvc") < 0) {
        av_log(avctx, AV_LOG_WARNING, "Python decoder not available, using stub\n");
    }
    
    ctx->width = avctx->width;
    ctx->height = avctx->height;
    ctx->initialized = 1;
    ctx->frames_decoded = 0;
    
    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    
    av_log(avctx, AV_LOG_INFO, "LeWM-VC Decoder initialized: %dx%d\n",
           avctx->width, avctx->height);
    
    return 0;
}

/* Decoder decode function - FFmpeg 7.x and earlier */
static int lewmvc_decoder_decode(AVCodecContext *avctx, AVFrame *frame,
                                 int *got_frame, AVPacket *pkt) {
    LeWMVCDecoderContext *ctx = avctx->priv_data;
    int ret = 0;
    
    *got_frame = 0;
    
    if (!ctx || !ctx->initialized) {
        return AVERROR_INVALIDDATA;
    }
    
    /* If no data, return */
    if (!pkt->data || pkt->size == 0) {
        return 0;
    }
    
    /* Allocate output buffer */
    frame->width = ctx->width;
    frame->height = ctx->height;
    frame->format = AV_PIX_FMT_YUV420P;
    
    ret = avcodec_default_get_buffer2(avctx, frame, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Failed to allocate frame buffer: %d\n", ret);
        return ret;
    }
    
    /* Fill with placeholder data (stub implementation) */
    /* In production, this would call the Python decoder */
    av_log(avctx, AV_LOG_DEBUG, "Decoding frame %d (stub)\n", ctx->frames_decoded);
    
    ctx->frames_decoded++;
    *got_frame = 1;
    
    return pkt->size;
}

/* Decoder close function - FFmpeg 7.x and earlier */
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
#endif /* FFmpeg_HAS_CALLBACKS */

/* FFmpeg 8.x compatible decoder definition
 * NOTE: FFmpeg 8.x removed callback fields from AVCodec struct.
 * For FFmpeg 8.x, use Python-based encoding/decoding directly.
 */
const AVCodec ff_lewmvc_decoder = {
    .name           = "lewmvc",
    .long_name      = "LeWM Video Codec (JEPA-based)",
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_NONE,  /* Must be assigned a real ID */
    .capabilities   = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .wrapper_name   = "lewm_vc",
#if FFmpeg_HAS_CALLBACKS
    .init           = lewmvc_decoder_init,
    .decode         = lewmvc_decoder_decode,
    .close          = lewmvc_decoder_close,
#endif
};
