/*
 * LeWM Video Codec - Encoder Implementation
 * FFmpeg 8.x compatible plugin for LeWM-VC
 *
 * NOTE: FFmpeg 8.x removed callback fields from AVCodec struct.
 * The callbacks (init, encode, close) are now in internal FFCodec.
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

/* Encoder private context */
typedef struct LeWMVCEncoderContext {
    LeWMVCContext *common;
    int width;
    int height;
    int quality;
    int initialized;
    int frames_encoded;
    int64_t bitstream_size;
} LeWMVCEncoderContext;

#if FFmpeg_HAS_CALLBACKS
/* Encoder initialization - FFmpeg 7.x and earlier */
static av_cold int lewmvc_encoder_init(AVCodecContext *avctx) {
    LeWMVCEncoderContext *ctx = av_mallocz(sizeof(LeWMVCEncoderContext));
    if (!ctx) {
        return AVERROR(ENOMEM);
    }
    avctx->priv_data = ctx;
    
    /* Initialize Python and LeWM-VC encoder */
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
    
    if (lewmvc_get_encoder(ctx->common, "lewmvc") < 0) {
        av_log(avctx, AV_LOG_WARNING, "Python encoder not available, using stub\n");
    }
    
    ctx->width = avctx->width;
    ctx->height = avctx->height;
    ctx->quality = 32;  /* Default QP */
    ctx->frames_encoded = 0;
    ctx->bitstream_size = 0;
    ctx->initialized = 1;
    
    avctx->pix_fmt = AV_PIX_FMT_YUV420P;
    
    av_log(avctx, AV_LOG_INFO, "LeWM-VC Encoder initialized: %dx%d QP=%d\n",
           avctx->width, avctx->height, ctx->quality);
    
    return 0;
}

/* Encoder encode function - FFmpeg 7.x and earlier */
static int lewmvc_encoder_encode(AVCodecContext *avctx, AVPacket *pkt,
                                const AVFrame *frame, int *got_packet) {
    LeWMVCEncoderContext *ctx = avctx->priv_data;
    int ret = 0;
    
    *got_packet = 0;
    
    if (!ctx || !ctx->initialized) {
        return AVERROR_INVALIDDATA;
    }
    
    /* If no frame, return for flushing */
    if (!frame) {
        return 0;
    }
    
    /* Allocate packet for encoded output */
    /* Stub: allocate minimal size. Production would use Python encoder output */
    ret = av_new_packet(pkt, ctx->width * ctx->height / 4);
    if (ret < 0) {
        return ret;
    }
    
    /* Stub: just mark as encoded with placeholder data */
    /* In production, this would call the Python encoder */
    av_log(avctx, AV_LOG_DEBUG, "Encoding frame %d (stub)\n", ctx->frames_encoded);
    
    ctx->frames_encoded++;
    ctx->bitstream_size += pkt->size;
    *got_packet = 1;
    
    return 0;
}

/* Encoder close function - FFmpeg 7.x and earlier */
static av_cold int lewmvc_encoder_close(AVCodecContext *avctx) {
    LeWMVCEncoderContext *ctx = avctx->priv_data;
    
    if (ctx) {
        if (ctx->common) {
            av_log(avctx, AV_LOG_INFO, "Encoded %d frames, %lld bytes\n",
                   ctx->frames_encoded, (long long)ctx->bitstream_size);
            lewmvc_free_context(ctx->common);
        }
        av_free(ctx);
    }
    
    av_log(avctx, AV_LOG_INFO, "LeWM-VC Encoder closed\n");
    return 0;
}
#endif /* FFmpeg_HAS_CALLBACKS */

/* FFmpeg 8.x compatible encoder definition
 * NOTE: FFmpeg 8.x removed callback fields from AVCodec struct.
 * For FFmpeg 8.x, use Python-based encoding/decoding directly.
 */
const AVCodec ff_lewmvc_encoder = {
    .name           = "lewmvc",
    .long_name      = "LeWM Video Codec (JEPA-based)",
    .type           = AVMEDIA_TYPE_VIDEO,
    .id             = AV_CODEC_ID_NONE,  /* Must be assigned a real codec ID */
    .capabilities   = AV_CODEC_CAP_DR1 | AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .wrapper_name   = "lewm_vc",
#if FFmpeg_HAS_CALLBACKS
    .init           = lewmvc_encoder_init,
    .encode         = lewmvc_encoder_encode,
    .close          = lewmvc_encoder_close,
#endif
};
