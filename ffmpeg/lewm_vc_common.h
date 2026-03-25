/*
 * LeWM Video Codec - Common Header
 * FFmpeg plugin for LeWM-VC (Lightweight Efficient Wavelet Motion Video Codec)
 */

#ifndef LEWM_VC_COMMON_H
#define LEWM_VC_COMMON_H

#include <libavutil/avutil.h>
#include <Python.h>

typedef struct LeWMVCContext {
    PyObject *module;
    PyObject *encoder;
    PyObject *decoder;
    PyObject *encode_func;
    PyObject *decode_func;
    int initialized;
} LeWMVCContext;

int lewmvc_init_python(void);
void lewmvc_finalize_python(void);
LeWMVCContext *lewmvc_create_context(void);
void lewmvc_free_context(LeWMVCContext *ctx);

#endif /* LEWM_VC_COMMON_H */