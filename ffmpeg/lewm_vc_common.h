/*
 * LeWM Video Codec - Common Header
 * FFmpeg plugin for LeWM-VC
 * Updated for FFmpeg 8.x API
 */

#ifndef LEWM_VC_COMMON_H
#define LEWM_VC_COMMON_H

/* __STDC_CONSTANT_MACROS is defined in Makefile for FFmpeg compatibility */

#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavcodec/avcodec.h>

/* Python.h must be included before any standard headers */
#include <Python.h>

/* Context structure for LeWM-VC codec */
typedef struct LeWMVCContext {
    PyObject *module;
    PyObject *encoder;
    PyObject *decoder;
    PyObject *encode_func;
    PyObject *decode_func;
    int initialized;
} LeWMVCContext;

/* Function declarations */
int lewmvc_init_python(void);
void lewmvc_finalize_python(void);
LeWMVCContext *lewmvc_create_context(void);
void lewmvc_free_context(LeWMVCContext *ctx);
int lewmvc_get_decoder(LeWMVCContext *ctx, const char *name);
int lewmvc_get_encoder(LeWMVCContext *ctx, const char *name);

#endif /* LEWM_VC_COMMON_H */
