/*
 * LeWM Video Codec - Common Implementation
 * FFmpeg plugin for LeWM-VC
 */

#include <libavutil/mem.h>
#include "lewm_vc_common.h"

static PyObject *lewmvc_py_module = NULL;
static int python_initialized = 0;

int lewmvc_init_python(void) {
    if (python_initialized)
        return 0;

    Py_Initialize();
    
    PyObject *sys_path = PySys_GetObject("path");
    if (sys_path && PyList_Check(sys_path)) {
        PyObject *project_root = PyUnicode_FromString("/Users/pm/Documents/dev/le-maia");
        if (project_root) {
            PyList_Insert(sys_path, 0, project_root);
            Py_DECREF(project_root);
        }
    }

    lewmvc_py_module = PyImport_ImportModule("lewm_vc");
    if (!lewmvc_py_module) {
        PyErr_Print();
        av_log(NULL, AV_LOG_ERROR, "LeWM-VC: Failed to import lewm_vc module\n");
        return AVERROR(EINVAL);
    }

    python_initialized = 1;
    av_log(NULL, AV_LOG_INFO, "LeWM-VC: Python initialized successfully\n");
    return 0;
}

void lewmvc_finalize_python(void) {
    if (lewmvc_py_module) {
        Py_DECREF(lewmvc_py_module);
        lewmvc_py_module = NULL;
    }

    if (python_initialized) {
        Py_Finalize();
        python_initialized = 0;
    }

    av_log(NULL, AV_LOG_INFO, "LeWM-VC: Python finalized\n");
}

LeWMVCContext *lewmvc_create_context(void) {
    LeWMVCContext *ctx = av_mallocz(sizeof(LeWMVCContext));
    if (!ctx)
        return NULL;

    ctx->module = PyImport_ImportModule("lewm_vc.codec");
    if (!ctx->module) {
        PyErr_Print();
        av_free(ctx);
        return NULL;
    }

    ctx->initialized = 1;
    return ctx;
}

void lewmvc_free_context(LeWMVCContext *ctx) {
    if (!ctx)
        return;

    if (ctx->encode_func) {
        Py_DECREF(ctx->encode_func);
    }
    if (ctx->decode_func) {
        Py_DECREF(ctx->decode_func);
    }
    if (ctx->encoder) {
        Py_DECREF(ctx->encoder);
    }
    if (ctx->decoder) {
        Py_DECREF(ctx->decoder);
    }
    if (ctx->module) {
        Py_DECREF(ctx->module);
    }

    av_free(ctx);
}

int lewmvc_get_encoder(LeWMVCContext *ctx, const char *encoder_name) {
    if (!ctx || !ctx->module)
        return AVERROR(EINVAL);

    PyObject *get_encoder = PyObject_GetAttrString(ctx->module, "get_encoder");
    if (!get_encoder) {
        av_log(NULL, AV_LOG_ERROR, "LeWM-VC: get_encoder function not found\n");
        return AVERROR(ENOSYS);
    }

    PyObject *args = Py_BuildValue("(s)", encoder_name);
    PyObject *encoder = PyObject_CallObject(get_encoder, args);
    Py_DECREF(args);
    Py_DECREF(get_encoder);

    if (!encoder) {
        PyErr_Print();
        return AVERROR(ENOSYS);
    }

    ctx->encoder = encoder;
    return 0;
}

int lewmvc_get_decoder(LeWMVCContext *ctx, const char *decoder_name) {
    if (!ctx || !ctx->module)
        return AVERROR(EINVAL);

    PyObject *get_decoder = PyObject_GetAttrString(ctx->module, "get_decoder");
    if (!get_decoder) {
        av_log(NULL, AV_LOG_ERROR, "LeWM-VC: get_decoder function not found\n");
        return AVERROR(ENOSYS);
    }

    PyObject *args = Py_BuildValue("(s)", decoder_name);
    PyObject *decoder = PyObject_CallObject(get_decoder, args);
    Py_DECREF(args);
    Py_DECREF(get_decoder);

    if (!decoder) {
        PyErr_Print();
        return AVERROR(ENOSYS);
    }

    ctx->decoder = decoder;
    return 0;
}