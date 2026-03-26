# LeWM-VC FFmpeg Plugin Makefile
# Supports: ARM64 (Apple Silicon), ARM32 (ARMv7), x86_64

CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall

# Build targets
TARGET ?= native

ifeq ($(TARGET),arm64)
    CC := arm64-apple-darwin21-
    CFLAGS += -march=armv8.2-a
else ifeq ($(TARGET),arm32)
    CC := arm-linux-gnueabihf-gcc
    CFLAGS += -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
else ifeq ($(TARGET),rpi4)
    CC := aarch64-linux-gnu-gcc
    CFLAGS += -march=armv8-a+crc+fp16+simd
endif

# Python configuration (use python3.14)
PYTHON_VERSION := 3.14
PYTHON_CFLAGS := -I/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/include/python3.14
PYTHON_LIBS := -F/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks -framework Python

# FFmpeg configuration
FFMPEG_CFLAGS := -I/opt/homebrew/Cellar/ffmpeg/8.1/include
FFMPEG_LIBS := -L/opt/homebrew/lib -lavcodec -lavutil -lavformat

# Additional flags
CFLAGS += $(PYTHON_CFLAGS) $(FFMPEG_CFLAGS) -D__STDC_CONSTANT_MACROS

# Source files
SOURCES = lewm_vc_common.c lewm_vc_decoder.c lewm_vc_encoder.c
OBJECTS = $(SOURCES:.c=.o)

# Output
PLUGIN_NAME = liblewmvc.so

.PHONY: all clean install test

all: $(PLUGIN_NAME)

$(PLUGIN_NAME): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(PYTHON_LIBS) $(FFMPEG_LIBS)
	@echo "Build complete: $@"

%.o: %.c lewm_vc_common.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(PLUGIN_NAME)
	@echo "Cleaned build artifacts"

install: $(PLUGIN_NAME)
	mkdir -p /usr/local/lib/ffmpeg
	cp $(PLUGIN_NAME) /usr/local/lib/ffmpeg/
	@echo "Plugin installed to /usr/local/lib/ffmpeg/"
	@echo "Load with: ffmpeg -codecs | grep lewmvc"

test: all
	@echo "Testing plugin registration..."
	@ffmpeg -hide_banner -codecs 2>/dev/null | grep -i lewmvc || \
		ffprobe -hide_banner -codecs 2>/dev/null | grep -i lewmvc || \
		(echo "Note: Plugin needs to be in FFmpeg's plugin path" && \
		 echo "Try: ffmpeg -hide_banner -i input.mp4 -c:v lewmvc output.lewmvc")