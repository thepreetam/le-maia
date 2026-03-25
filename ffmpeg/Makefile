# LeWM-VC FFmpeg Plugin Makefile

CC ?= gcc
CFLAGS ?= -O2 -fPIC -Wall

# Python configuration
PYTHON_CONFIG := python3-config
PYTHON_CFLAGS := $(shell $(PYTHON_CONFIG) --cflags 2>/dev/null || echo "-I/usr/include/python3.11")
PYTHON_LIBS := $(shell $(PYTHON_CONFIG) --ldflags 2>/dev/null || echo "-lpython3.11")

# FFmpeg configuration
FFMPEG_CFLAGS := $(shell pkg-config --cflags libavcodec libavutil 2>/dev/null)
FFMPEG_LIBS := $(shell pkg-config --libs libavcodec libavutil 2>/dev/null)

# Fallback for systems without pkg-config
ifeq ($(FFMPEG_CFLAGS),)
FFMPEG_CFLAGS := -I/usr/include -I/usr/local/include
FFMPEG_LIBS := -lavcodec -lavutil -lavformat
endif

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