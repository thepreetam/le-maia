# LeWM-VC FFmpeg Plugin

FFmpeg external codec plugin for LeWM-VC (Lightweight Efficient Wavelet Motion Video Codec).

## Prerequisites

- FFmpeg development libraries (libavcodec, libavutil)
- Python 3.8+
- Python development headers

### Install Dependencies (macOS)

```bash
brew install ffmpeg python3
python3 -m pip install --upgrade pip setuptools wheel
```

### Install Dependencies (Ubuntu/Debian)

```bash
sudo apt-get install ffmpeg libavcodec-dev libavutil-dev python3-dev
```

## Building

```bash
cd /Users/pm/Documents/dev/le-maia/ffmpeg
make clean
make
```

This produces `liblewmvc.so` - the FFmpeg plugin shared library.

## Installation

### System-wide (requires root)

```bash
sudo make install
```

This copies the plugin to `/usr/local/lib/ffmpeg/`.

### Per-user

```bash
mkdir -p ~/.local/lib/ffmpeg
cp liblewmvc.so ~/.local/lib/ffmpeg/
export LD_LIBRARY_PATH=~/.local/lib/ffmpeg:$LD_LIBRARY_PATH
```

## Usage

### Encoding

```bash
ffmpeg -i input.mp4 -c:v lewmvc -qp 24 output.lewmvc
```

Options:
- `-qp N` - Quantization parameter (0-51, lower = better quality)
- `-b N` - Target bitrate (influences QP calculation)

### Decoding

```bash
ffmpeg -i input.lewmvc -c:v libx264 -pix_fmt yuv420p output.mp4
```

### Listing codecs

```bash
ffmpeg -codecs 2>&1 | grep lewmvc
```

## Python Module Requirements

The plugin expects a Python module at:

```
/Users/pm/Documents/dev/le-maia/lewm_vc/
```

With the following structure:

```
lewm_vc/
├── __init__.py
├── codec.py        # get_encoder(), get_decoder()
└── (other codec components)
```

### codec.py API

```python
class Encoder:
    def encode(self, yuv_data, width, height) -> bytes: ...

class Decoder:
    def decode(self, data) -> bytes: ...

def get_encoder(name: str) -> Encoder: ...
def get_decoder(name: str) -> Decoder: ...
```

## ARM Build Targets

### Apple Silicon (ARM64)

```bash
cd ffmpeg
make clean
make TARGET=arm64
# Output: liblewmvc.so (Mach-O arm64)
```

### Raspberry Pi Builds

#### ARM32 (ARMv7) - Raspberry Pi 2/3/Zero 2 W

Requires cross-compiler:
```bash
# Install cross-compiler on macOS
brew install arm-linux-gnueabihf-gcc

# Or on Linux
sudo apt-get install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# Build
make clean
make TARGET=arm32 CC=arm-linux-gnueabihf-gcc
```

#### ARM64 (ARMv8) - Raspberry Pi 4/5

```bash
# Install cross-compiler
brew install aarch64-linux-gnu-gcc

# Or on Linux
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Build
make clean
make TARGET=rpi4 CC=aarch64-linux-gnu-gcc
```

#### Native build on Raspberry Pi

```bash
sudo apt-get install ffmpeg libavcodec-dev libavutil-dev python3-dev
cd ffmpeg
make
```

### Cross-compilation from macOS

1. Install toolchain:
```bash
brew install --cask gcc-arm-embedded
# or
brew install crosstool-ng
```

2. Configure toolchain for ARM:
```bash
export PATH="/opt/arm/bin:$PATH"
export CC=arm-none-eabi-gcc
```

### Verification

Check architecture of built library:
```bash
file liblewmvc.so
# Expected outputs:
# - Mach-O 64-bit arm64 (Apple Silicon)
# - ELF 32-bit ARM (Raspberry Pi ARM32)
# - ELF 64-bit ARM aarch64 (Raspberry Pi ARM64)
```

## Troubleshooting

### "codec not found" error

1. Ensure plugin is in FFmpeg's library path
2. Check: `ldd liblewmvc.so` for missing dependencies
3. Try: `export LD_LIBRARY_PATH=/path/to/plugin:$LD_LIBRARY_PATH`

### Python import errors

1. Verify Python can import the module: `python3 -c "import lewm_vc"`
2. Check PYTHONPATH includes project root

### Build errors

```bash
# Check FFmpeg version
ffmpeg -version

# Check Python config
python3-config --cflags --ldflags

# Use explicit paths if needed
make CC=gcc PYTHON_CFLAGS="-I/usr/include/python3.11" FFMPEG_CFLAGS="-I/usr/include"
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FFmpeg                            │
│  ┌─────────────────────────────────────────────┐   │
│  │  lewmvc_decoder / lewmvc_encoder (C)        │   │
│  │  - AVCodec interface                         │   │
│  │  - PyObject calls to Python codec           │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               lewm_vc Python Module                   │
│  ┌──────────────┐    ┌──────────────┐               │
│  │   Encoder    │    │   Decoder   │               │
│  │ (wavelet +   │    │  (inverse   │               │
│  │  motion)     │    │   wavelet)   │               │
│  └──────────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────┘
```

## License

See project root for licensing information.