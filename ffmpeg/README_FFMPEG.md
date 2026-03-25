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