# LeWM-VC

JEPA-based Video Codec - Learning Energy-based Model for Video Coding

## Project Overview

LeWM-VC is a deep learning-based video codec using Joint Embedding Predictive Architecture (JEPA) with Vision Transformer (ViT) architecture for video frame compression and reconstruction.

## Key Files and Their Purposes

### Core Components

| File | Purpose |
|------|---------|
| `src/lewm_vc/encoder.py` | ViT-Tiny style encoder - converts YUV420 frames to latent representations |
| `src/lewm_vc/decoder.py` | Decoder network - upsamples quantized latents back to frames |
| `src/lewm_vc/predictor.py` | Motion-compensated prediction for inter-frame coding |
| `src/lewm_vc/quant.py` | Latent quantization (scalar/vector) |
| `src/lewm_vc/entropy.py` | Learned entropy model for bitrate optimization |

### Bitstream

| File | Purpose |
|------|---------|
| `src/lewm_vc/bitstream/writer.py` | NAL unit serialization with arithmetic coding |
| `src/lewm_vc/bitstream/reader.py` | Bitstream parsing and decoding |

### Utilities

| File | Purpose |
|------|---------|
| `src/lewm_vc/utils/rate_control.py` | Rate distortion optimization |
| `src/scripts/train.py` | Training script |

### Tests

| File | Purpose |
|------|---------|
| `tests/test_encoder.py` | Encoder unit tests |
| `tests/test_decoder.py` | Decoder unit tests |
| `tests/test_predictor.py` | Predictor tests |
| `tests/test_quant.py` | Quantization tests |
| `tests/test_entropy.py` | Entropy coding tests |
| `tests/test_bitstream.py` | Bitstream tests |
| `tests/test_rate_control.py` | Rate control tests |
| `tests/test_training.py` | Training integration tests |
| `tests/test_integration.py` | End-to-end integration tests |

## Architecture

```
Input Frame (YUV420)
       ↓
  ┌─────────┐
  │ Encoder │ (ViT-Tiny, 6 layers)
  └────┬────┘
       ↓
   Latent (192 channels, H/16 x W/16)
       ↓
  ┌──────────┐
  │ Quantizer│
  └────┬─────┘
       ↓
  ┌────────────┐
  │ Entropy    │
  │ Coder      │
  └─────┬──────┘
       ↓
  Bitstream
```

## Development Workflow

1. **Install dependencies**: `pip install -e ".[dev]"`
2. **Run tests**: `pytest tests/ -v`
3. **Lint code**: `ruff check src/ tests/`

## Configuration

- Default latent dimension: 192
- Patch size: 16x16
- Transformer layers: 6
- Attention heads: 3
- Supported resolutions: Any multiple of 16

## FFmpeg Plugin

Located in `ffmpeg/` - provides integration with FFmpeg for hardware encoding/decoding (optional).
