# LeWM-VC

JEPA-based Video Codec - Learning Energy-based Model for Video Coding

## Overview

LeWM-VC is a deep learning-based video codec built on the Joint Embedding Predictive Architecture (JEPA) paradigm. It uses a Vision Transformer (ViT) encoder to compress video frames into latent representations, which are then quantized and entropy-coded for efficient storage and transmission.

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

## Features

- **ViT-based Encoder**: Vision Transformer for efficient video frame compression
- **Temporal Predictor**: 8-layer transformer with SIGReg Gaussian output
- **Learned Entropy Model**: Hyperprior + arithmetic coding for bitrate optimization
- **Semantic Surprise Detection**: Physics implausibility detection for quality assurance
- **Perceptual Post-Filter**: LPIPS-trained refinement for improved visual quality
- **Bitstream Support**: NAL unit serialization (7 types)
- **Rate Control**: Learned λ adaptation + CRF-style QP selection
- **FFmpeg Plugin**: C wrapper for native FFmpeg integration

## Project Structure

```
src/lewm_vc/
├── encoder.py          # ViT-Tiny encoder (192-dim latents)
├── predictor.py        # 8-layer transformer predictor
├── decoder.py          # ConvTranspose + post-filter
├── entropy.py          # Hyperprior + SIGReg KL
├── quant.py            # STE quantization + QAT stubs
├── bitstream/
│   ├── writer.py       # NAL unit writer
│   └── reader.py      # NAL unit reader
└── utils/
    └── rate_control.py # Learned rate controller

ffmpeg/                 # FFmpeg plugin (C wrapper)
tests/                  # 159 unit tests
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Encoding and Decoding

```python
import torch
from lewm_vc import LeWMEncoder, LeWMDecoder

encoder = LeWMEncoder(latent_dim=192)
decoder = LeWMDecoder(latent_dim=192)
encoder.eval()
decoder.eval()

frame = torch.rand(1, 3, 256, 256)  # [B, 3, H, W]

with torch.no_grad():
    latent = encoder(frame)
    reconstructed = decoder(latent)
```

### Training

```bash
python src/scripts/train.py --config configs/dataset.yaml --phase 0
```

## Testing

```bash
pytest tests/ -v
```

## Configuration

- Default latent dimension: 192
- Patch size: 16x16
- Transformer layers: 6 (encoder), 8 (predictor)
- Attention heads: 3 (encoder), 4 (predictor)

## References

- LeWM (LeWorldModel): https://github.com/lucas-maes/le-wm
- Based on JEPA architecture with SIGReg regularization

## License

MIT License
