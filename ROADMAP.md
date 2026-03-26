# LeWM-VC Roadmap

Prioritized next-step roadmap for LeWM-VC, a JEPA-based video codec with stable SIGReg training.

## Current Status

- Working prototype with ViT encoder, temporal predictor, quantization, entropy coding, NAL bitstream
- FFmpeg plugin (C wrapper) implemented
- 159 tests passing
- CI/CD configured

---

## 1. Immediate: Make it reproducible and benchmarkable (1–2 weeks)

Highest-leverage step. Without solid numbers, it's hard for others to care or build on it.

### Run standard video compression benchmarks

**Datasets:**
- UVG, HEVC Class B/C/D/E, MCL-JCV, or Vimeo-90K for training/validation

**Metrics:**
- BD-rate (vs. x265 medium, VTM/H.266, DCVC, ELF-VC) at multiple rate points
- PSNR, MS-SSIM, LPIPS/DISTS/VMAF (perceptual quality)

**Baselines to compare:**
- Traditional: x265, VVC/VTM
- Neural: DCVC family, recent real-time NVCs, VC-VAE-style hybrids

**Expected advantages:**
- Semantic surprise detection + SIGReg could shine on perceptual quality
- Robustness to "physics implausible" artifacts

### Add ablation studies

- Impact of SIGReg on rate-distortion
- Temporal predictor depth/layers vs. bitrate savings
- Semantic surprise module: rate allocation or quality assurance

### Polish the repo

```
benchmarks/
├── scripts/          # Benchmark scripts
├── results/          # Results tables
├── README.md         # Commands and expected numbers
```

- Pre-trained models on Hugging Face or Google Drive
- Clear README with encoding/decoding commands
- FFmpeg integration example: `ffmpeg -i input.mp4 -c:v lewm_vc output.mkv`

---

## 2. Short-term: Improve performance & practicality (2–6 weeks)

Focus on making it competitive or uniquely useful.

### Rate-distortion optimization

- Fine-tune learned λ adaptation and CRF-style QP
- Variable-rate support (single model for multiple bitrates)
- Better entropy modeling (advanced hyperpriors, context-adaptive arithmetic coding)

### Speed & efficiency

- Profile encoding/decoding FPS on CPU/GPU/mobile
- Quantization-aware training (QAT) improvements
- Reduce latency for real-time use

### Leverage JEPA strengths

- Use **semantic surprise** for adaptive rate control
- World-model integration: better inter-frame prediction
- Perceptual post-filter: LPIPS + adversarial loss

### FFmpeg plugin maturity

- Bidirectional compatibility (encode/decode)
- Support for common containers and streaming protocols

---

## 3. Medium-term: Expand & differentiate (1–3 months)

Where LeWM-VC can stand out from generic neural codecs.

### Write and submit a paper

**Target venues:**
- NeurIPS, CVPR, ICCV
- Workshops on neural compression / world models

**Angle:**
> "First JEPA-based end-to-end video codec with stable training via SIGReg, semantic surprise awareness, and production FFmpeg integration."

### Open-source & community

- Release the full repo publicly
- Add examples for robotics/video generation downstream tasks
- Integrate with LeWorldModel repo (lucas-maes/le-wm)

### Unique extensions

- Scalable to higher resolutions using compact 192-dim CLS token
- Multimodal or action-conditioned compression
- Error-resilient streaming with surprise detection

---

## 4. Long-term vision

Position LeWM-VC as a bridge between **learned compression** and **predictive world models**.

**Killer applications:**
- Extremely low-bitrate video for robotics/teleoperation
- Perceptually superior compression for UGC or streaming
- Unified framework for compression + simulation/prediction in agents

---

## Quick prioritization checklist

| Priority | Task | Timeline |
|----------|------|----------|
| 1 | Run benchmarks on UVG + push results to README | Today |
| 2 | Fix installation/FFmpeg issues, add inference speed numbers | This week |
| 3 | Ablations + perceptual improvements | Next |
| 4 | Paper draft + public release | Then |

---

## Technical notes

### Why LeWM-VC is different

1. **SIGReg stability**: Only 2 losses needed (MSE + SIGReg), no EMA teacher
2. **Gaussian latents**: 192-dim isotropic Gaussians = near-ideal entropy prior
3. **Semantic surprise**: Physics implausibility detection for adaptive compression
4. **JEPA architecture**: Temporal prediction without explicit motion estimation

### Key parameters (v1)

| Component | Value |
|----------|-------|
| Latent dim | 192 |
| Patch size | 16x16 |
| Encoder layers | 6 |
| Predictor layers | 8 |
| Encoder heads | 3 |
| Predictor heads | 4 |
| Target GOP | 32 frames |

---

## Resources

- [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) - MIT licensed
- [V-JEPA](https://github.com/facebookresearch/vjepa) - Meta's JEPA variant
- [DCVC](https://arxiv.org/abs/2009.13108) - Deep contextual video compression
- [VTM Reference Software](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM) - VVC test model

---

*Last updated: March 2026*
