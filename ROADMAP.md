# LeWM-VC Roadmap

Prioritized next-step roadmap for LeWM-VC, a JEPA-based video codec with stable SIGReg training.

## Current Status

- Working prototype with ViT encoder, temporal predictor, quantization, entropy coding, NAL bitstream
- FFmpeg plugin (C wrapper) implemented
- 159 tests passing
- CI/CD configured

---

## Proprietary IP Strategy

**This roadmap assumes LeWM-VC is treated as proprietary IP** — internal tool, startup asset, licensing candidate, or acquisition bait. Focus on defensibility, performance proof, and controlled demonstration rather than public visibility.

---

## 1. Secure and Benchmark Internally (1–2 weeks)

### Lock it down

- [ ] Move full codebase to private repo (GitHub private, GitLab, or internal system)
- [ ] Add strict access controls
- [ ] Watermark outputs if needed
- [ ] Document every unique component for IP protection

**Documented components:**
- SIGReg Gaussian entropy modeling
- Semantic surprise detection for quality gating
- JEPA-specific temporal predictor with NAL serialization
- Learned λ + CRF-style rate control
- FFmpeg C wrapper integration

### Run rigorous private benchmarks

**Test sequences:**
- UVG, HEVC CTC Class B/C/D/E, Xiph
- Internal high-value content: 4K/8K, surveillance, gaming, medical video

**Metrics:**
- Full rate-distortion curves: bitrate vs. PSNR, MS-SSIM, VMAF, LPIPS
- Encode/decode speed (FPS on CPU/GPU)
- Memory footprint
- Surprise-detection bitrate savings

**Baselines:**
- Traditional: x265 (HEVC), AV1 (libaom/SVT-AV1), VVC
- Learned: DCVC, VCT, recent neural codecs

**Edge cases to test:**
- Long sequences, scene cuts, high-motion, low-light
- Where JEPA world-model prediction should outperform traditional motion compensation

### Internal documentation

```
benchmarks/
├── scripts/           # Reproducible benchmark scripts
├── results/           # Raw RD curves and logs
├── comparison/        # vs. baselines
└── internal_memo.md  # Technical summary
```

---

## 2. Strengthen Uniqueness and IP (Weeks 1–3, parallel)

### Ablate and harden

- [ ] Internal ablations on JEPA-specific pieces
  - ViT-Tiny encoder + 8-layer SIGReg predictor vs. plain VAE/transformer baselines
  - Quantify semantic surprise detection improvements
  - Rate allocation gains from physics-implausibility gating
- [ ] Document failure modes and where LeWM-VC holds up vs. traditional codecs

### Patent considerations

Consult IP counsel on filing for:
- "JEPA-based latent video coding with SIGReg entropy modeling"
- "Semantic surprise-aware rate allocation in learned video compression"
- "NAL-unit integration with learned world-model latents"
- Specific NAL serialization format extensions

Even provisional filings create defensibility.

### Internal API layer

Build clean interfaces for:
- Drop-in FFmpeg filter or library integration
- Configurable modes:
  - `low-latency`: Real-time streaming
  - `high-compression`: Maximum RD efficiency
  - `surveillance`: Surprise gating for anomaly detection

---

## 3. Controlled Demonstrations and Business Case (Weeks 3–6)

### Private demos

- [ ] Side-by-side comparisons (original vs. encoded/decoded)
- [ ] Real-world use cases:
  - "30% lower bitrate at same perceptual quality for drone footage"
  - "Faster-than-real-time decoding with world-model prediction"
  - "Surprise detection flags anomalous events in security footage"
- [ ] Runtime metrics dashboard
- [ ] Failure-mode analysis (where traditional codecs break but LeWM-VC holds)

### Internal materials

**Technical memo structure:**
```
1. Architecture overview (1 page)
2. Key innovations (SIGReg, surprise detection, JEPA predictor)
3. Benchmark results (RD curves, speed, memory)
4. Use cases and ROI projections
5. IP status and protection steps
```

**Pitch deck angle:**
> "LeWM-VC: JEPA World-Model Video Codec — turning stable pixel-to-latent prediction into practical compression with semantic awareness."

### Monetization paths

| Path | Description |
|------|-------------|
| Internal deployment | Roll out in product/pipeline for storage/bandwidth savings |
| Licensing | Under NDA to video platforms, cloud providers, hardware vendors |
| Acquisition | Position as differentiated neural codec asset |
| Partnership | Selective collab with LeWM team under NDA |

---

## 4. Longer-term Technical Roadmap

### Scale and enhance

- [ ] 1080p/4K/8K testing
- [ ] Longer GOP support (hierarchical prediction)
- [ ] Surprise-aware rate control (allocate bits to anomalous events)
- [ ] Multi-scale latents for progressive quality

### Extensions

- Action-conditioned prediction (if action inputs available)
- Analysis on compressed latents without full decode
- Downstream task integration (robotics, autonomous systems)

### Keep training stable

- Continue QAT stubs and hyperprior entropy model improvements
- Maintain SIGReg regularization discipline

---

## Quick-win Actions This Week

- [ ] Run benchmarks on 2–3 standard sequences vs. x265 at multiple QP/CRF points
- [ ] Draft short internal technical memo summarizing architecture + early RD results
- [ ] Review IP protection steps with legal/management
- [ ] Set up access controls on repo

---

## Open Source Alternative Path

*For reference only — not the primary strategy.*

If open-sourcing later:
1. Pre-trained models on Hugging Face
2. Benchmark results in README
3. Paper submission (NeurIPS, CVPR, ICCV workshops)
4. Community building around learned compression + JEPA

---

## Technical Summary

### Why LeWM-VC is defensible

1. **SIGReg stability**: Only 2 losses (MSE + SIGReg), no EMA teacher
2. **Gaussian latents**: 192-dim isotropic Gaussians = near-ideal entropy prior
3. **Semantic surprise**: Physics implausibility detection (unique)
4. **JEPA architecture**: Temporal prediction without explicit motion estimation
5. **Clean integration**: Production-ready FFmpeg C wrapper

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

## Confidentiality

**This document is proprietary and confidential.**

Do not share externally without authorization.

---

*Last updated: March 2026*
