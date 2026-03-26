# LeWM-VC Roadmap

Prioritized next-step roadmap for LeWM-VC, a JEPA-based video codec with stable SIGReg training.

**Goal**: Acquisition positioning — shift from prototype to acquirable asset.

---

## Current Status

- Working prototype with ViT encoder, temporal predictor, quantization, entropy coding, NAL bitstream
- FFmpeg plugin (C wrapper) implemented
- 159 tests passing
- CI/CD configured

---

## Critical Shift: Commercial Proof Over Technical Storytelling

Acquirers (especially InterDigital-style buyers) care about:
- **Quantified ROI**: Real bitrate savings, real storage reductions
- **Production hardening**: Does it actually work in deployment?
- **Risk signals**: Where does it break? What are the failure modes?

The JEPA + semantic surprise story is compelling, but it's just narrative until backed by data.

---

## Proprietary IP Strategy

**This roadmap assumes LeWM-VC is treated as proprietary IP** — internal tool, startup asset, licensing candidate, or acquisition bait. Focus on defensibility, performance proof, and controlled demonstration rather than public visibility.

### Key Differentiators

| Feature | Why It Matters | Evidence Needed |
|---------|---------------|-----------------|
| JEPA World-Model Foundation | Stable pixel-to-latent prediction via SIGReg | Ablation vs. optical flow |
| Semantic Surprise Detection | Physics-implausibility gating for quality/rate | ROI on real surveillance footage |
| SIGReg Gaussian Entropy | Near-ideal entropy prior, minimal rate overhead | Rate overhead < 2% |
| LPIPS Perceptual Post-Filter | Superior subjective quality | VMAF/LPIPS scores vs. baselines |
| Production FFmpeg Integration | Clean deployment path | Thread-safe, ARM builds, fuzz-tested |

### Timing Advantage

- InterDigital acquired Deep Render (Oct 2025) for AI-native codec tech
- MPEG/JVET actively exploring NNVC and Beyond-VVC (H.267)
- World-model funding surging ($1B+ in early 2026)
- LeWM-VC positioned as "next logical acquisition" after Deep Render

---

## Phase 0: Production Hardening (This Week)

**Before any NDA demo, FFmpeg plugin must be production-ready.**

### FFmpeg Plugin Requirements (Non-Negotiable)

- [ ] Thread-safety validation
- [ ] Seek/timestamp handling correctness
- [ ] Frame drop handling
- [ ] Corrupted bitstream resilience (fuzzing)
- [ ] ARM builds (Apple Silicon, Raspberry Pi, mobile SoC targets)
- [ ] Worst-case decode latency measurement
- [ ] CPU-only performance baseline

### Quick FFmpeg Checklist

```
[ ] Thread-safety: Multiple parallel encodes/decodes
[ ] Seek test: FFmpeg seeking with -ss flag
[ ] Fuzzing: Inject random corruption, verify graceful failure
[ ] ARM64: Build on Apple Silicon M-series
[ ] ARM32: Build on Raspberry Pi OS
[ ] Latency: Measure p99 decode time per frame
[ ] CPU-only: Verify no GPU required for decode
```

---

## Phase 1: Quantified Surprise ROI (Week 1)

**Top priority experiment — turn the most distinctive feature from claim to evidence.**

### Surveillance Benchmark

- [ ] Acquire 100+ hours of real surveillance footage
  - Diverse: indoor/outdoor, day/night, low-motion vs. events
  - Sources: internal, licensed, or public datasets (if permissible)
- [ ] Encode with/without surprise-gating
- [ ] Measure:
  - Actual storage/bandwidth savings on "normal" segments
  - Quality preservation on "surprise" segments
  - Compute overhead for surprise detection
- [ ] Document fallback modes for latency-critical paths

### ROI Evidence Template

```
Surveillance Footage Test Results:
- Total footage: XX hours
- Normal segments: XX% (surprise-gating enabled)
- Anomaly segments: XX% (full quality mode)
- Bitrate savings: XX% on normal segments
- Quality delta: VMAF/LPIPS scores
- Compute overhead: XX ms per frame for surprise detection
```

### Demo Video (2 minutes)

- [ ] Side-by-side: original vs. LeWM-VC at equivalent bitrate
- [ ] Surprise overlay: visual flagging of gating decisions
- [ ] Include both "normal" and "surprise" segments
- [ ] No audio, clean cuts

---

## Phase 2: JEPA Advantage Ablation (Week 1–2)

**Prove the predictor actually helps — or adjust the narrative.**

### Ablation Study

Compare your 8-layer SIGReg predictor against:

| Baseline | What to Test |
|----------|--------------|
| Optical flow + residual | Standard motion compensation |
| Temporal VAE conditioning | Simple temporal latent conditioning |
| Basic transformer motion | Transformer without SIGReg |

### Metrics

- Rate-distortion curves (PSNR, MS-SSIM, VMAF)
- Long-horizon prediction stability (30+ frame GOPs)
- Surprise detection synergy (does SIGReg improve anomaly detection?)
- Failure modes (where does each approach break?)

### Decision Gate

If JEPA predictor doesn't win clearly:
- De-emphasize "JEPA" in teaser
- Lead with: "Predictive world-model latent coding with semantic rate allocation"

---

## Phase 3: Patent Filings (Week 2–3)

**File provisional patents NOW — before any public discussion.**

### Provisional Patent Claims

1. **Core**: JEPA-based latent video coding with SIGReg entropy modeling
2. **Semantic**: Semantic surprise-aware rate allocation in learned video compression
3. **Integration**: NAL-unit integration with learned world-model latents
4. **Format**: Specific NAL serialization format extensions

### Attach to Filing

- Early RD curves
- Ablation results
- Surveillance savings data

This strengthens arguments vs. prior art in neural codecs.

---

## Phase 4: Confidential Teaser & Data Room (Week 3–4)

### One-Pager Structure

**Headline**: AI-Native Video Compression for Edge Analytics

**1. Problem (1 sentence)**
> Edge video analytics needs intelligent compression that saves bits on predictable scenes while preserving quality where it matters.

**2. Solution (2–3 sentences)**
> LeWM-VC is the first JEPA-world-model video codec with built-in semantic intelligence. Semantic surprise detection automatically identifies physics-implausible events and allocates bits accordingly. Built on stable SIGReg training with production FFmpeg integration.

**3. Differentiators (bullets)**
- Semantic surprise detection for physics-aware bit allocation
- Stable SIGReg training — no EMA teacher, no collapse
- Production-ready: FFmpeg plugin, rate control, NAL bitstream
- LPIPS-trained perceptual post-filter

**4. Proof Points (private data)**
- XX% bitrate savings on surveillance footage (normal segments)
- XX% bitrate savings on drone footage (high-motion)
- XX fps decode on ARM Cortex-A series
- [Attach: full RD curves, ablation results, demo video]

**5. Team & IP**
- [Team bios]
- [Provisional patents filed]
- [Timeline to production-ready]

### Data Room Contents (Under NDA)

| Category | Contents |
|----------|----------|
| Demos | 2-min surveillance demo, side-by-side clips |
| RD Data | Full curves vs. H.264/H.265/AV1, raw logs |
| Ablations | JEPA predictor vs. baselines |
| Technical | High-level architecture (no source) |
| IP | Patent claims, filing status |
| Team | Bios, advisor list |

---

## Phase 5: NDA Outreach (Week 4–6)

### Target Sequence (by fit + receptivity)

| Priority | Target | Angle |
|----------|--------|-------|
| 1 | **InterDigital** | Post-Deep Render, AI-native codec focus |
| 2 | **Qualcomm** | Edge/hardware synergy |
| 3 | **Defense/Drone player** | Surveillance ROI story |
| 4 | Meta | World-model synergy |

### Outreach Approach

1. **NDA-first**: Never send technical details without NDA
2. **Exclusive framing**: "Exclusive evaluation opportunity"
3. **Black-box demos**: Let them run on their content
4. **Quantified claims**: Lead with numbers, not narrative

### Demo Targets

- "30–50% effective bitrate reduction on normal surveillance footage"
- "Better perceptual quality at equivalent bitrate vs. H.265"
- "Faster-than-real-time decode on mobile ARM"

---

## Risk Mitigation for Acquirers

| Risk | Acquirer Concern | Mitigation |
|------|------------------|------------|
| Obsolescence | NNVC track advancing, standards competition | Frame as complementary (long-term prediction, semantic features) |
| Training dependency | What training data was used? | Document internally, avoid copyrighted content |
| Hardware lock-in | Does it only run on NVIDIA? | FFmpeg plugin, CPU fallback, ARM builds |
| Team depth | Can you scale? | Line up codec-industry advisor (MPEG/JVET experience) |
| Quantization instability | Does INT8 break quality? | Document QAT procedures, measure quality delta |

---

## Exit Gate: Month 2–3 Decision

**Define your minimum number now.**

| Signal | Action |
|--------|--------|
| Strong surveillance data + hardened plugin + positive NDA feedback | Accelerate outreach |
| Weak data or plugin still fragile | Reassess: build more vs. fast exit |

**The NNVC window is not infinite** — move while momentum is fresh.

---

## Domain Focus: AI-Native Compression for Edge Analytics

### Primary: Surveillance & Security

**ROI hook**: 
> "XX% effective bitrate reduction on normal footage while auto-highlighting implausible events — no extra compute for separate anomaly detection."

**Why now**:
- Edge AI exploding (2026 trends)
- Low-bandwidth monitoring demand
- AI-driven incident response

### Secondary: Drones / UAV / Defense

**ROI hook**:
> "Physics-aware rate control reduces satellite/5G link costs dramatically."

**Why now**:
- Defense budgets for autonomous ISR
- Bandwidth-constrained transmission
- Onboard AI integration

### Positioning: "AI-Native Compression for Edge Analytics"

Frame as **intelligent compression** rather than general-purpose codec:
- Higher margin
- Less direct standards competition
- Clear ROI story

---

## Valuation Realism

### Reference Points

- **Deep Render / InterDigital** (Oct 2025): Cash deal, value in patents + AI talent
- Deep Render had: independent evaluations, FFmpeg/VLC integration, BD-rate vs. AV1

### To Command Better Valuation

- Quantified surprise savings on edge content
- Production readiness signals (fuzz-tested, ARM builds)
- Patent-backed metrics

### Valuation Factors

| Factor | Impact |
|--------|--------|
| JEPA + semantic features | Novelty premium |
| Surveillance ROI data | Commercial proof premium |
| Hardened FFmpeg plugin | Production readiness |
| Provisional patents | Defensibility |
| Team + advisors | Scalability signal |

---

## Team & Advisors

**If solo/small, line up advisors NOW:**

- [ ] Codec industry advisor (MPEG/JVET experience)
- [ ] IP/patent attorney
- [ ] Business development (optional for deal flow)

Acquirers like InterDigital bought teams + patents — show you can integrate.

---

## Quick-Win Checklist: This Week

| # | Action | Owner | Done |
|---|--------|-------|------|
| 1 | FFmpeg thread-safety test | | [ ] |
| 2 | FFmpeg fuzzing | | [ ] |
| 3 | ARM64 build + test | | [ ] |
| 4 | Acquire surveillance footage | | [ ] |
| 5 | Run surprise-gating benchmark | | [ ] |
| 6 | Create 2-min demo video | | [ ] |
| 7 | File provisional patents | | [ ] |
| 8 | Draft confidential teaser | | [ ] |

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
