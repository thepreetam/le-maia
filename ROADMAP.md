# LeWM-VC Roadmap

**Goal**: Acquisition positioning — shift from prototype to acquirable asset.

**Reference**: Deep Render / InterDigital (Oct 30, 2025) — cash tuck-in focused on AI talent, patents, and accelerating "AI-native" video research. Bulk of value in patents/goodwill. LeWM-VC positions as the logical next step: world-model priors + semantic intelligence on learned compression.

---

## Current Status

- Working prototype: ViT encoder, temporal predictor, quantization, entropy coding, NAL bitstream
- FFmpeg plugin (C wrapper) implemented
- 159 tests passing, CI/CD configured

---

## Immediate Priorities (This Week – Week 2)

### 1. Surveillance ROI Benchmark + Demo (Highest Impact)

**This is the centerpiece of the acquisition narrative.**

#### Surveillance Footage Test

- [ ] Acquire 100+ hours of diverse surveillance footage
  - Indoor/outdoor, day/night, low-motion vs. events
  - Public datasets or rights-cleared internal clips only
  - **Prepare clean data provenance statement for data room**
- [ ] Encode with surprise-gating ON vs. OFF
- [ ] Compare against strong anchors: x265 HEVC, SVT-AV1

#### Metrics to Capture

```
Surveillance Footage Test Results:
- Total footage: XX hours
- Normal segments: XX% (surprise-gating enabled)
- Anomaly segments: XX% (full quality mode)
- Bitrate savings: XX% on normal segments
- Quality delta: VMAF/LPIPS scores
- Compute overhead: XX ms per frame for surprise detection
- Latency impact: XX ms added
```

#### Demo Video (2 minutes)

- [ ] Split-screen: Original | LeWM-VC with gating (surprise overlay) | LeWM-VC without gating
- [ ] Make the trade-off instantly visible to non-technical CD teams
- [ ] Clean cuts, no audio

---

### 2. Production Hardening (Parallel)

**FFmpeg C wrapper must be production-ready before NDA demos.**

#### FFmpeg Checklist

- [ ] **Thread-safety**: Multiple parallel encodes/decodes
- [ ] **Seek/timestamp**: Test with `-ss` flag
- [ ] **Frame drops**: Verify graceful handling
- [ ] **Fuzzing**: Inject random corruption, verify graceful failure
- [ ] **ARM64**: Build on Apple Silicon M-series
- [ ] **ARM32**: Build on Raspberry Pi OS
- [ ] **Mobile SoC**: Representative target (Qualcomm, MediaTek)
- [ ] **Latency**: Measure p99 decode time per frame
- [ ] **CPU-only fallback**: Verify no GPU required for decode

#### Hardware Mapping Notes

- [ ] ONNX export path documented (even if prototype)
- [ ] Tensor Core / NPU-friendly layer notes
- Shows ecosystem awareness for Qualcomm/Nvidia conversations

---

### 3. JEPA Ablation (One Week)

**Decisive test — adjust narrative based on results.**

Compare 8-layer SIGReg predictor against:

| Baseline | What to Test |
|----------|--------------|
| Optical flow + residual | Standard motion compensation |
| Temporal VAE conditioning | Simple temporal latent conditioning |
| Basic transformer motion | Transformer without SIGReg |

**Metrics**:
- Rate-distortion (PSNR, MS-SSIM, VMAF)
- Long-horizon stability (30+ frame GOPs)
- Surprise detection synergy

**Decision**:
- If wins clearly → Keep "stable world-model latent coding" framing
- If not → Reframe around SIGReg training stability + semantic surprise as core moat

---

### 4. Provisional Patents (Parallel, Week 1)

**File NOW — before any public discussion.**

#### Claims to File

1. **Semantic surprise-driven latent-space rate allocation**
2. **NAL extensions with world-model priors**
3. **Overall JEPA/SIGReg + perceptual post-filter pipeline**

#### Attach to Filing

- Early RD curves
- Ablation results
- Surveillance savings data

This strengthens prior-art arguments and enables "patent pending" in teaser.

---

## Week 3–6: Clean NDA Package & Outreach

### One-Pager Hook

> "AI-native edge video compression with built-in anomaly awareness — semantic surprise detection delivers measurable bitrate savings on predictable surveillance scenes while preserving quality for events."

**Lead with**:
- One clear graph: surveillance bitrate savings vs. H.265
- Split-screen demo link (under NDA)

### Data Room Checklist

| Category | Contents |
|----------|----------|
| Demo | 2-min surveillance split-screen video |
| RD Data | Full curves vs. H.264/H.265/AV1, raw logs |
| Ablations | JEPA predictor vs. baselines |
| Hardware | FFmpeg metrics (ARM, latency, fuzz results) |
| Technical | High-level architecture (NO source) |
| Data | Training data provenance statement |
| IP | Patent claims, filing status ("patent pending") |
| Team | Bios, advisor list, integration willingness |

### NDA Discipline

- Use your own standard NDA first
- **Black-box API option**: Upload clip → compressed output
- Maintains control, enables early evaluations

### Outreach Sequence

1. **InterDigital** — warmest strategic fit post-Deep Render
2. **Qualcomm** — NPU/hardware edge synergy
3. **Defense/Drone** — surveillance ROI story resonates

Use advisor intros where available.

---

## Risk Mitigation (Acquirer Lens)

| Risk | Acquirer Concern | Mitigation |
|------|------------------|------------|
| **Standards pressure** | NNVC / Beyond-VVC (H.267) targeting 2027 | Position as complementary: predictive latent + semantic for edge/analytics that standards may lag on. Patents gain value if similar ideas appear in NNVC. |
| **Hardware questions** | Does it only run on NVIDIA? | ARM builds + CPU fallback documented; graceful degradation shown; NPU mapping notes included. |
| **Training data** | Rights-cleared? | Rights-cleared surveillance footage only; explicit provenance statement. |
| **Team / scalability** | Can you scale? | Highlight integration willingness; line up MPEG/JVET-experienced advisor before outreach. |
| **Valuation realism** | Inflated claims? | Anchor to Deep Render (talent + patents) but differentiate with harder surveillance-specific proof and production hooks. Use-case-relative gains (strongest on low-motion edge) instead of blanket claims. |

---

## Exit Timing & Decision Gate

### Target Timeline

| Milestone | Target |
|-----------|--------|
| Initial NDA conversations | End of Month 2 |
| Term sheets | Month 3–4 |
| Close | Month 4–6 |

### Hard Gate: Month 3

**Define your minimum now** (number + must-have terms like team role).

If you have:
- [ ] Quantified surprise ROI ✓
- [ ] Hardened plugin ✓
- [ ] Positive NDA feedback ✓
- [ ] At least one serious discussion ✓

**Then**: Push for term sheets.

If NOT:
- Reassess: build more vs. fast exit
- NNVC window is not infinite

### The Provocative Question

A solid mid-eight-figure exit in 3 months is attractive if:
- Numbers are right
- Team fit is right
- De-risks fast-moving NNVC/ECM landscape

**Holding 12 months only makes sense with**:
- Clear traction signals (paid pilots, stronger benchmarks)
- No cooling in AI-video IP appetite

---

## Positioning: "AI-Native Edge Video Compression"

### Why This Framing Works

| Element | Why It Matters |
|---------|---------------|
| "AI-native" | Signals learned/hybrid approach, not traditional codec |
| "Edge" | Targets fastest-growing market (edge AI, cameras, IoT) |
| "Built-in anomaly awareness" | Concrete, demo-able, defensible |
| "Semantic surprise detection" | The technical moat — easy to explain |

### Core Narrative

> "LeWM-VC turns recent stable pixel-to-latent prediction into production compression with built-in semantic intelligence. Semantic surprise detection automatically identifies physics-implausible events and allocates bits accordingly. Positioned exactly where InterDigital is betting post-Deep Render."

---

## Valuation Anchors

### Reference: Deep Render / InterDigital

- **Oct 30, 2025** cash deal
- Value: patents/goodwill > product
- Focus: AI talent + patents + "AI-native" acceleration

### To Command Better Valuation

| Factor | Impact |
|--------|--------|
| Surveillance-specific ROI data | Commercial proof premium |
| Hardened FFmpeg plugin | Production readiness |
| Patent pending on surprise | Defensibility |
| Team + advisor | Scalability signal |

---

## Team & Advisors

### If Solo/Small — Line Up Advisors NOW

- [ ] **Codec industry advisor** (MPEG/JVET experience)
  - Signals scalability during diligence
  - Opens warm doors
- [ ] **IP/patent attorney**
- [ ] **Business development** (optional for deal flow)

Acquirers like InterDigital bought teams + patents.

---

## Quick-Win Checklist: This Week

| # | Action | Done |
|---|--------|------|
| 1 | FFmpeg thread-safety test | [ ] |
| 2 | FFmpeg fuzzing | [ ] |
| 3 | ARM64 build + test | [ ] |
| 4 | Acquire surveillance footage | [ ] |
| 5 | Run surprise-gating benchmark | [ ] |
| 6 | Create 2-min split-screen demo | [ ] |
| 7 | File provisional patents | [ ] |
| 8 | Draft confidential teaser | [ ] |
| 9 | Line up codec advisor | [ ] |
| 10 | Define minimum exit number | [ ] |

---

## Technical Summary

### Why LeWM-VC is Defensible

1. **SIGReg stability**: Only 2 losses (MSE + SIGReg), no EMA teacher
2. **Gaussian latents**: 192-dim isotropic Gaussians = near-ideal entropy prior
3. **Semantic surprise**: Physics implausibility detection (unique)
4. **JEPA architecture**: Temporal prediction without explicit motion estimation
5. **Clean integration**: Production-ready FFmpeg C wrapper

### Key Parameters (v1)

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
