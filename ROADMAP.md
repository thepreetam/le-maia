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

### Key Differentiators

| Feature | Why It Matters |
|---------|---------------|
| JEPA World-Model Foundation | Stable pixel-to-latent prediction via SIGReg |
| Semantic Surprise Detection | Physics-implausibility gating for quality/rate |
| SIGReg Gaussian Entropy | Near-ideal entropy prior, minimal rate overhead |
| LPIPS Perceptual Post-Filter | Superior subjective quality |
| Production FFmpeg Integration | Clean deployment path |

### Timing Advantage

- InterDigital acquired Deep Render (Oct 2025) for AI-native codec tech
- MPEG/JVET actively exploring NNVC and Beyond-VVC (H.267)
- World-model funding surging ($1B+ in early 2026)
- LeWM-VC positioned as "next logical acquisition" after Deep Render

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

## 4. Controlled Validation & Early Traction (Weeks 6–12)

### Internal deployment pilot

- [ ] Integrate LeWM-VC into one real production pipeline:
  - Internal video storage or streaming
  - Analytics or content delivery
  - Security footage with surprise detection
- [ ] Track end-to-end gains:
  - Storage/bandwidth savings
  - Decode speed on target hardware (CPU/mobile/edge)
  - Qualitative wins from perceptual post-filter
  - Auto-flagging implausible frames in security footage
- [ ] Measure over weeks of real traffic: total cost reduction, quality consistency

### Selective NDA-based evaluations

- [ ] Identify 2–4 target partners/customers who value your differentiators
- [ ] Possible targets:
  - Video platform/cloud providers (heavy compression needs)
  - Drone/autonomous systems companies (surprise detection)
  - Hardware vendors looking for neural codec IP
- [ ] Share: binaries, APIs, or black-box access + side-by-side clips/metrics
- [ ] **Never share source code**
- [ ] Gather feedback: integration ease, hardware compatibility, use-case fit

### Refine based on data

- [ ] Tune hyperparameters from pilot results:
  - Latent dim, predictor depth, rate controller
- [ ] Add targeted features:
  - Low-power mobile decode mode
  - Downstream analysis directly on latents (no full decode)

---

## 5. Business Execution (3–12+ months)

### If building as product/feature

- [ ] Scale to production:
  - Real-time/low-latency modes
  - Multi-GPU training
  - Deployment: Docker, edge devices
- [ ] Expand use cases:
  - Surveillance (surprise-aware bit allocation)
  - Cloud storage compression
  - Hybrid workflows (video search/analysis on latents)
- [ ] Build team:
  - Hardware acceleration (tensor cores optimization)
  - Compliance testing (regulated industries)
  - Reliability hardening

### If pursuing licensing/partnership/acquisition

- [ ] Prepare pitch package:
  - One-pager
  - Confidential technical whitepaper (architecture overview, benchmark curves vs. HEVC/AV1, demo videos)
  - No code or implementation details
- [ ] Approach ecosystem players:
  - Companies investing in learned/neural compression
  - World-model video players (robotics, AR/VR, streaming)
- [ ] Engage standards bodies:
  - Monitor JVET/MPEG Neural Network Video Coding (NNVC)
  - Beyond-VVC (H.267) efforts
  - File patents before contributing

### Defensive/offensive IP strategy

- [ ] Continue patent filings:
  - Semantic surprise + SIGReg entropy in predictive world-model codec
- [ ] Monitor patent landscape in learned video coding
- [ ] Watch competitor publications (transformer-based, predictive latent codecs)

### Risk mitigation

| Risk | Mitigation |
|------|------------|
| Fast-moving research | Iterate quickly on longer sequences, hierarchical prediction |
| Standards competition | File patents first, engage indirectly |
| Resource constraints | Budget for compute, legal, hardware ports |

---

## 7. Acquisition Positioning & Outreach (Months 3–9)

**Goal**: Turn benchmarks + JEPA foundation into a compelling, high-valuation asset.

### Market timing

- InterDigital acquired Deep Render (AI video codec startup) October 2025
- GTT Group divesting AI-assisted video compression patents
- High interest in AI video/world-model (Runway, Luma AI raising hundreds of millions)
- Acquirers seeking: compression, temporal prediction, semantic understanding

### Polish the acquisition package (4–6 weeks)

#### Confidential teaser / one-pager

**Core innovation:**
- First JEPA-based end-to-end video codec
- Built on stable LeWM world model foundation
- SIGReg Gaussian entropy modeling
- Semantic surprise detection for physics-aware quality/rate control
- Perceptual LPIPS post-filter

**Performance edge:**
- BD-rate savings vs. HEVC/AV1
- Perceptual quality gains
- Runtime, memory, FFmpeg integration metrics
- Domain-specific wins (high-motion, surveillance)

**Defensibility:**
- Provisional patents filed
- Clean integration path
- Team expertise

#### Data room preparation (under NDA)

- Side-by-side demo videos/clips
- Full RD curves + raw logs
- Ablation studies (JEPA predictor vs. traditional motion compensation)
- Integration examples (FFmpeg wrapper)
- High-level architecture overview (no source)

#### Valuation anchors

- Reference Deep Render / InterDigital deal
- Factor in: LeWM timing advantage, semantic features, production-ready elements
- Early-stage neural codec IP commands strong multiples with clear gains + moat

### Identify and prioritize targets

#### Primary acquirers

| Tier | Company Type | Examples |
|------|-------------|----------|
| 1 | Video IP/licensing leaders | InterDigital, Dolby, Adeia |
| 2 | Big Tech with video/cloud | Google, Meta, Amazon, Netflix, Apple, ByteDance |
| 3 | Hardware/edge | Qualcomm, Nvidia, Intel, ARM |
| 4 | World-model/robotics | Embodied AI companies |

#### Approach strategy

- Start with 2–3 "warm" fits (already investing in neural codecs or world models)
- Use intermediaries if needed: investment bankers, patent brokers
- Avoid public leaks

### Controlled outreach & due diligence

**NDA-first approach:**
- Warm connections, corporate development teams, or conferences
- Emphasize "exclusive evaluation opportunity"
- Offer virtual/in-person black-box evaluations on their content

**Demo targets:**
- "20–30% effective bitrate reduction on 4K streaming test set"
- Better perceptual scores at equivalent bitrate

**Diligence readiness:**
- Team bios
- IP status summary
- Scalability roadmap
- Risk/mitigation notes

**Timeline:**
| Milestone | Target |
|-----------|--------|
| Initial meetings | 2–4 months |
| Term sheets | 6–9 months |

### Parallel de-risking

- [ ] Continue internal pilots (surveillance, cloud storage ROI stories)
- [ ] Monitor MPEG NNVC and Beyond-VVC efforts quietly
- [ ] Consider key hires: codec optimization, business development
- [ ] Engage advisors with AI acquisition experience
- [ ] Backup: licensing deals or strategic partnerships

### Key risks & mitigation

| Risk | Mitigation |
|------|------------|
| Competition | JEPA + semantic surprise combo remains differentiated |
| Timing | Move while LeWM momentum and deals are fresh |
| Valuation pressure | Strong private data + patents help |

---

## Target Acquirer Profiles

### InterDigital
- Already acquiring AI codec tech (Deep Render)
- Leading beyond VVC/HEVC positioning
- Warm fit for video IP licensing model

### Dolby
- Heavy in compression licensing
- Interest in neural/hybrid approaches
- Strong monetization infrastructure

### Google/Meta/Netflix
- Internal video platforms need cost savings
- World-model interest (especially Meta)
- Can pay premium for clear gains

### Qualcomm/Nvidia/ARM
- Edge/mobile deployment value
- Tensor core optimization synergies
- Hardware vendor interest in codec IP

### Robotics/Embodied AI
- JEPA predictor aligns with physics-aware modeling
- Surprise detection for autonomous systems
- Growing investment in world models

---

## Domain Focus Priorities

### 1. Surveillance & Security (Highest Priority)

**Focus**: Edge-AI cameras, VSaaS platforms, long-term storage, real-time analytics

**ROI hook**:
> "30–50% effective bitrate reduction on normal footage while auto-highlighting implausible events — no extra compute for separate anomaly detection."

**Why it matches:**
- Semantic surprise detection = automatic anomaly flagging
- Aggressive compression on predictable scenes
- Physics priors detect out-of-distribution events
- Latent-space analysis without full decode

**2026 trends:**
- Real-time edge analytics
- Low-bandwidth proactive monitoring
- AI-driven incident response

### 2. Drones / UAV / Remote Sensing & Defense

**Focus**: ISR (Intelligence, Surveillance, Reconnaissance), autonomous flight

**ROI hook**:
> "Physics-aware rate control reduces satellite/5G link costs dramatically."

**Why it matches:**
- Bandwidth-constrained transmission
- World-model prediction for stable long-GOP compression
- High-motion, low-light robustness
- Edge deployment optimization

**Market:**
- Defense budgets exploding for autonomous ISR
- Onboard AI integration

### 3. Robotics & Embodied AI

**Focus**: Video-based planning, teleoperation, physics-aware world models

**ROI hook**:
> "Compressed latents enable faster downstream analysis without full decode."

**Why it matches:**
- Direct synergy with JEPA-style world models
- V-JEPA lineage alignment (Meta)
- Surprise detection for real-world robustness
- World-model funding surging ($1B+ in early 2026)

### 4. Cloud Streaming / OTT / Video Archiving

**Focus**: Massive video libraries, Netflix/YouTube-style AI encoding

**ROI hook**:
> "BD-rate gains + VMAF/LPIPS leadership on long-form content."

**Why it matches:**
- LPIPS-trained perceptual post-filter
- Learned λ adaptation
- Massive storage/transmission savings

### 5. Edge / Mobile / Hardware-Constrained Video

**Focus**: On-device decode, semiconductor integration

**Why it matches:**
- FFmpeg plugin ready
- Quantization-aware training
- FPS/memory optimizations

---

## Tiered Acquirer Strategy

### Tier 1: Video IP / Licensing Powerhouses (Best Fit, Fastest Path)

| Company | Why Target | Approach |
|---------|-----------|----------|
| **InterDigital** | Just acquired Deep Render Oct 2025 for AI-native codec tech. JEPA + surprise = cleaner extension. | NDA + patent portfolio discussion |
| Dolby | Compression licensing heavyweight, expanding AI portfolios | Perceptual quality wins |
| Adeia | Patent monetization, AI compression interest | Novel IP positioning |
| Nokia | Video IP portfolio, Beyond-VVC positioning | Standards complementarity |
| Sisvel | Patent pools, pure IP play | Quick liquidity option |

### Tier 2: Big Tech Video / Cloud Platforms

| Company | Why Target | Approach |
|---------|-----------|----------|
| **Meta** | V-JEPA lineage, LeCun connection, world-model investment | Cultural fit + latent analysis synergies |
| Google/YouTube | Massive bandwidth bills, AI encoding pipelines | Direct compression ROI |
| Amazon/AWS | Prime Video, cloud storage costs | Storage/bandwidth savings |
| Netflix | Video quality optimization expertise | Perceptual quality focus |
| ByteDance/TikTok | Video-heavy, efficiency-focused | Scale + efficiency |
| Apple | History of quiet AI video acquisitions | Premium positioning |

### Tier 3: Hardware / Semiconductor Leaders

| Company | Why Target | Approach |
|---------|-----------|----------|
| Qualcomm | Edge/mobile inference, 5G video | FPS/memory on mobile |
| Nvidia | Tensor core optimization, DRIVE platform | Hardware synergy |
| Intel | Edge AI, video analytics | Integration ease |
| ARM | Mobile silicon, IoT edge | Edge deployment |

### Tier 4: Robotics / Physical AI Pure-Plays

**Profile**: Companies building embodied AI or their acquirers

**Why**: Physics-aware video understanding, world-model integration

---

## Positioning Narrative

**Lead with Surveillance + Drones** in confidential teaser:
- Easiest to demo with side-by-side clips
- Strongest differentiation ("this isn't just another neural codec")
- Clear ROI story

**Core narrative**:
> "JEPA-world-model codec that turns recent stable pixel-to-latent prediction into production compression with built-in semantic intelligence — positioned exactly where InterDigital is betting post-Deep Render."

**Valuation signal**:
- Reference Deep Render deal (AI talent + patents)
- NNVC track in MPEG/JVET momentum
- Explosive world-model funding (2026)

---

## Outreach Strategy

### Phase 1: Immediate (Weeks 1–4)
1. **InterDigital** — NDA + patent portfolio discussion
2. **Meta** — world-model synergy angle
3. **Qualcomm** — hardware edge play

### Phase 2: Expansion (Weeks 5–8)
- Dolby, Adeia (licensing)
- Google, Amazon (streaming)
- Defense contractors (drones)

### Demo package
- Side-by-side clips: original vs. LeWM-VC at equivalent bitrate
- Surprise detection demo: auto-flagging anomalous events
- RD curves: vs. H.264, H.265, AV1
- FPS/memory benchmarks on target hardware

---

## 6. Longer-term Technical Roadmap

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
