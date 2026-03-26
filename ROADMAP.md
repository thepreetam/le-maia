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

### 1. Next-Generation Video Compression (Core Fit)

**Focus**: AI/hybrid codecs with better perceptual quality or bitrate savings vs. HEVC/AV1/VVC

**Why it matches:**
- Semantic surprise detection enables physics-aware bit allocation
- Lower quality on predictable frames, higher on anomalous events
- Production hooks: FFmpeg plugin, rate control

**Use cases:**
- Streaming platforms
- Cloud video storage
- Broadcast
- Mobile delivery

**Standards context:**
- MPEG/JVET exploring NNVC and Beyond-VVC (H.267)
- InterDigital acquiring Deep Render signals IP-heavy players betting on AI-native compression

### 2. Embodied AI / Robotics & World Models

**Focus**: JEPA-style predictors for temporal forecasting, physics simulation, surprise handling

**Why it matches:**
- 8-layer transformer predictor + semantic surprise detection
- Action anticipation, long-horizon planning
- Robustness to physics-implausible events
- V-JEPA 2 alignment (zero-shot robot control)

**Use cases:**
- Robotic manipulation
- Autonomous drones/vehicles
- Simulation for training

### 3. Surveillance / Security / Edge Video Analytics

**Focus**: Physics-aware quality assurance, dynamic rate allocation for anomalous events

**Why it matches:**
- Surprise detection auto-optimizes bitrate
- Latent-space analysis without full decode
- Flag anomalies in security footage

**Use cases:**
- Drone footage
- Security cameras
- Low-bandwidth remote monitoring

### 4. Cloud / Streaming Optimization

**Focus**: Perceptual delivery with LPIPS-trained post-filter + learned λ adaptation

**Why it matches:**
- User-perceived quality focus
- Storage/transmission cost reduction
- Clean FFmpeg integration path

---

## Tiered Acquirer Strategy

### Tier 1: Best Fit (High Likelihood of Interest)

| Company | Profile | Approach Angle |
|---------|---------|----------------|
| InterDigital | AI-native push, post-VVC positioning | BD-rate gains, standards complementarity |
| Dolby | Compression licensing heavyweight | Perceptual quality wins |
| Adeia | Patent monetization | Novel IP in learned compression |
| Nokia | Video IP portfolio | JEPA novelty for standards |
| Sisvel | Patent pools | Pure IP play |

### Tier 2: Strong Strategic Buyers

| Company | Profile | Approach Angle |
|---------|---------|----------------|
| Google/YouTube | Massive video infrastructure | Direct compression savings |
| Meta | V-JEPA investment, world models | Latent-space advantages |
| Amazon/AWS | Prime Video, cloud storage | ROI on bandwidth costs |
| Netflix | Video quality optimization | Perceptual quality focus |
| ByteDance/TikTok | Video-heavy platform | Scale + efficiency gains |
| Qualcomm | Edge/mobile inference | FPS/memory on mobile |
| Nvidia | Tensor core optimization | Hardware synergy |
| ARM | Mobile silicon | Edge deployment value |

### Tier 3: Emerging Fits

| Company | Profile | Approach Angle |
|---------|---------|----------------|
| Robotics/Embodied AI | Physical agent simulation | Surprise detection for robustness |
| Patent brokers | GTT-style divestitures | Pure IP liquidity |

---

## Positioning by Profile

### For IP/Licensing Companies
- Stress patentable novelty (JEPA + surprise in codec context)
- Potential SEP contribution to future standards
- Patent portfolio strengthening

### For Big Tech/Platforms
- Highlight private benchmark wins on their content types
- High-motion streaming, drone video examples
- Integration ease

### For Hardware
- Provide FPS/memory numbers on target devices
- Quantization-aware training details
- Tensor core optimization paths

---

## Outreach Strategy

### Initial outreach (2–4 months)
1. Select 2–3 Tier 1 targets (warm fits)
2. Use: corporate development, advisors, intermediaries
3. Lead with NDA + teaser

### Demo package
- "20–30% effective bitrate reduction on 4K streaming"
- Better perceptual scores at equivalent bitrate
- Side-by-side clips under evaluation

### Diligence readiness
- Team bios
- IP status summary
- Scalability roadmap
- Clean RD curve data

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
