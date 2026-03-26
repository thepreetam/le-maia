# LeWM-VC Roadmap

**Goal**: Acquisition positioning — shift from working prototype to acquirable asset within 3–6 months.  
**Reference**: Deep Render / InterDigital (Oct 30, 2025) — cash tuck-in focused on AI talent, patents, and accelerating "AI-native" video research. Bulk of value assigned to patents/goodwill. LeWM-VC positions as the logical next step: world-model priors + semantic intelligence on learned compression.

---

## Current Status

- Working prototype: ViT encoder, temporal predictor, quantization, entropy coding, NAL bitstream
- FFmpeg plugin (C wrapper) implemented
- 159 tests passing, CI/CD configured

---

## Immediate Priorities (Weeks 1–2)

### 1. Surveillance ROI Benchmark + Demo (Highest Impact / Centerpiece)

**Why**: This delivers the most compelling commercial proof for acquirers (InterDigital, Qualcomm, defense/drone).

| Task | Details |
|------|---------|
| **Footage** | Acquire 100+ hours of diverse surveillance footage (indoor/outdoor, day/night, low-motion vs. events). Use **only** public datasets or rights-cleared internal clips. Prepare clean data provenance statement for data room. |
| **Encoding** | Encode with surprise-gating **ON** vs. **OFF**. Compare against x265 (HEVC) and SVT-AV1 at multiple bitrates. |
| **Metrics** | Bitrate savings on normal segments; quality (VMAF, LPIPS) on anomaly segments; compute overhead (ms/frame) for surprise detection; latency impact. |

**Output**:
- Spreadsheet with full results (include total footage, % normal/anomaly segments)
- **2-minute split-screen demo video**: Original | LeWM-VC with gating (surprise overlay) | LeWM-VC without gating  
  (Clean cuts, no audio — make the trade-off instantly visible to non-technical teams)

**Decision gate**: If savings <15% on normal segments, reframe narrative toward "intelligent analytics-friendly compression" (emphasizing anomaly preservation and downstream value) rather than pure bitrate leadership.

---

### 2. Production Hardening (Parallel)

**FFmpeg C wrapper must be production-ready** (thread-safe, resilient, multi-platform) before any NDA demos.

| Requirement | Status |
|-------------|--------|
| Thread-safety (multiple parallel encodes/decodes) | [ ] |
| Seek/timestamp handling (test with `-ss` flag) | [ ] |
| Frame drop handling (graceful degradation) | [ ] |
| Fuzzing (inject random corruption, no crashes) | [ ] |
| ARM64 build (Apple Silicon M-series, Raspberry Pi) | [ ] |
| ARM32 build (Raspberry Pi OS) | [ ] |
| Mobile SoC (representative Qualcomm/MediaTek target) | [ ] |
| p99 decode latency per frame | [ ] |
| CPU-only fallback (no GPU required for decode) | [ ] |
| ONNX export path documented + Tensor Core/NPU-friendly layer notes | [ ] |

---

### 3. JEPA Predictor Ablation (One Week)

**Decisive test** — adjust narrative based on results.  
Compare 8-layer SIGReg predictor against:  
- Optical flow + residual coding  
- Temporal VAE conditioning  
- Basic transformer motion (no SIGReg)  

**Metrics**: Rate-distortion (PSNR, MS-SSIM, VMAF), long-horizon stability (30+ frame GOPs), surprise detection synergy.  

**Decision**:  
- If wins clearly → Keep "stable world-model latent coding" framing.  
- If not → Lead with SIGReg training stability + semantic surprise as core moat; de-emphasize "JEPA" in teaser.

---

### 4. Provisional Patents (Week 1 – *before any outreach*)

File to secure priority. Attach early RD curves, ablation results, and surveillance savings data.

**Claim Families**:
1. Semantic surprise-driven latent-space rate allocation  
2. NAL extensions with world-model priors  
3. Overall JEPA/SIGReg + perceptual post-filter pipeline  

**Result**: Enables "patent pending" in teaser and data room.

---

## Week 3–6: NDA Package & Outreach

### One-Pager Teaser Hook

> **AI-native edge video compression with built-in anomaly awareness** — semantic surprise detection delivers measurable bitrate savings on predictable surveillance scenes while preserving quality for events.

**Lead with**:
- One clear graph: surveillance bitrate savings vs. H.265  
- Bullet differentiators (SIGReg stability, hardened FFmpeg integration, LPIPS post-filter)  
- Link to 2-min demo (under NDA)  
- "Patent pending" note + team/advisor summary

### Data Room (Under NDA)

| Category | Contents |
|----------|----------|
| Demo | 2-min split-screen video + raw clips |
| RD Data | Full curves vs. H.264/H.265/AV1, raw logs, ablation results |
| Hardware | FFmpeg metrics (ARM, latency, fuzz, CPU-only) |
| Technical | High-level architecture (**no source code**) |
| Data | Training data provenance statement (rights-cleared only) |
| IP | Patent filings, status ("patent pending") |
| Team | Bios, advisor list, integration willingness |

### NDA Discipline

- Use your own standard NDA for first contact.  
- Offer a **black-box API** (upload clip → compressed output via simple Flask/FastAPI wrapper) instead of binaries. This maintains control while enabling real evaluation.

### Outreach Sequence

(Only after advisor is secured)  
1. **InterDigital** — warmest strategic fit post-Deep Render (AI talent + patents focus)  
2. **Qualcomm** — edge NPU/hardware synergy  
3. **Defense / drone** — surveillance ROI story resonates strongly  

Use advisor introductions where available. Corporate development calendars fill quickly — start as soon as benchmark + demo are ready.

---

## Risk Mitigation (Acquirer Lens)

| Risk | Mitigation |
|------|------------|
| **Standards pressure** (NNVC / Beyond-VVC, CfP expected 2026–2027) | Position as complementary: predictive latent + semantic awareness for edge/analytics use cases that standards may lag on. Patents gain value if similar ideas appear. |
| **Hardware dependency** | ARM builds + CPU fallback documented; graceful degradation shown; NPU mapping notes included. |
| **Training data rights** | Rights-cleared surveillance footage only; explicit provenance statement. |
| **Team / scalability** | Secure MPEG/JVET-experienced codec advisor *before* outreach; highlight integration willingness. |
| **Valuation inflation** | Anchor to Deep Render precedent but differentiate with surveillance-specific proof and production hooks. Use use-case-relative gains (strongest on low-motion edge). |

---

## Exit Timing & Decision Gate

### Target Timeline

| Milestone | Target |
|-----------|--------|
| Initial NDA conversations | End of Month 2 |
| Term sheets | Month 3–4 |
| Close | Month 4–6 |

### Hard Gate: Month 3

**Define your minimum exit number now** (e.g., all-cash amount + must-have terms like team role/integration period).  

If by Month 3 you have:  
- Quantified surprise ROI (≥15% savings on normal segments)  
- Hardened plugin (all checks green)  
- At least one serious NDA conversation with follow-up  

**Then**: Push for term sheets.  
If not: Reassess (build more vs. fast exit). NNVC/ECM window is moving but not infinite.

**Provocative Question**:  
If a solid mid-eight-figure exit ($25M–$35M range) in 3 months is on the table, do you take it? Decide internally now — it will focus your efforts. Upper end ($40M+) requires competitive bidding or a signed pilot.

---

## Positioning: "AI-Native Edge Video Compression"

| Element | Why It Works |
|---------|--------------|
| "AI-native" | Signals learned/hybrid approach, not traditional codec |
| "Edge" | Targets fastest-growing market (cameras, IoT, analytics) |
| "Built-in anomaly awareness" | Concrete, demo-able, defensible |
| "Semantic surprise detection" | The technical moat — easy to explain |

**Core Narrative**:  
"LeWM-VC turns stable pixel-to-latent prediction into production compression with built-in semantic intelligence. Semantic surprise detection automatically identifies physics-implausible events and allocates bits accordingly. Positioned exactly where InterDigital is betting post-Deep Render."

---

## Valuation Anchors

- **Deep Render reference**: Cash deal (Oct 30, 2025) with vast majority of value in patents/goodwill + AI talent.  
- **To command better**: Surveillance-specific ROI data (commercial proof), hardened FFmpeg plugin (production readiness), patent pending on surprise (defensibility), team + advisor (scalability signal).  

Use-case-relative gains instead of blanket claims.

---

## Team & Advisors

If solo/small team:  
- [ ] **Codec industry advisor** (MPEG/JVET experience) — secure *before* outreach for credibility and warm intros  
- [ ] IP/patent attorney (already engaged for filings)  
- [ ] Optional: business development advisor for deal flow  

Acquirers buy teams + patents — show you can integrate smoothly.

---

## Quick-Win Checklist – This Week

**Note**: Parallelize where possible. Core must-haves by end of Week 1: hardening basics, footage acquisition, patents, advisor, and minimum exit definition. Full benchmark + demo by end of Week 2.

| # | Action | Done |
|---|--------|------|
| 1 | FFmpeg thread-safety + fuzzing tests | [ ] |
| 2 | ARM64/ARM32 builds + CPU-only validation | [ ] |
| 3 | Acquire rights-cleared surveillance footage + provenance statement | [ ] |
| 4 | Run surprise-gating benchmark | [ ] |
| 5 | Create 2-min split-screen demo | [ ] |
| 6 | File provisional patents | [ ] |
| 7 | Draft confidential one-pager teaser | [ ] |
| 8 | Line up codec advisor (on board) | [ ] |
| 9 | Define minimum exit number (internal) | [ ] |
| 10 | Build black-box API stub | [ ] |

---

## Technical Summary (for Data Room Reference)

**Key Parameters (v1)**:
- Latent dim: 192
- Patch size: 16×16
- Encoder: 6 layers, 3 heads
- Predictor: 8 layers, 4 heads
- Target GOP: 32 frames

**Why defensible**:
1. SIGReg stability (only 2 losses, no EMA teacher)
2. Gaussian latents (near-ideal entropy prior)
3. Semantic surprise (physics implausibility detection — unique)
4. Predictive world-model architecture — temporal prediction without explicit motion estimation (subject to ablation confirmation)
5. Clean, production-ready FFmpeg C wrapper

---

## Confidentiality

**This document is proprietary and intended for internal planning only.**  
Do not share externally without authorization.

*Last updated: March 2026*
