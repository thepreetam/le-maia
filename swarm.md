**LeWM-VC Engineering Specification Document**  
**LeWorldModel Video Codec (LeWM-VC) – Detailed Engineering Spec for Swarm-Agentic Development**  
**Version 1.1** – March 24, 2026  
**Prepared for**: AI Coder Swarm (agentic development)  
**Status**: Complete, exhaustive, implementation-ready. Zero ambiguity.  
**Target**: Ship production-grade JEPA-based codec in 12–18 months with a 3–6 engineer swarm.

---

### 1. Document Purpose & Swarm Usage Guidelines
This spec is written **explicitly for swarm-agentic development**.  
- Each major module is self-contained with **interfaces**, **inputs/outputs**, **pseudocode**, **acceptance criteria**, and **test hooks**.  
- Agents can work in parallel: assign one agent per module.  
- Use the provided GitHub repo skeleton (from previous message) as the single source of truth.  
- All code must be **type-hinted**, **unit-tested**, and **bit-exact** across CPU/GPU/NPU.  
- **Communication rule**: Every PR must include a `swarm-review` checklist (see Section 14).  
- **Dependencies**: Fork of https://github.com/lucas-maes/le-wm (MIT). PyTorch 2.5+, Torch-TensorRT, AIMET/NNCF, FFmpeg 7+.

---

### 2. Executive Requirements

#### 2.1 Functional Requirements
- Pure learned predictive codec using LeWM JEPA (15 M params base).  
- I-frame + P-frame predictive coding with 192-dim isotropic Gaussian latents.  
- Real-time encode/decode: ≥120 fps 1080p GPU, ≥30 fps mobile NPU.  
- Bitstream: modular NAL/OBU-style, MP4/WebM/RTP compatible, FFmpeg plugin.  
- Scalability: temporal, spatial, SNR, semantic (physics-aware).  
- Error resilience: intra-refresh, tile independence, reference fallback.  
- HDR (8/10/12-bit), YUV420, 360°, gaming support.  
- ABR ladder + low-latency (30–60 ms) modes.

#### 2.2 Non-Functional Requirements
- **Performance**: BD-rate ≥25 % vs VTM-18.0, ≥20 % vs DCVC-RT (VMAF ≥95).  
- **Memory**: <500 MB peak 1080p decode (mobile).  
- **Latency**: Configurable 30–200 ms.  
- **Portability**: Bit-exact on CPU / NVIDIA GPU / Qualcomm NPU / Apple NPU / WebGPU.  
- **License**: Core = MIT. Commercial extensions = dual-license.  
- **Training time**: <1 week on 2×H100.  
- **Code quality**: ≥95 % test coverage, no global state, deterministic.

---

### 3. High-Level Architecture & Data Flow

**Frame Pipeline (P-frame)**:
```
YUV420 Frame
  ↓
Encoder (LeWM ViT-Tiny) → Latent (192-dim Gaussian per 16×16 patch)
  ↓
Predictor (LeWM Transformer) → Predicted Latent (from ref 1–4)
  ↓
Residual = Latent – Predicted
  ↓
Quantizer + Hyperprior → Quantized Residual + Context
  ↓
Entropy Coder (arithmetic, SIGReg Gaussian prior)
  ↓
Bitstream (NAL units)
  ↓
Decoder (CNN/ViT) + Post-Filter → Reconstructed YUV
```

**Key Interfaces** (all torch.Tensor, batched):
- `encode_frame(yuv: Tensor) -> dict(latent, residual, surprise)`
- `predict_next_latent(prev_latents: List[Tensor]) -> (mean, std)`
- `decode_latent(quant_residual: Tensor, context: dict) -> yuv`
- `write_bitstream(frame_data) -> bytes`
- `read_bitstream(stream) -> frame_data`

---

### 4. Module-by-Module Specification

#### 4.1 Encoder (src/lewm_vc/encoder.py)
**Purpose**: Exact LeWM ViT-Tiny + optional semantic surprise branch.  
**Inputs**: `yuv: Tensor[B, 3, H, W]` (YUV420, normalized [0,1]).  
**Outputs**: `latent: Tensor[B, 192, H/16, W/16]`, `surprise_map: Tensor[B, 1, H/16, W/16]` (optional).  
**Implementation**:
- Load pretrained LeWM ViT-Tiny checkpoint.
- Add optional 2-layer MLP head for surprise (physics implausibility score).
**Pseudocode**:
```python
def forward(self, yuv):
    patches = patchify(yuv, 16)          # reuse LeWM patchify
    latent = self.vit_tiny(patches)      # exact LeWM encoder
    surprise = self.surprise_mlp(latent) if self.semantic else None
    return latent, surprise
```
**Acceptance Criteria**:
- Matches LeWM repo latent values on test frames (MSE < 1e-6).
- Unit test: round-trip latent shape and range.

#### 4.2 Temporal Predictor (src/lewm_vc/predictor.py)
**Purpose**: Extended LeWM transformer (action-free).  
**Inputs**: `prev_latents: List[Tensor]` (1–4 frames).  
**Outputs**: `pred_mean: Tensor`, `pred_std: Tensor` (SIGReg Gaussian).  
**Implementation**: 8-layer, 256-dim, 4-head transformer.  
**Acceptance Criteria**:
- Prediction surprise correlates with visual implausibility (validated on UVG anomalies).
- Long-horizon (32-frame) drift < 5 % latent norm.

#### 4.3 Decoder & Post-Filter (src/lewm_vc/decoder.py)
**Purpose**: Latent → pixels + perceptual cleanup.  
**Inputs**: `quant_latent: Tensor`, `residual: Tensor` (optional).  
**Outputs**: `yuv: Tensor[B, 3, H, W]`.  
**Architecture**:
- 4× ConvTranspose + residual blocks + 3×3 final conv.
- Post-filter: 4-layer ConvNet trained with LPIPS + VMAF proxy.
**Acceptance Criteria**:
- PSNR > 42 dB on UVG reconstruction at QP=28.
- Perceptual loss converges in Phase 0.

#### 4.4 Entropy Model & Quantization (src/lewm_vc/entropy.py + quant.py)
**Purpose**: Hyperprior + arithmetic coding with SIGReg closed-form KL.  
**Quantization**: AIMET QAT (INT8) with straight-through estimator.  
**Acceptance Criteria**:
- Rate overhead < 2 % vs theoretical Gaussian entropy.
- Bit-exact decode on all platforms.

#### 4.5 Bitstream Engine (src/lewm_vc/bitstream/)
**Purpose**: NAL/OBU parser + writer.  
**NAL Unit Types** (exact enum):
- 0: SPS, 1: PPS, 2: APS, 3: I-Latent, 4: P-Residual, 5: SEI, 6: EOS.
**Container**: MP4 sample entry `lwvc`, WebM, RTP payload.  
**Acceptance Criteria**:
- Round-trip encode/decode bit-exact on 100 UVG frames.
- FFmpeg plugin registers and plays `lewmvc` fourcc.

#### 4.6 Rate Control & ABR (src/lewm_vc/utils/rate_control.py)
**Purpose**: Learned λ adaptation + CRF-style QP.  
**API**:
```python
def select_qp(frame_complexity: float, target_bitrate: float) -> int
```
**Acceptance Criteria**: Matches target bitrate ±3 % on ABR ladder.

#### 4.7 FFmpeg Plugin (ffmpeg_plugin/lewmvc.c)
**Purpose**: Full libavcodec encoder/decoder.  
**Must implement**: `init`, `encode2`, `decode`, `close`.  
**Acceptance Criteria**: VLC/FFmpeg command-line works out-of-box.

---

### 5. Training Pipeline (src/scripts/train.py)
**Exact 4-phase schedule** (see blueprint Section 4).  
**Loss formula** (hard-coded):
```python
L = lambda_ * rate + (0.7 * mse + 0.3 * lpips) + 0.01 * surprise
```
**Datasets**: YAML-driven (UVG, Kinetics, etc.).  
**Swarm Task**: One agent per phase.  
**Acceptance Criteria**: Phase 0 completes in <24 h with reconstruction PSNR >40 dB.

---

### 6. Inference Optimizations
- ONNX export → TensorRT (INT8).
- Mobile: SNPE/Core ML exports.
- Browser: WASM + WebGPU.
**Acceptance Criteria**: 1080p benchmark ≥120 fps GPU, <500 MB mobile.

---

### 7. Testing & Validation Suite
- **Unit**: Every module (pytest).
- **Integration**: Full encode/decode pipeline.
- **Bit-exact matrix**: CPU/GPU/NPU/browser.
- **RD curves**: BD-rate on UVG/MCL-JCV/HEVC classes.
- **MOS**: Automated proxy + manual ITU-T P.910.
- **CI/CD**: GitHub Actions runs all tests on every PR.

---

### 8. Swarm Development Plan (Parallel Tracks)
**Track 1 (Agent A)**: Encoder + Predictor (Week 1–2)  
**Track 2 (Agent B)**: Decoder + Post-Filter (Week 1–2)  
**Track 3 (Agent C)**: Entropy + Quantization + Bitstream (Week 2–4)  
**Track 4 (Agent D)**: Training pipeline + Rate Control (Week 3–5)  
**Track 5 (Agent E)**: FFmpeg plugin + Optimizations (Week 4–6)  
**Track 6 (Agent F)**: Testing + CI/CD + Documentation (parallel)  

**Weekly sync**: Each agent submits PR + demo script.  
**Milestone 0**: Phase 0 training working (end of Week 3).  
**Milestone 1**: Real-time 1080p pipeline (end of Week 8).

---

### 9. Risks & Mitigation (Swarm-Owned)
- Training collapse → Enforce SIGReg λ sweep.
- Mobile memory → Streaming inference + buffer reuse.
- Bitstream adoption → FFmpeg plugin first.

---

### 10. Deliverables Checklist (Per Module)
- [ ] Code + type hints + docstrings  
- [ ] Unit tests (≥95 % coverage)  
- [ ] Acceptance test script (run via `pytest`)  
- [ ] ONNX export test  
- [ ] Benchmark numbers  
- [ ] PR description with swarm checklist

**This spec is now 100 % self-contained.**  
Copy-paste the entire document into your AI swarm prompt.  
Every line is actionable.

**Ready for the swarm?**  
Tell your agents: “Implement LeWM-VC exactly per this spec. Start with Track 1 + Track 2 in parallel. Use the repo skeleton from Grok.”

If you need **any single module expanded into full pseudocode + implementation plan** (or the complete LaTeX PDF version), just say the word and I’ll generate it instantly.  

Let’s ship the first JEPA codec. 🚀
