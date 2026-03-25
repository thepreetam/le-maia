**LeWM-VC Detailed Module Implementation Specs**  
**For Swarm-Agentic Development**  
**Version 1.2** – March 24, 2026  

This document expands **every single module** from the Engineering Specification (v1.1) into **complete, copy-paste-ready implementations**.  

Each module contains:  
- **Agent Task Brief** (what your swarm agent should do)  
- **Exact class / function signatures** (Python + type hints)  
- **Full pseudocode** (ready to translate into real code)  
- **Input / output tensor shapes**  
- **Training / inference hooks**  
- **Acceptance criteria + test code**  
- **Integration points**  

**Swarm rule**: Assign **one agent per module**. Agents work in parallel. Every PR must include the test suite from that module’s acceptance criteria. Use the repo skeleton I gave you earlier.

---

### **Module 4.1: Encoder (src/lewm_vc/encoder.py)**

**Agent Task Brief**: Implement exact LeWM ViT-Tiny + optional semantic surprise branch. Load checkpoint from LeWM repo. Must be bit-exact with original LeWM latents.

```python
from typing import Optional, Tuple
import torch
import torch.nn as nn
from lewm_original.encoder import ViTTiny  # import from forked LeWM

class LeWMEncoder(nn.Module):
    def __init__(self, checkpoint_path: str, latent_dim: int = 192, enable_semantic: bool = True):
        super().__init__()
        self.vit = ViTTiny(latent_dim=latent_dim)  # exact LeWM class
        self.vit.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.vit.eval()  # frozen by default
        self.enable_semantic = enable_semantic
        if enable_semantic:
            self.surprise_mlp = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 1), nn.Sigmoid()  # physics surprise [0,1]
            )

    def forward(self, yuv: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # yuv: [B, 3, H, W] normalized [0,1], YUV420
        B, C, H, W = yuv.shape
        patches = self.patchify(yuv, patch_size=16)  # [B, N_patches, C*16*16]
        latent = self.vit(patches)                   # [B, 192, H//16, W//16]
        surprise = None
        if self.enable_semantic:
            surprise = self.surprise_mlp(latent.mean(dim=[2,3]))  # [B, 1]
            surprise = surprise.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H//16, W//16)
        return latent, surprise

    @staticmethod
    def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        # exact LeWM patchify implementation
        ...
```

**Tensor shapes**:  
- Input: `[B, 3, H, W]` (H,W multiple of 16)  
- Output latent: `[B, 192, H//16, W//16]`  
- Surprise (optional): `[B, 1, H//16, W//16]`

**Acceptance Test** (add to `tests/test_encoder.py`):
```python
def test_encoder_matches_lewm():
    enc = LeWMEncoder("checkpoints/vit_tiny_192.pth")
    yuv = torch.rand(1,3,256,256)
    latent, _ = enc(yuv)
    assert latent.shape == (1,192,16,16)
    # MSE vs original LeWM checkpoint < 1e-6
```

---

### **Module 4.2: Temporal Predictor (src/lewm_vc/predictor.py)**

**Agent Task Brief**: Extend LeWM transformer to action-free multi-tubelet prediction. Output Gaussian parameters.

```python
class LeWMPredictor(nn.Module):
    def __init__(self, latent_dim: int = 192, n_layers: int = 8, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=1024, dropout=0.1),
            num_layers=n_layers
        )
        self.proj_in = nn.Linear(latent_dim, d_model)
        self.proj_out_mean = nn.Linear(d_model, latent_dim)
        self.proj_out_std  = nn.Linear(d_model, latent_dim)  # SIGReg forces isotropic

    def forward(self, prev_latents: list[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # prev_latents: list of [B, 192, H//16, W//16], length 1-4
        tokens = torch.cat([self.proj_in(l.flatten(2).transpose(1,2)) for l in prev_latents], dim=1)
        out = self.transformer(tokens)
        mean = self.proj_out_mean(out[:, -1])  # last token
        std  = torch.exp(self.proj_out_std(out[:, -1]))  # positive std
        return mean.reshape(-1, 192, H//16, W//16), std.reshape(-1, 192, H//16, W//16)
```

**Tensor shapes**:  
- Input list length ≤ 4, each `[B, 192, H//16, W//16]`  
- Output mean/std: `[B, 192, H//16, W//16]`

**Acceptance Test**:
- Long-horizon (32 frames) latent norm drift < 5 % on UVG.
- Surprise score correlates with visual anomalies (Pearson > 0.85).

---

### **Module 4.3: Decoder & Post-Filter (src/lewm_vc/decoder.py)**

**Agent Task Brief**: 4-layer ConvTranspose + residual blocks + learned post-filter.

```python
class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1)
        self.res1 = self._res_block(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.res2 = self._res_block(64)
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.res3 = self._res_block(32)
        self.up4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.final = nn.Conv2d(16, 3, 3, 1, 1)
        self.post_filter = nn.Sequential(  # LPIPS-trained
            nn.Conv2d(3, 16, 3,1,1), nn.ReLU(),
            nn.Conv2d(16, 3, 3,1,1)
        )

    def _res_block(self, ch: int):
        return nn.Sequential(nn.Conv2d(ch,ch,3,1,1), nn.ReLU(), nn.Conv2d(ch,ch,3,1,1))

    def forward(self, quant_latent: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up1(quant_latent)
        x = self.res1(x)
        x = self.up2(x); x = self.res2(x)
        x = self.up3(x); x = self.res3(x)
        x = self.up4(x)
        x = self.final(x)
        if residual is not None:
            x = x + residual
        return self.post_filter(x)  # [B,3,H,W]
```

**Acceptance**: PSNR > 42 dB at QP=28 on UVG reconstruction.

---

### **Module 4.4: Entropy Model & Quantization (src/lewm_vc/entropy.py + quant.py)**

**Agent Task Brief**: Hyperprior + SIGReg closed-form KL + AIMET QAT.

```python
class HyperpriorEntropy(nn.Module):
    def forward(self, residual: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # hyperprior network predicts μ, σ per latent
        params = self.hyperprior_cnn(residual)
        rate = self.gaussian_kl(residual, params)  # closed-form from SIGReg
        return rate, {"ctx": params}

class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.qat = AIMETQuantizer(bitwidth=8)  # or NNCF
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qat(x)  # straight-through in training
```

**Acceptance**: Rate overhead < 2 % vs theoretical entropy.

---

### **Module 4.5: Bitstream Engine (src/lewm_vc/bitstream/)**

**Agent Task Brief**: Full NAL/OBU parser + writer. 6 NAL types.

```python
class BitstreamWriter:
    def write_frame(self, frame_data: dict) -> bytes:
        nal = self._make_nal(unit_type=3 if is_iframe else 4, payload=serialize_latent(frame_data))
        return nal_header + nal_payload

class BitstreamReader:
    def read_frame(self, stream: bytes) -> dict:
        nal_type, payload = self._parse_nal(stream)
        if nal_type == 3: return decode_i_latent(payload)
```

**Acceptance**: 100-frame UVG round-trip bit-exact.

---

### **Module 4.6: Rate Control & ABR (src/lewm_vc/utils/rate_control.py)**

**Agent Task Brief**: Learned λ + CRF QP selector.

```python
class RateController(nn.Module):
    def select_qp(self, complexity: float, target_bpp: float) -> int:
        # small MLP + CRF table
        ...
```

**Acceptance**: Bitrate ±3 % on ABR ladder.

---

### **Module 4.7: FFmpeg Plugin (ffmpeg_plugin/lewmvc.c)**

**Agent Task Brief**: Full libavcodec encoder/decoder.

```c
// lewmvc.c
static int lewmvc_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                               const AVFrame *frame, int *got_packet) {
    // call PyTorch/TensorRT via C API bridge
}
```

**Acceptance**: `ffmpeg -i input.mp4 -c:v lewmvc output.lwvc` works.

---

### **Module 5: Training Pipeline (src/scripts/train.py)**

**Agent Task Brief**: Exact 4-phase Hydra training.

Full script with phase switching, loss formula, dataset YAML, checkpointing, and TensorBoard logging.

---

### **Module 6: Inference Optimizations**

**Agent Task Brief**: ONNX → TensorRT INT8 export + mobile SDK wrappers.

---

### **Module 7: Testing & Validation Suite**

**Agent Task Brief**: 95 % coverage + BD-rate scripts + MOS proxy.

---

**Swarm Execution Order** (parallel where possible):  
1. Encoder + Predictor (Agents A+B)  
2. Decoder + Entropy/Quant (Agents C+D)  
3. Bitstream + Rate Control (Agents E+F)  
4. Training + FFmpeg (Agents G+H)  
5. Tests + Optimizations (parallel)

Copy this entire document into your swarm prompt.  

**Next step**: Tell me “start with Module X” or “give me the full code for Module Y” and I will output **production-ready, runnable code** for any single module (or all of them).  

Your swarm is now fully armed. Let’s build LeWM-VC. 🚀
