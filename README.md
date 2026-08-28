# 🎥 ECCV 2026 - Controlling Motion Transfer in Diffusion Transformers via Attention Heads

<p align="center">
  <a href="https://sunyj-hxppy.github.io/halo"><img src="https://img.shields.io/badge/Project-Page-brightgreen" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/diffusers-0.30.2-orange" alt="diffusers">
  <img src="https://img.shields.io/badge/base-CogVideoX--5B-9cf" alt="CogVideoX-5B">
  <img src="https://img.shields.io/badge/training--free-✔-success" alt="Training-free">
</p>

<p align="center">
  <img src="assets/fig1_final.png" width="100%" alt="HALO teaser">
</p>
<p align="center">
  <em>HALO identifies motion- and structure-specialized attention heads within video DiTs, generating videos that follow the target prompt while remaining motion- and structure-aligned with the reference — achieving accurate motion transfer.</em>
</p>

Official implementation of **"Controlling Motion Transfer in Diffusion Transformers via Attention Heads" (HALO)**.

> Sunyoung Jung¹\*, Jiwoo Park¹²\*, Yoonseok Choi¹, Kyobin Choo¹, Ming-Hsuan Yang³, Seong Jae Hwang¹†
>
> ¹Yonsei University · ²LG Electronics · ³University of California, Merced
> <br>\*Equal contribution · †Corresponding author

🔗 **Project page:** https://sunyj-hxppy.github.io/halo

---

## 📋 Overview

Extending video **Diffusion Transformers (DiTs)** to **motion transfer** — generating a video that follows a *reference motion* while adhering to a *target prompt* — is challenging because DiTs use a *unified* attention that entangles motion and structure, obscuring how each is encoded.

**HALO** answers *how motion and structure are encoded in video DiTs* through a **head-level analysis**, and turns that insight into a **training-free** motion transfer framework (no parameter updates):

| Insight | What we observe | How HALO uses it |
| --- | --- | --- |
| 🎯 **Motion-Specific Heads** | A subset of heads shows strong cross-frame patch correspondences → clear **displacement maps** that faithfully reflect motion. | **Semantic-aware displacement optimization** — refines motion cues from these heads using semantic correspondences from diffusion features (DIFT). |
| 🧱 **Structure-Specialized Heads** | Another subset shows **low attention-map entropy** (sharp diagonal patterns) → concentrated structural information. | **Selective structural feature injection** — injects reference attention features through low-entropy heads to preserve spatial layout. |

This head-level control yields motion transfer that is both **motion-faithful** and **structurally aligned** with the reference, while remaining **interpretable**.

---

## 📁 Project Structure

```
.
├── inference_ourmethod_final.py     # 🚀 Main entry point (motion transfer inference)
├── final.sh                         # 📜 Example run script
├── environment.yml                  # 🐍 Conda environment (name: halo)
├── assets/                          # 🎬 Reference clips + teaser & sample outputs
│   ├── bmx-trees.mp4 · camel.mp4 · lucia.mp4 · rallye.mp4
│   └── sample_outputs/                 # Example result GIFs
├── configs/
│   └── guidance_config_inject12.yaml   # ⚙️ Guidance / injection hyperparameters
├── guidance_utils/
│   ├── custom_transformer_inject.py    # ControlledTransformer (DiT with head-aware control)
│   ├── custom_modules_inject_final.py  # ModuleWithGuidance / InjectionProcessor (feature injection)
│   ├── motion_flow_utils_final.py      # Displacement / motion-flow computation & guidance loss
│   ├── custom_embeddings.py            # Rotary positional embeddings
│   └── utils.py                        # Entropy-based structure-head selection (mask_head_byentropy)
└── dift_models/                     # DIFT semantic feature extractor (Stable Diffusion 2-1)
    ├── dift_sd.py                      # SDFeaturizer
    ├── unet.py / unet_blocks.py / attention.py / resnet.py / motion_module.py ...
    └── pipeline_cove.py
```

---

## 🛠️ Installation

**Prerequisites:** Linux · NVIDIA GPU with CUDA · [Anaconda / Miniconda](https://docs.conda.io/).

```bash
# 1. Clone
git clone https://github.com/SunYJ-hxppy/HALO_official_code.git
cd HALO_official_code

# 2. Create the conda environment (name: halo, Python 3.10)
conda env create -f environment.yml
conda activate halo
```

> 💡 The environment includes `diffusers==0.30.2`, `transformers==4.55.0`, `decord`, `imageio`, `omegaconf`, and `open-clip-torch`, among others.

**Models used (downloaded automatically from 🤗 Hugging Face on first run):**
- `THUDM/CogVideoX-5b` — base video DiT (default; use `--model 2b` for `THUDM/CogVideoX-2b`)
- `stabilityai/stable-diffusion-2-1` — DIFT semantic features for displacement refinement

> ⚠️ A high-VRAM GPU is recommended (the 5B model uses `bfloat16`; `enable_gradient_checkpointing: True` in the config reduces memory usage).

---

## 🚀 Usage

### Quick start

```bash
bash final.sh
```

which runs:

```bash
python inference_ourmethod_final.py \
    --video_path ./assets/drift-turn.mp4 \
    --prompt "A jet plane racing through the sky, with smoke trailing behind it." \
    --video_length 24 \
    --output_path tmp_result \
    --config_path ./configs/guidance_config_inject12.yaml
```

> 📌 Place your reference video (`.mp4`, or a directory of `.png` frames) at `--video_path`. The generated result is written to `--output_path`.

### Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `-v, --video_path` | *(required)* | Reference video to transfer motion **from** (`.mp4` or dir of `.png`). |
| `-p, --prompt` | *(required)* | Target text prompt for the new generation. |
| `--model` | `5b` | Base DiT: `5b` or `2b`. |
| `-n, --video_length` | `49` | Number of leading frames loaded from the reference video. |
| `--config_path` | — | YAML with guidance / injection hyperparameters. |
| `--output_path` | `output_path` | Directory for the generated video and intermediates. |
| `--loss_type` | `flow` | Guidance signal: `flow`, `moft`, or `smm`. |
| `--no_guidance` | off | Disable semantic-aware displacement guidance. |
| `--no_injection` | off | Disable structural feature (KV) injection. |
| `--seed` | `1` | Random seed. |
| `--save_format` | `mp4` | `mp4`, `gif`, or `frames`. |
| `--negative_prompt` | *(see code)* | Negative prompt for generation. |

Run `python inference_ourmethod_final.py --help` for the full list.

---

## 🎬 Results

HALO transfers the **turning motion** of a drifting car onto a completely different subject while following the target prompt — here, an airplane executing the same trajectory:

<table>
  <tr>
    <td align="center"><b>Reference</b></td>
    <td align="center"><b>HALO (Ours)</b></td>
  </tr>
  <tr>
    <td><img src="assets/sample_outputs/drift-turn.gif" width="100%"></td>
    <td><img src="assets/sample_outputs/ours_driftturn.gif" width="100%"></td>
  </tr>
</table>

> 🎞️ Reference clips (`bmx-trees`, `camel`, `lucia`, `rallye`) are available under [`assets/`](assets) to reproduce more results.

---

## ⚙️ Configuration

Key fields in `configs/guidance_config_inject12.yaml`:

```yaml
lr: [0.002, 0.001]              # LR range over the guided timestep range
optimization_steps: 5           # Guidance optimization steps per denoising step
guidance_timestep_range: [50, 38]  # Timesteps where guidance is applied
injection_blocks: [15, ..., 26] # Transformer blocks used for structural injection
guidance_blocks_5b: [20]        # Blocks providing motion guidance (CogVideoX-5B)
num_inference_steps: 50
guidance_scale: 7
height: 480
width: 720
prop_motion: 0.04               
argmax_motion_flow: True        # Argmax over reference motion flow
```

---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{jung2025halo,
  title   = {Controlling Motion Transfer in Diffusion Transformers via Attention Heads},
  author  = {Jung, Sunyoung and Park, Jiwoo and Choi, Yoonseok and Choo, Kyobin
             and Yang, Ming-Hsuan and Hwang, Seong Jae},
  year    = {202ECCV}
}
```

---

