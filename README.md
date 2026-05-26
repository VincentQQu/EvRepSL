# EvRepSL: Event-Stream Representation via Self-Supervised Learning for Event-Based Vision

This repository contains the official implementation of the paper **"EvRepSL: Event-Stream Representation via Self-Supervised Learning for Event-Based Vision"** [IEEE TIP](https://ieeexplore.ieee.org/document/10758409), [arXiv](https://arxiv.org/abs/2412.07080). EvRepSL introduces a novel self-supervised approach for generating event-stream representations, which significantly improves the quality of event-based vision tasks.

## Overview

EvRepSL provides **event-stream encodings** for downstream vision tasks. The repo includes:

- **Hand-crafted representations** — voxel grid, two-channel, four-channel, TORE, EvRep
- **Learned generators** — RepGen (EvRep → EvRepSL) and **PIE-Net / PIE-Net-Lite** (voxel → PIEM)

Hand-crafted methods convert raw events into tensors directly. Learned generators map those tensors to richer, task-ready representations using pretrained neural networks.

## Learned Representation Generators

| Generator | Input | Output | Weights |
|-----------|-------|--------|---------|
| **RepGen** | EvRep `[3, H, W]` | EvRepSL features | [Google Drive](https://drive.google.com/drive/folders/1poN9xeTUrJhpBgHV2xGRxkR1Ymx4IbXt?usp=sharing) (`RepGen.pth`) |
| **PIE-Net** | Voxel grid `[5, H, W]` | PIEM `[5, H, W]` | `pip install event-pienet` |
| **PIE-Net-Lite** | Voxel grid `[5, H, W]` | PIEM `[5, H, W]` | `pip install event-pienet` |

**PIE-Net** is the next generation of [E2HQV](https://github.com/VincentQQu/E2HQV) and uses **Probabilistic Intensity-Event Mapping (PIEM)**. Full implementation, real-time demo, and benchmarks: **[github.com/VincentQQu/pie-net](https://github.com/VincentQQu/pie-net)**.

### PIEM representation channels

PIE-Net outputs a 5-channel **PIEM representation** (not just a reconstructed frame):

| Channel | Key | Meaning |
|---------|-----|---------|
| 0 | `mean_exp_z` | Expected log-intensity change (Z mean) |
| 1 | `var_exp_z` | Uncertainty of Z |
| 2 | `k` | Learned PIEM scaling parameter |
| 3 | `mean_f1` | Reconstructed intensity frame |
| 4 | `var_f1` | Per-pixel frame uncertainty |

```text
events  →  voxel [5,H,W]  →  PIE-Net / PIE-Net-Lite  →  PIEM [5,H,W]  →  downstream task
events  →  EvRep [3,H,W]  →  RepGen                  →  EvRepSL       →  downstream task
```

## Repository Structure

- **event_representations.py**: Hand-crafted encodings (voxel, EvRep, TORE, …) and wrappers for learned generators (RepGen, PIE-Net, PIE-Net-Lite).
- **models.py**: RepGen architecture (EffWNet).
- **RepGen.pth**: RepGen weights ([Google Drive](https://drive.google.com/drive/folders/1poN9xeTUrJhpBgHV2xGRxkR1Ymx4IbXt?usp=sharing)).
- **PIE-Net / PIE-Net-Lite**: Provided via [`event-pienet`](https://pypi.org/project/event-pienet/) (weights included).

## Getting Started

### Prerequisites

```bash
pip3 install -r requirements.txt
```

Or manually:

```bash
pip3 install torch numpy
pip3 install event-pienet   # for PIE-Net / PIE-Net-Lite generators
```

### Quick examples

**EvRep → EvRepSL (RepGen):**

```python
from event_representations import events_to_EvRep, load_RepGen, EvRep_to_EvRepSL

ev_rep = events_to_EvRep(xs, ys, ts, pols, resolution=(320, 240))
model = load_RepGen(device="cuda")
evrepsl = EvRep_to_EvRepSL(model, ev_rep[np.newaxis], device="cuda")
```

**Events → PIEM (PIE-Net):**

```python
from event_representations import events_to_PIEM_representation, reset_piem_states

piem = events_to_PIEM_representation(xs, ys, ts, pols, resolution=(320, 240), variant="pie-net")
rep = piem["piem"]          # [5, H, W] stacked PIEM representation
mean_z = piem["mean_exp_z"] # [1, H, W] individual maps also available

reset_piem_states()         # call between independent sequences
```

**PIE-Net-Lite (faster):**

```python
piem = events_to_PIEM_representation(xs, ys, ts, pols, variant="pie-net-lite")
```

Run the demo script:

```bash
python3 event_representations.py
```

## To Cite

EvRepSL paper:

<pre>
@article{qu2024evrepsl,
  title={EvRepSL: Event-Stream Representation via Self-Supervised Learning for Event-Based Vision},
  author={Qu, Qiang and Chen, Xiaoming and Chung, Yuk Ying and Shen, Yiran},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}

@inproceedings{qu2024e2hqv,
  title={E2HQV: High-Quality Video Generation from Event Camera via Theory-Inspired Model-Aided Deep Learning},
  author={Qu, Qiang and Shen, Yiran and Chen, Xiaoming and Chung, Yuk Ying and Liu, Tongliang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4632--4640},
  year={2024}
}
</pre>
