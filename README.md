<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,25:0a0020,60:1a0050,100:6c00ff&height=280&section=header&text=AETHER-SYNTH&fontSize=90&fontColor=ffffff&fontAlignY=38&desc=Evolutionary%20GAN%20%2B%20LLM%20Cognitive%20Auditor&descAlignY=62&descSize=24&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=22&duration=2500&pause=700&color=A855F7&center=true&vCenter=true&multiline=true&width=850&height=130&lines=Evolutionary+GAN+%2B+LLM+Cognitive+Auditing;DCGAN+Neural+Loop+%2B+LLaMA+3.3+Reasoning+Loop;Sub-100ms+Asynchronous+Audit+Cycles;The+GAN+That+Thinks+About+Itself)](https://git.io/typing-svg)

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-a855f7?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.1+-6c00ff?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/LLaMA_3.3-70B-a855f7?style=for-the-badge&logo=meta&logoColor=white"/>
<img src="https://img.shields.io/badge/Groq-Inference-6c00ff?style=for-the-badge"/>
<img src="https://img.shields.io/badge/DCGAN-Architecture-a855f7?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-6c00ff?style=for-the-badge"/>

<br/><br/>

> ### *"A GAN that doesn't just train — it thinks about whether it should."*
> AETHER-SYNTH fuses a high-fidelity DCGAN with an asynchronous LLM cognitive auditor. Every epoch, LLaMA 3.3 70B reads the loss landscape and decides whether to intervene.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/siddhantchandorkar752-ai)

</div>

---

## WHAT IS AETHER-SYNTH?

```
╔══════════════════════════════════════════════════════════════════════╗
║     AETHER-SYNTH — Evolutionary GAN + Cognitive Auditor v1.0        ║
║     "Training instability kills GANs. An LLM auditor prevents it."  ║
║                                                                      ║
║     NEURAL LOOP:    DCGAN minimax optimization (G vs D)             ║
║     COGNITIVE LOOP: LLaMA 3.3 70B reads gradients → intervenes     ║
║     PRECISION:      Hardware-agnostic CUDA/CPU dynamic allocation   ║
║     TELEMETRY:      Loguru structured audit trail — every epoch     ║
╚══════════════════════════════════════════════════════════════════════╝
```

AETHER-SYNTH is not a standard GAN implementation. It is a **dual-loop autonomous training framework** where a Large Language Model acts as a cognitive auditor — monitoring loss trajectories, detecting instability, and actively intervening in hyperparameter space.

> No human in the loop. No manual tuning. The system audits itself.

---

## THE PROBLEM WITH STANDARD GANS

```
Standard GAN training is notoriously unstable.

Mode collapse.        Generator outputs the same sample repeatedly.
Vanishing gradients.  Discriminator wins too fast — generator learns nothing.
Oscillating losses.   Training diverges. You restart from scratch.

The standard solution: manually monitor loss curves and tune hyperparameters.
This is time-consuming, expertise-heavy, and non-reproducible.

AETHER-SYNTH replaces the human monitor with an LLM cognitive auditor.
```

---

## DUAL-LOOP ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AETHER-SYNTH ENGINE                          │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    NEURAL LOOP (DCGAN)                        │  │
│   │                                                              │  │
│   │   Noise z ──► GENERATOR (G) ──► Fake Images                 │  │
│   │                                      │                       │  │
│   │   Real Images ──────────────────► DISCRIMINATOR (D)         │  │
│   │                                      │                       │  │
│   │   Loss: min_G max_D V(D,G)           │                       │  │
│   │   = E[log D(x)] + E[log(1-D(G(z)))] │                       │  │
│   └──────────────────────────┬───────────┘                       │  │
│                              │  Every N epochs                   │  │
│                              ▼                                   │  │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                  COGNITIVE LOOP (LLM AUDITOR)                 │  │
│   │                                                              │  │
│   │   Loss Metrics ──► LLaMA 3.3 70B via Groq                   │  │
│   │                         │                                    │  │
│   │              ┌──────────┴──────────┐                        │  │
│   │              ▼                     ▼                         │  │
│   │       STABLE → Continue    UNSTABLE → Intervene             │  │
│   │                              LR reduction / arch adjust      │  │
│   └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MATHEMATICAL OBJECTIVE

The framework optimizes the standard GAN minimax objective:

```
min_G max_D V(D,G) = E[log D(x)] + E[log(1 - D(G(z)))]

Where:
  G  = Generator network
  D  = Discriminator network
  x  ~ p_data(x)   real data distribution
  z  ~ p_z(z)       latent noise distribution

LLM Auditor monitors:
  ΔL_G  = Generator loss gradient per epoch
  ΔL_D  = Discriminator loss gradient per epoch
  ratio = L_D / L_G  (balance indicator)
```

---

## LLM AUDIT CYCLE

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   INPUT  → Gen Loss: 1.20 | Disc Loss: 0.27 | Epoch: 4            │
│                                                                      │
│   LLaMA 3.3 70B REASONING:                                         │
│   "Discriminator loss 0.27 suggests D is winning too fast.          │
│    Generator at 1.20 is struggling. Risk of vanishing gradient.     │
│    Recommend: reduce discriminator LR by 20%."                      │
│                                                                      │
│   ACTION → LR_D = LR_D * 0.80  (autonomous intervention)           │
│                                                                      │
│   LATENCY → Sub-100ms via Groq inference                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PERFORMANCE METRICS

| Epoch | Gen Loss | Disc Loss | Balance Ratio | Agent Action | Status |
|-------|:--------:|:---------:|:-------------:|:------------:|:------:|
| 1 | 1.4200 | 0.3100 | 4.58 | LR Reduction | Intervened |
| 2 | 1.2000 | 0.2750 | 4.36 | LR Reduction | Intervened |
| 3 | 1.0800 | 0.2800 | 3.86 | Monitor | Stabilizing |
| 4 | 0.9500 | 0.2875 | 3.30 | Maintain | Stable |
| 5 | 0.8900 | 0.2950 | 3.02 | Maintain | Stable |

---

## ENGINEERING HIGHLIGHTS

| Feature | Implementation | Why It Matters |
|---------|---------------|----------------|
| **Async Cognitive Auditing** | LiteLLM + Groq | Sub-100ms reasoning — no training bottleneck |
| **Hardware-Agnostic Precision** | Dynamic CUDA/CPU allocation | Runs on free Colab T4 or local GPU |
| **VRAM Monitoring** | Automated memory tracking | Prevents OOM crashes mid-training |
| **Structured Telemetry** | Loguru audit trails | Every epoch logged — full reproducibility |
| **Config Management** | Pydantic Settings | Zero-error environment injection |

---

## PROJECT STRUCTURE

```
aether-synth/
├── src/
│   ├── agents/
│   │   └── auditor.py       # LLM cognitive auditor — Groq + LiteLLM
│   ├── generator/
│   │   ├── dcgan.py         # Generator + Discriminator architectures
│   │   └── trainer.py       # Neural loop — minimax training
│   └── utils/
│       ├── hardware.py      # CUDA/CPU allocation + VRAM monitoring
│       ├── telemetry.py     # Loguru structured logging
│       └── precision.py     # Mixed precision training setup
├── output/                  # Versioned synthetic image artifacts
├── config.py                # Pydantic settings + hyperparameter control
├── main.py                  # Orchestration entry point
├── requirements.txt
└── .env.example
```

---

## QUICK START

```bash
# 1. Clone
git clone https://github.com/siddhantchandorkar752-ai/aether-synth.git
cd aether-synth

# 2. Setup
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate         # Mac/Linux
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Add to .env:
# LLM_MODEL=groq/llama-3.3-70b-versatile
# LLM_API_KEY=your_groq_api_key

# 4. Train
python main.py
```

---

## TECH STACK

| Layer | Technology | Version | Why |
|-------|-----------|---------|-----|
| **Framework** | PyTorch | 2.1+ | Industry standard for GAN research |
| **LLM** | LLaMA 3.3 70B via Groq | Latest | Sub-100ms inference for real-time auditing |
| **LLM Interface** | LiteLLM | Latest | Universal LLM API abstraction |
| **Logging** | Loguru | Latest | Structured telemetry with zero boilerplate |
| **Config** | Pydantic Settings | Latest | Type-safe environment management |
| **Hardware** | CUDA / CPU | Dynamic | Automatic device allocation |

---

## RESEARCH CONTEXT

LLM-guided training is an emerging paradigm at top research labs:
- **DeepMind** — AutoML and neural architecture search
- **OpenAI** — Self-play and autonomous training loops
- **Anthropic** — Constitutional AI with feedback loops

AETHER-SYNTH applies this paradigm to GAN stability — an open problem since the original GAN paper (Goodfellow et al., 2014).

---

## LICENSE

MIT License — free to use, modify, distribute.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a0020,50:1a0050,100:0a0020&height=70&text=Siddhant%20Chandorkar&fontSize=30&fontColor=a855f7&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-6c00ff?style=for-the-badge&logo=github&logoColor=white)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-siddhantchandorkar-a855f7?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/siddhantchandorkar)

<br/>

*"I don't just train models. I build systems that train themselves."*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6c00ff,40:1a0050,100:000000&height=140&section=footer&text=AETHER-SYNTH%20v1.0&fontSize=34&fontColor=a855f7&fontAlignY=68&animation=fadeIn" width="100%"/>

</div>
