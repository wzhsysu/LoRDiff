# LoRDiff

**Official implementation of the paper:**
**"Robust Low-Light Image Enhancement In the Wild via Data Synthesis and Generative Diffusion Priors"** [Paper](https://doi.org/10.1016/j.patcog.2026.113336)

**Authors:** Zhihua Wang, Qinghua Lin, Feiyang Liu, Weixia Zhang, Wei Zhou

## 🌟 Introduction

Low-light image enhancement (LLIE) often struggles with real-world applicability due to the scarcity of paired training data and the complexity of diverse degradations. 

**LoRDiff** addresses these challenges by combining an **ISP-guided data synthesis pipeline** with a **Generative Diffusion Prior**. Specifically, we propose:
1.  **ISP-Guided Synthesis:** A pipeline that models sensor physics (RAW domain) to generate virtually unlimited paired training data with realistic degradations (noise, color shifts, JPEG artifacts).
2.  **LoRDiff Model:** A Low-light Residual Diffusion model that adapts pre-trained text-to-image diffusion models (Stable Diffusion) using Low-Rank Adaptation (LoRA).

This approach allows the model to capture complex degradations and produce visually pleasing, perceptually consistent enhancements, outperforming SOTA methods especially in challenging "in the wild" scenarios.

## Checkpoints
The weight file can be downloaded from the following link：[Google Drive](https://drive.google.com/drive/folders/1IWqACBZgauw80d7HpuFMbMTRr2fo9OQm?usp=sharing)
