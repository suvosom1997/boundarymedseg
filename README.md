
# BoundaryMedSeg

**Unified Boundary-Aware Medical Image Segmentation Across Modalities**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🧠 Overview

Medical image segmentation across diverse modalities (ultrasound, MRI, dermoscopy, endoscopy) faces critical challenges—each domain demands custom architectures, limits generalization, and hinders clinical deployment. **BoundaryMedSeg** addresses this with a unified, transformer-enhanced framework that consistently delivers **high-accuracy segmentation with precise boundary delineation**.

<p align="center">
  <img src="https://raw.githubusercontent.com/suvosom1997/boundarymedseg/main/docs/architecture_diagram.png" width="80%">
</p>

## 🚀 Key Contributions

- 🔍 **Cross-Modality Segmentation** using a **single architecture** across:
  - BUSI (Ultrasound)
  - BraTS 2020 (MRI)
  - ISIC 2018 (Dermoscopy)
  - Kvasir-SEG (Endoscopy)
  - CVC-ClinicDB (Colonoscopy)
- 📐 **PVTv2 Encoder + Enhanced FPN** for deep hierarchical features
- 🧱 **Residual Dense Blocks (RDBs)** in decoder for contextual fusion
- 🎯 **CBAM Attention** for adaptive channel/spatial enhancement
- ✂️ **Active Boundary Guidance (ABG)** module with edge supervision for crisp boundary refinement
- ⚙️ **Combined Dice + Focal Tversky + Boundary BCE Loss** for robust optimization
- 🌐 **Domain-agnostic training protocol** — no changes to architecture

## 📊 Performance Highlights

| Dataset       | Dice Score ↑ | IoU ↑     | Boundary Accuracy ↑ |
|---------------|--------------|-----------|----------------------|
| **BUSI**      | 97.40%       | 95.81%    | High                 |
| **BraTS 2020**| 92.30%       | 89.50%    | High                 |
| **ISIC 2018** | 91.65%       | 85.40%    | High                 |
| **Kvasir-SEG**| 94.05%       | 90.78%    | High                 |
| **ClinicDB**  | 94.02%       | 89.41%    | High                 |

---

## 🧩 Architecture Summary



Image Input
↓
PVTv2 Encoder
↓
Enhanced FPN
↓
Progressive Residual Decoder (RDBs)
↓
CBAM Attention
↓
Active Boundary Guidance Module
↓
Final Segmentation Output




boundarymedseg/
├── dataset\_download.py     # Script to download/prep datasets (auto/manual)
├── losses.py               # Dice + Focal Tversky + BCE Loss functions
├── model.py                # Full BoundaryMedSeg architecture
├── train.py                # Training pipeline across datasets
├── utils.py                # Data preprocessing & metrics
├── LICENSE
└── README.md



## 📥 Datasets

Use `dataset_download.py` to download:
- ✅ [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
- ✅ [CVC-ClinicDB](https://polyp.grand-challenge.org/databases/)

**Manual download instructions included** for:
- 📦 [BUSI Dataset](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
- 📦 [ISIC 2018](https://challenge.isic-archive.com/data/)
- 📦 [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html)



## 📜 Citation

If you use this repository or its ideas, please cite:

```
@misc{boundarymedseg2025,
  title={BoundaryMedSeg: A Unified Framework for Boundary-Aware Medical Image Segmentation},
  author={Your Name(s)},
  year={2025},
  note={Under Submission}
}
```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

* [PVTv2](https://github.com/whai362/PVT)
* [CBAM](https://arxiv.org/abs/1807.06521)
* [Focal Tversky Loss](https://arxiv.org/abs/1810.07842)

