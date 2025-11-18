# SSL-Knowledge-Distillation-on-STL10

A project investigating the synergy of Self-Supervised Learning (SimCLR, BYOL, DINO) and Knowledge Distillation (KD) to create efficient, lightweight computer vision models. This repository implements the training of ResNet-18 teachers (pretrained via SSL) to distill knowledge to a MobileNetV3-Small student on the STL-10 dataset.

> This is a project for the **CHE 567: AI in Chemical Engineering** course at KFUPM.

## 🎯 Project Goal

This project systematically investigates the combination of Self-Supervised Learning (SSL) and Knowledge Distillation (KD) to address the dual challenges of computational inefficiency and data dependency in modern deep learning.

Our key research questions are:
* Does SSL pretraining of teachers improve knowledge distillation effectiveness compared to supervised teachers?
* Can ensemble or progressive distillation from multiple SSL teachers outperform single-teacher distillation?
* Can a lightweight student (MobileNetV3-Small) achieve near-teacher accuracy with significantly reduced parameters and computational cost?

## 🔬 Methodology

The project follows a 3-stage pipeline to train the final student model.

1.  **SSL Pretraining:** Three ResNet-18 backbones are pretrained on the 100,000 *unlabeled* images of the STL-10 dataset using **SimCLR**, **BYOL**, and **DINO**.
2.  **Teacher Fine-Tuning:** The three SSL-pretrained backbones (plus one supervised-from-scratch baseline) are fine-tuned on the 5,000 *labeled* images to create four distinct teacher models.
3.  **Knowledge Distillation:** The "knowledge" from the teacher models is distilled into a lightweight **MobileNetV3-Small** student using various strategies (single-teacher, ensemble, etc.).

### Models & Dataset

| Role | Architecture | Rationale |
| :--- | :--- | :--- |
| **Teachers** | ResNet-18 | Standard SSL backbone, balances expressiveness and efficiency. |
| **Student** | MobileNetV3-Small | Lightweight design for deployment efficiency. |

* **Dataset:** **STL-10**
    * **Unlabeled set:** 100,000 images (for SSL pretraining)
    * **Training set:** 5,000 labeled images (for fine-tuning and distillation)
    * **Test set:** 8,000 labeled images (for final evaluation)

## 🛠️ Repository Structure

```
SSL-Knowledge-Distillation/
│
├── 📂 checkpoints/    # (This is where trained .pth models are saved)
├── 📂 data/            # (This is where the STL-10 dataset is downloaded)
├── 📂 logs/            # (This is where all experiment .csv logs are saved)
├── 📂 models/          # (Contains model definitions: resnet.py, mobilenet.py)
│   ├── mobilenet.py
│   ├── resnet.py
├── 📂 plots/           # (This is where final .png plots are saved)
├── 📂 training/        # (Contains all training scripts)
│   ├── train_supervised.py  # (For baselines and fine-tuning)
│   ├── train_ssl.py         # (For SimCLR, BYOL, and DINO pretraining)
│   └── train_kd.py          # (For all knowledge distillation)
├── 📂 utils/           # (Helper scripts)
│   ├── dataset.py         # (Data loaders and SSL augmentations)
│   └── logger.py          # (CSV logging utility)
│
├── 📄 measure_models.py  # (Script to measure model FLOPs and Params)
├── 📄 plot_results.py    # (Script to generate all plots from logs)
├── 📄 generate_csv_summary.py    # (Script to generate final_results_summary.csv from logs)
├── 📄 run_all_experiments.bat # (Main script to run all 25 experiments)
└── 📄 README.md          # (You are here)
```

## 🚀 Getting Started

### 1. Prerequisites
* Windows / Linux
* Anaconda (or Miniconda)
* NVIDIA GPU with CUDA

### 2. Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ahmad-Nayfeh/Distill-SSL.git
    cd Distill-SSL
    ```
2.  **Create and activate the Conda environment:**
    ```bash
    # (We use Python 3.10 for stable PyTorch compatibility)
    conda create -n ssl-kd python=3.10
    conda activate ssl-kd
    ```
3.  **Install PyTorch and dependencies:**
    ```bash
    # Install PyTorch (check PyTorch website for your specific CUDA version)
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # Install other required libraries
    pip install pandas seaborn matplotlib torchmetrics thop tqdm
    ```

## 📈 How to Run

This project is designed to be run from a single script that executes all experiments in order, in addition to executing `measure_models.py`, `plot_results.py`, and `generate_csv_summary.py`.

**Warning: This might take a very long time to run (likely 1-2 days: the main bottleneck is in the experiments of `SSL PRETRAINING`).**
```bash
# On Windows
run_all_experiments.bat
```
