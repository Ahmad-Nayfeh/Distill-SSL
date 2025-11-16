\# SSL-Knowledge-Distillation-on-STL10



A project investigating the synergy of Self-Supervised Learning (SimCLR, BYOL, DINO) and Knowledge Distillation (KD) to create efficient, lightweight computer vision models. This repository implements the training of ResNet-18 teachers (pretrained via SSL) to distill knowledge to a MobileNetV3-Small student on the STL-10 dataset.



> This is a project for the \*\*CHE 567: AI in Chemical Engineering\*\* course at KFUPM.



\## 🎯 Project Goal



This project systematically investigates the combination of Self-Supervised Learning (SSL) and Knowledge Distillation (KD) to address the dual challenges of computational inefficiency and data dependency in modern deep learning.



Our key research questions are:

\* Does SSL pretraining of teachers improve knowledge distillation effectiveness compared to supervised teachers?

\* Can ensemble or progressive distillation from multiple SSL teachers outperform single-teacher distillation?

\* Can a lightweight student (MobileNetV3-Small) achieve near-teacher accuracy with significantly reduced parameters and computational cost?



\## 🔬 Methodology



The project follows a 3-stage pipeline to train the final student model.



1\.  \*\*SSL Pretraining:\*\* Three ResNet-18 backbones are pretrained on the 100,000 \*unlabeled\* images of the STL-10 dataset using \*\*SimCLR\*\*, \*\*BYOL\*\*, and \*\*DINO\*\*.

2\.  \*\*Teacher Fine-Tuning:\*\* The three SSL-pretrained backbones (plus one supervised-from-scratch baseline) are fine-tuned on the 5,000 \*labeled\* images to create four distinct teacher models.

3\.  \*\*Knowledge Distillation:\*\* The "knowledge" from the teacher models is distilled into a lightweight \*\*MobileNetV3-Small\*\* student using various strategies (single-teacher, ensemble, etc.).



\### Models \& Dataset



| Role | Architecture | Rationale |

| :--- | :--- | :--- |

| \*\*Teachers\*\* | ResNet-18 | Standard SSL backbone, balances expressiveness and efficiency. |

| \*\*Student\*\* | MobileNetV3-Small | Lightweight design for deployment efficiency. |



\* \*\*Dataset:\*\* \*\*STL-10\*\*

&nbsp;   \* \*\*Unlabeled set:\*\* 100,000 images (for SSL pretraining)

&nbsp;   \* \*\*Training set:\*\* 5,000 labeled images (for fine-tuning and distillation)

&nbsp;   \* \*\*Test set:\*\* 8,000 labeled images (for final evaluation)



\## 🛠️ Repository Structure



```

SSL-Knowledge-Distillation/

│

├── 📂 checkpoints/    # (This is where trained .pth models are saved)

├── 📂 data/            # (This is where the STL-10 dataset is downloaded)

├── 📂 logs/            # (This is where all experiment .csv logs are saved)

├── 📂 models/          # (Contains model definitions: resnet.py, mobilenet.py)

├── 📂 plots/           # (This is where final .png plots are saved)

├── 📂 training/        # (Contains all training scripts)

│   ├── train\_supervised.py  # (For baselines and fine-tuning)

│   ├── train\_ssl.py         # (For SimCLR, BYOL, and DINO pretraining)

│   └── train\_kd.py          # (For all knowledge distillation)

├── 📂 utils/           # (Helper scripts)

│   ├── dataset.py         # (Data loaders and SSL augmentations)

│   └── logger.py          # (CSV logging utility)

│

├── 📄 measure\_models.py  # (Script to measure model FLOPs and Params)

├── 📄 plot\_results.py    # (Script to generate all plots from logs)

├── 📄 run\_all\_experiments.bat # (Main script to run all 25 experiments)

└── 📄 README.md          # (You are here)

```



\## 🚀 Getting Started



\### 1. Prerequisites

\* Windows / Linux

\* Anaconda (or Miniconda)

\* NVIDIA GPU with CUDA



\### 2. Installation

1\.  \*\*Clone the repository:\*\*

&nbsp;   ```bash

&nbsp;   git clone \[https://github.com/YOUR\_USERNAME/SSL-Knowledge-Distillation.git](https://github.com/YOUR\_USERNAME/SSL-Knowledge-Distillation.git)

&nbsp;   cd SSL-Knowledge-Distillation

&nbsp;   ```

2\.  \*\*Create and activate the Conda environment:\*\*

&nbsp;   ```bash

&nbsp;   # (We use Python 3.10 for stable PyTorch compatibility)

&nbsp;   conda create -n ssl-kd python=3.10

&nbsp;   conda activate ssl-kd

&nbsp;   ```

3\.  \*\*Install PyTorch and dependencies:\*\*

&nbsp;   ```bash

&nbsp;   # Install PyTorch (check PyTorch website for your specific CUDA version)

&nbsp;   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia



&nbsp;   # Install other required libraries

&nbsp;   pip install pandas seaborn matplotlib torchmetrics thop tqdm

&nbsp;   ```



\## 📈 How to Run



This project is designed to be run from a single script that executes all experiments in order.



\### Step 1: Run All Experiments

This is the main script that will run all 16+ experiments, from pretraining the SSL models to fine-tuning them and distilling them into the final students.



\*\*Warning: This will take a very long time to run (likely 1-2 days).\*\*

```bash

\# On Windows

run\_all\_experiments.bat



\# (Note: The script is configured to run SSL pretraining for 50 epochs

\# to save time, but all experiments are included.)

```



\### Step 2: Measure Model Efficiency

To get the FLOPs and Parameter counts for the report.

```bash

python measure\_models.py

```



\### Step 3: Generate All Plots

After the experiments are done, this script will read all `.csv` files in the `logs/` directory and automatically generate a corresponding plot for each one in the `plots/` directory.

```bash

python plot\_results.py

```



\## 📊 Results \& Analysis (In Progress)



This section will be populated with the final results after all experiments are complete.



\### 1. Baseline Performance (Overfitting)





\### 2. SSL Teacher Performance

\* This section will compare the fine-tuned test accuracy of the baseline `T-SUP` (64.88%) against the SSL-pretrained teachers (`T-SimCLR`, `T-BYOL`, `T-DINO`).





\### 3. Student Distillation Performance

\* This section will compare the test accuracy of all final student models.





