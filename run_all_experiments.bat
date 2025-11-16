@ECHO OFF
ECHO "================================================="
ECHO "STARTING ALL 25 PROJECT EXPERIMENTS"
ECHO "================================================="
ECHO "Activating conda environment..."
CALL conda activate CHE-AI-Project

:: ====================================================================
:: --- GROUP 1 & 2: SUPERVISED BASELINES ---
:: (These should already be done, but we run them again for clean logs)
:: ====================================================================

:ECHO "--- IGNORING: Group 1 (B1) Student Baseline (Exp 1/25) BECAUSE IT WAS RUN ---"
:: ECHO "--- RUNNING: Group 1 (B1) Student Baseline (Exp 1/25) ---"
:: python -m training.train_supervised --model mobilenetv3 --epochs 50 --save_name B1_student_baseline.pth

:ECHO "--- IGNORING: Group 2 (T-SUP) Teacher Baseline (Exp 2/25) BECAUSE IT WAS RUN ---"
:: ECHO "--- RUNNING: Group 2 (T-SUP) Teacher Baseline (Exp 2/25) ---"
:: python -m training.train_supervised --model resnet18 --epochs 50 --save_name T-SUP_teacher_baseline.pth

:: ====================================================================
:: --- GROUP 2 (Continued): SSL PRETRAINING (THE SLOW PART) ---
:: --- We will use 100 epochs for SSL as a good default ---
:: ====================================================================

ECHO "--- RUNNING: Group 2 (T-SimCLR) SSL Pretraining (Exp 3/25) ---"
python -m training.train_ssl --method simclr --epochs 100 --batch_size 64 --save_name T-SimCLR_backbone.pth

ECHO "--- RUNNING: Group 2 (T-BYOL) SSL Pretraining (Exp 4/25) ---"
python -m training.train_ssl --method byol --epochs 100 --batch_size 64 --save_name T-BYOL_backbone.pth

ECHO "--- RUNNING: Group 2 (T-DINO) SSL Pretraining (Exp 5/25) ---"
python -m training.train_ssl --method dino --epochs 100 --batch_size 64 --save_name T-DINO_backbone.pth

:: ====================================================================
:: --- GROUP 2 (Continued): FINE-TUNE SSL TEACHERS ---
:: --- We fine-tune the backbones we just trained ---
:: ====================================================================

ECHO "--- RUNNING: Group 2 (T-SimCLR) Fine-Tuning (Exp 6/25) ---"
python -m training.train_supervised --model resnet18 --load_path checkpoints/T-SimCLR_backbone.pth --epochs 50 --save_name T-SimCLR_teacher.pth

ECHO "--- RUNNING: Group 2 (T-BYOL) Fine-Tuning (Exp 7/25) ---"
python -m training.train_supervised --model resnet18 --load_path checkpoints/T-BYOL_backbone.pth --epochs 50 --save_name T-BYOL_teacher.pth

ECHO "--- RUNNING: Group 2 (T-DINO) Fine-Tuning (Exp 8/25) ---"
python -m training.train_supervised --model resnet18 --load_path checkpoints/T-DINO_backbone.pth --epochs 50 --save_name T-DINO_teacher.pth

:: ====================================================================
:: --- GROUP 3: KNOWLEDGE DISTILLATION BASELINE ---
:: ====================================================================

ECHO "--- IGNORING: Group 3 (KD-Baseline) (Exp 9/25) BECAUSE IT WAS RUN BEFORE ---"
:: ECHO "--- RUNNING: Group 3 (KD-Baseline) (Exp 9/25) ---"
:: python -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-SUP_teacher_baseline.pth --epochs 50 --save_name KD-Baseline.pth

:: ====================================================================
:: --- GROUP 4: SINGLE-TEACHER KD (from SSL Teachers) ---
:: ====================================================================

ECHO "--- RUNNING: Group 4 (KD from T-SimCLR) (Exp 10/25) ---"
python -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth --epochs 50 --save_name KD-SimCLR.pth

ECHO "--- RUNNING: Group 4 (KD from T-BYOL) (Exp 11/25) ---"
python -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth --epochs 50 --save_name KD-BYOL.pth

ECHO "--- RUNNING: Group 4 (KD from T-DINO) (Exp 12/25) ---"
python -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-DINO.pth

:: ====================================================================
:: --- GROUP 5: ENSEMBLE KD ---
:: ====================================================================

ECHO "--- RUNNING: Group 5 (Ensemble SimCLR+BYOL) (Exp 13/25) ---"
python -m training.train_kd --teachers resnet18 resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth checkpoints/T-BYOL_teacher.pth --epochs 50 --save_name KD-Ensemble-SimCLR-BYOL.pth

ECHO "--- RUNNING: Group 5 (Ensemble SimCLR+DINO) (Exp 14/25) ---"
python -m training.train_kd --teachers resnet18 resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-Ensemble-SimCLR-DINO.pth

ECHO "--- RUNNING: Group 5 (Ensemble BYOL+DINO) (Exp 15/25) ---"
python -m training.train_kd --teachers resnet18 resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-Ensemble-BYOL-DINO.pth

ECHO "--- RUNNING: Group 5 (Ensemble All 3) (Exp 16/25) ---"
python -m training.train_kd --teachers resnet18 resnet18 resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth checkpoints/T-BYOL_teacher.pth checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-Ensemble-All.pth

:: ====================================================================
:: --- GROUP 6 & 7: PROGRESSIVE KD (This is long) ---
:: ====================================================================
ECHO "--- NOTE: Skipping Progressive KD for now ---"
ECHO "--- This requires 18 more complex commands ---"
ECHO "--- Let's analyze the first 16 experiments first ---"


ECHO "================================================="
ECHO "--- PRIMARY EXPERIMENTS FINISHED ---"
ECHO "================================================="
pause