@ECHO OFF
ECHO "================================================="
ECHO "STARTING ALL 25 PROJECT EXPERIMENTS"
ECHO "================================================="
ECHO "Activating conda environment..."
CALL conda activate CHE-AI-Project

:: ====================================================================
:: --- GROUP 1 & 2: BASELINES ---
:: ====================================================================

ECHO "--- RUNNING: Group 1 (B1) Student Baseline (Exp 1/25) ---"
python -u -m training.train_supervised --model mobilenetv3 --epochs 50 --save_name B1_student_baseline.pth

ECHO "--- RUNNING: Group 2 (T-SUP) Teacher Baseline (Exp 2/25) ---"
python -u -m training.train_supervised --model resnet18 --epochs 50 --save_name T-SUP_teacher_baseline.pth

:: ====================================================================
:: --- GROUP 2 (Continued): SSL PRETRAINING (THE SLOW PART) ---
:: ====================================================================

ECHO "--- RUNNING: Group 2 (T-SimCLR) SSL Pretraining (Exp 3/25) ---"
:: python -u -m training.train_ssl --method simclr --epochs 50 --batch_size 64 --save_name T-SimCLR_backbone.pth

ECHO "--- RUNNING: Group 2 (T-BYOL) SSL Pretraining (Exp 4/25) ---"
:: python -u -m training.train_ssl --method byol --epochs 50 --batch_size 64 --save_name T-BYOL_backbone.pth

ECHO "--- RUNNING: Group 2 (T-DINO) SSL Pretraining (Exp 5/25) ---"
:: python -u -m training.train_ssl --method dino --epochs 50 --batch_size 64 --save_name T-DINO_backbone.pth

:: ====================================================================
:: --- GROUP 2 (Continued): FINE-TUNE SSL TEACHERS ---
:: ====================================================================

ECHO "--- RUNNING: Group 2 (T-SimCLR) Fine-Tuning (Exp 6/25) ---"
python -u -m training.train_supervised --model resnet18 --load_path checkpoints/T-SimCLR_backbone.pth --epochs 50 --save_name T-SimCLR_teacher.pth

ECHO "--- RUNNING: Group 2 (T-BYOL) Fine-Tuning (Exp 7/25) ---"
python -u -m training.train_supervised --model resnet18 --load_path checkpoints/T-BYOL_backbone.pth --epochs 50 --save_name T-BYOL_teacher.pth

ECHO "--- RUNNING: Group 2 (T-DINO) Fine-Tuning (Exp 8/25) ---"
python -u -m training.train_supervised --model resnet18 --load_path checkpoints/T-DINO_backbone.pth --epochs 50 --save_name T-DINO_teacher.pth

:: ====================================================================
:: --- GROUP 3: KNOWLEDGE DISTILLATION BASELINE ---
:: ====================================================================

ECHO "--- RUNNING: Group 3 (KD-Baseline) (Exp 9/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-SUP_teacher_baseline.pth --epochs 50 --save_name KD-Baseline.pth

:: ====================================================================
:: --- GROUP 4: SINGLE-TEACHER KD ---
:: ====================================================================

ECHO "--- RUNNING: Group 4 (KD from T-SimCLR) (Exp 10/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth --epochs 50 --save_name KD-SimCLR.pth

ECHO "--- RUNNING: Group 4 (KD from T-BYOL) (Exp 11/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth --epochs 50 --save_name KD-BYOL.pth

ECHO "--- RUNNING: Group 4 (KD from T-DINO) (Exp 12/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-DINO.pth

:: ====================================================================
:: --- GROUP 5: ENSEMBLE KD ---
:: ====================================================================

ECHO "--- RUNNING: Group 5 (Ensemble SimCLR+BYOL) (Exp 13/25) ---"
python -u -m training.train_kd --teachers resnet18 resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth checkpoints/T-BYOL_teacher.pth --epochs 50 --save_name KD-Ensemble-SimCLR-BYOL.pth

ECHO "--- RUNNING: Group 5 (Ensemble SimCLR+DINO) (Exp 14/25) ---"
python -u -m training.train_kd --teachers resnet18 resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-Ensemble-SimCLR-DINO.pth

ECHO "--- RUNNING: Group 5 (Ensemble BYOL+DINO) (Exp 15/25) ---"
python -u -m training.train_kd --teachers resnet18 resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-Ensemble-BYOL-DINO.pth

ECHO "--- RUNNING: Group 5 (Ensemble All 3) (Exp 16/25) ---"
python -u -m training.train_kd --teachers resnet18 resnet18 resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth checkpoints/T-BYOL_teacher.pth checkpoints/T-DINO_teacher.pth --epochs 50 --save_name KD-Ensemble-All.pth

:: ====================================================================
:: --- GROUP 6: PROGRESSIVE KD (Two Stages) ---
:: ====================================================================

ECHO "--- RUNNING: Progressive SimCLR -> BYOL (Exp 17/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth --student_load_path checkpoints/KD-SimCLR.pth --epochs 30 --save_name KD-Prog-SimCLR-BYOL.pth

ECHO "--- RUNNING: Progressive SimCLR -> DINO (Exp 18/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-DINO_teacher.pth --student_load_path checkpoints/KD-SimCLR.pth --epochs 30 --save_name KD-Prog-SimCLR-DINO.pth

ECHO "--- RUNNING: Progressive BYOL -> SimCLR (Exp 19/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth --student_load_path checkpoints/KD-BYOL.pth --epochs 30 --save_name KD-Prog-BYOL-SimCLR.pth

ECHO "--- RUNNING: Progressive BYOL -> DINO (Exp 20/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-DINO_teacher.pth --student_load_path checkpoints/KD-BYOL.pth --epochs 30 --save_name KD-Prog-BYOL-DINO.pth

ECHO "--- RUNNING: Progressive DINO -> SimCLR (Exp 21/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-SimCLR_teacher.pth --student_load_path checkpoints/KD-DINO.pth --epochs 30 --save_name KD-Prog-DINO-SimCLR.pth

ECHO "--- RUNNING: Progressive DINO -> BYOL (Exp 22/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth --student_load_path checkpoints/KD-DINO.pth --epochs 30 --save_name KD-Prog-DINO-BYOL.pth

:: ====================================================================
:: --- GROUP 7: PROGRESSIVE KD (Three Stages) ---
:: ====================================================================

ECHO "--- RUNNING: Progressive SimCLR -> BYOL -> DINO (Exp 23/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-DINO_teacher.pth --student_load_path checkpoints/KD-Prog-SimCLR-BYOL.pth --epochs 30 --save_name KD-Prog-SimCLR-BYOL-DINO.pth

ECHO "--- RUNNING: Progressive SimCLR -> DINO -> BYOL (Exp 24/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-BYOL_teacher.pth --student_load_path checkpoints/KD-Prog-SimCLR-DINO.pth --epochs 30 --save_name KD-Prog-SimCLR-DINO-BYOL.pth

ECHO "--- RUNNING: Progressive BYOL -> SimCLR -> DINO (Exp 25/25) ---"
python -u -m training.train_kd --teachers resnet18 --teacher_paths checkpoints/T-DINO_teacher.pth --student_load_path checkpoints/KD-Prog-BYOL-SimCLR.pth --epochs 30 --save_name KD-Prog-BYOL-SimCLR-DINO.pth

:: ====================================================================
:: --- METRICS & PLOTTING ---
:: ====================================================================

ECHO "--- Measuring Model Efficiency (FLOPs/Throughput) ---"
:: python -u measure_models.py

ECHO "--- Generating Results Plots ---"
:: python -u plot_results.py

ECHO "================================================="
ECHO "--- ALL 25 EXPERIMENTS COMPLETED SUCCESSFULLY ---"
ECHO "================================================="
pause