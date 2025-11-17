import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import glob

# --- Configuration ---
LOGS_DIR = 'logs'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid") 

# Define Experiment Groups for Analysis
GROUPS = {
    'Baselines': ['B1_student_baseline', 'KD-Baseline'],
    'Single-Teacher': ['KD-SimCLR', 'KD-BYOL', 'KD-DINO'],
    'Ensemble': [
        'KD-Ensemble-SimCLR-BYOL', 'KD-Ensemble-SimCLR-DINO', 
        'KD-Ensemble-BYOL-DINO', 'KD-Ensemble-All'
    ],
    'Progressive (2-Stage)': [
        'KD-Prog-SimCLR-BYOL', 'KD-Prog-SimCLR-DINO', 
        'KD-Prog-BYOL-SimCLR', 'KD-Prog-BYOL-DINO', 
        'KD-Prog-DINO-SimCLR', 'KD-Prog-DINO-BYOL'
    ],
    'Progressive (3-Stage)': [
        'KD-Prog-SimCLR-BYOL-DINO', 'KD-Prog-SimCLR-DINO-BYOL', 
        'KD-Prog-BYOL-SimCLR-DINO', 'KD-Prog-BYOL-DINO-SimCLR', 
        'KD-Prog-DINO-SimCLR-BYOL', 'KD-Prog-DINO-BYOL-SimCLR'
    ],
    'Teachers': ['T-SUP_teacher_baseline', 'T-SimCLR_teacher', 'T-BYOL_teacher', 'T-DINO_teacher']
}

# --- Helper Functions ---
def load_best_acc(filename):
    path = os.path.join(LOGS_DIR, f"{filename}.csv")
    if not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        return df['test_acc'].max() * 100
    except: return None

def load_full_log(filename):
    path = os.path.join(LOGS_DIR, f"{filename}.csv")
    if not os.path.exists(path): return None
    return pd.read_csv(path)

# ========================================================
# 1. Master Leaderboard (Legend Fixed)
# ========================================================
def plot_master_leaderboard():
    print("Generating Master Leaderboard...")
    data = []
    for group_name, filenames in GROUPS.items():
        if group_name == 'Teachers': continue 
        for fname in filenames:
            acc = load_best_acc(fname)
            if acc is not None:
                clean_name = fname.replace('KD-', '').replace('_baseline', '').replace('Prog-', '')
                data.append({'Model': clean_name, 'Accuracy': acc, 'Group': group_name})
    
    if not data: return
    df = pd.DataFrame(data).sort_values('Accuracy', ascending=True)
    
    plt.figure(figsize=(12, 10)) # Standard size
    
    palette = {
        'Baselines': '#7f7f7f',
        'Single-Teacher': '#1f77b4',
        'Ensemble': '#ff7f0e',
        'Progressive (2-Stage)': '#2ca02c',
        'Progressive (3-Stage)': '#d62728'
    }
    
    bars = plt.barh(df['Model'], df['Accuracy'], color=[palette[g] for g in df['Group']])
    
    plt.title('Master Leaderboard: All Student Models Ranked', fontsize=16)
    plt.xlabel('Top-1 Test Accuracy (%)', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # --- FIXED LEGEND ---
    # Placed inside the plot at the bottom right
    handles = [plt.Rectangle((0,0),1,1, color=color) for label, color in palette.items()]
    plt.legend(handles, palette.keys(), title="Strategy", fontsize=10, loc='upper left') # you can adjust loc as needed for example ('lower right' or 'upper left')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}%', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Master_Leaderboard.png'), dpi=300)
    plt.close()

# ========================================================
# 2. Strategy Comparison
# ========================================================
def plot_strategy_comparison():
    print("Generating Strategy Comparison...")
    data = []
    for group_name, filenames in GROUPS.items():
        if group_name == 'Teachers': continue
        for fname in filenames:
            acc = load_best_acc(fname)
            if acc is not None:
                data.append({'Group': group_name, 'Accuracy': acc})
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group', y='Accuracy', data=df, palette="Set2", showfliers=False)
    sns.stripplot(x='Group', y='Accuracy', data=df, color='black', alpha=0.6, jitter=True)
    plt.title('Performance Distribution by Distillation Strategy', fontsize=16)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Strategy_Comparison.png'), dpi=300)
    plt.close()

# ========================================================
# 3. Teacher Quality
# ========================================================
def plot_teacher_quality():
    print("Generating Teacher Quality Plot...")
    data = []
    for fname in GROUPS['Teachers']:
        acc = load_best_acc(fname)
        if acc is not None:
            clean_name = fname.split('_')[0].replace('T-', '')
            data.append({'Teacher': clean_name, 'Accuracy': acc})
    if not data: return
    df = pd.DataFrame(data).sort_values('Accuracy', ascending=False)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df['Teacher'], df['Accuracy'], color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
    plt.title('Teacher Quality (Fine-Tuned ResNet-18)', fontsize=16)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, 
                 f'{height:.2f}%', ha='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Teacher_Quality.png'), dpi=300)
    plt.close()

# ========================================================
# 4. Training Dynamics
# ========================================================
def plot_training_dynamics():
    print("Generating Training Dynamics Plot...")
    plt.figure(figsize=(12, 7))
    representatives = {
        'Baseline': 'KD-Baseline',
        'Best Single (SimCLR)': 'KD-SimCLR',
        'Best Ensemble': 'KD-Ensemble-All',
        'Best Progressive': 'KD-Prog-SimCLR-BYOL-DINO' 
    }
    for label, fname in representatives.items():
        df = load_full_log(fname)
        if df is not None:
            sns.lineplot(x='epoch', y='test_acc', data=df, label=label, linewidth=2.5)
    plt.title('Training Dynamics: Test Accuracy vs Epochs', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Legend placed bottom right to avoid covering the lines
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Training_Dynamics.png'), dpi=300)
    plt.close()

# ========================================================
# 5. Teacher-Student Correlation
# ========================================================
def plot_teacher_student_correlation():
    print("Generating Teacher-Student Correlation Plot...")
    pairs = [
        ('T-SUP_teacher_baseline', 'KD-Baseline', 'Supervised'),
        ('T-SimCLR_teacher', 'KD-SimCLR', 'SimCLR'),
        ('T-BYOL_teacher', 'KD-BYOL', 'BYOL'),
        ('T-DINO_teacher', 'KD-DINO', 'DINO')
    ]
    data = []
    for teacher_file, student_file, label in pairs:
        t_acc = load_best_acc(teacher_file)
        s_acc = load_best_acc(student_file)
        if t_acc is not None and s_acc is not None:
            data.append({'Method': label, 'Teacher Accuracy': t_acc, 'Student Accuracy': s_acc})
    if not data: return
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Teacher Accuracy', y='Student Accuracy', 
                    hue='Method', style='Method', s=300, palette='deep')
    if len(df) > 1:
        z = np.polyfit(df['Teacher Accuracy'], df['Student Accuracy'], 1)
        p = np.poly1d(z)
        plt.plot(df['Teacher Accuracy'], p(df['Teacher Accuracy']), "r--", alpha=0.4, label='Trend')

    plt.title('Correlation: Teacher Quality vs. Student Performance', fontsize=16)
    plt.xlabel('Teacher Test Accuracy (%)', fontsize=14)
    plt.ylabel('Student Test Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    for i, row in df.iterrows():
        plt.text(row['Teacher Accuracy']+0.05, row['Student Accuracy']+0.05, 
                 f"{row['Method']}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Teacher_Student_Correlation.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    print("--- Starting Comprehensive Plotting ---")
    try: plot_master_leaderboard()
    except Exception as e: print(f"Error in Master: {e}")
    try: plot_strategy_comparison()
    except Exception as e: print(f"Error in Strategy: {e}")
    try: plot_teacher_quality()
    except Exception as e: print(f"Error in Quality: {e}")
    try: plot_training_dynamics()
    except Exception as e: print(f"Error in Dynamics: {e}")
    try: plot_teacher_student_correlation()
    except Exception as e: print(f"Error in Correlation: {e}")
    print("\n--- All Plots Generated in /plots folder! ---")