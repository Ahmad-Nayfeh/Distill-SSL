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

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'text.usetex': False,  # Set to True if you have LaTeX installed
})

sns.set_palette("deep")

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

# Professional color palette (colorblind-friendly)
PROFESSIONAL_PALETTE = {
    'Baselines': '#696969',          # Dim gray
    'Single-Teacher': '#0173B2',     # Blue
    'Ensemble': '#DE8F05',           # Orange
    'Progressive (2-Stage)': '#029E73',  # Green
    'Progressive (3-Stage)': '#CC78BC'   # Purple
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
# 1. Master Leaderboard
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
    
    fig, ax = plt.subplots(figsize=(7, 8))
    
    bars = ax.barh(df['Model'], df['Accuracy'], 
                   color=[PROFESSIONAL_PALETTE[g] for g in df['Group']],
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xlabel('Top-1 Test Accuracy (%)', fontweight='bold')
    ax.set_title('Performance Ranking of Student Models', fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center', fontsize=8, fontweight='normal')
    
    # Professional legend
    handles = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='black', 
                            linewidth=0.5, alpha=0.85) 
              for color in PROFESSIONAL_PALETTE.values()]
    ax.legend(handles, PROFESSIONAL_PALETTE.keys(), 
             title="Distillation Strategy", title_fontsize=9,
             loc='upper left', frameon=True, fancybox=False, 
             edgecolor='black', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Master_Leaderboard.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'Master_Leaderboard.pdf'), bbox_inches='tight')
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
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create custom order for groups
    group_order = ['Baselines', 'Single-Teacher', 'Ensemble', 
                   'Progressive (2-Stage)', 'Progressive (3-Stage)']
    
    # Boxplot with custom styling
    bp = ax.boxplot([df[df['Group']==g]['Accuracy'].values for g in group_order],
                     labels=group_order,
                     patch_artist=True,
                     widths=0.6,
                     showfliers=False,
                     boxprops=dict(linewidth=1.2, edgecolor='black'),
                     whiskerprops=dict(linewidth=1.2, color='black'),
                     capprops=dict(linewidth=1.2, color='black'),
                     medianprops=dict(linewidth=2, color='darkred'))
    
    # Color boxes
    for patch, group in zip(bp['boxes'], group_order):
        patch.set_facecolor(PROFESSIONAL_PALETTE[group])
        patch.set_alpha(0.7)
    
    # Overlay strip plot
    for i, group in enumerate(group_order, 1):
        group_data = df[df['Group']==group]['Accuracy'].values
        x = np.random.normal(i, 0.04, size=len(group_data))
        ax.scatter(x, group_data, alpha=0.6, s=40, color='black', zorder=3)
    
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('Performance Distribution by Distillation Strategy', 
                fontweight='bold', pad=15)
    ax.set_xticklabels(group_order, rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Strategy_Comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'Strategy_Comparison.pdf'), bbox_inches='tight')
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
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = ['#6A4C93', '#8B4513', '#C44569', '#556270']
    bars = ax.bar(df['Teacher'], df['Accuracy'], 
                  color=colors, edgecolor='black', 
                  linewidth=0.8, alpha=0.85, width=0.6)
    
    ax.set_title('Teacher Model Performance (ResNet-18)', fontweight='bold', pad=15)
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.5, 
                f'{height:.1f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Teacher_Quality.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'Teacher_Quality.pdf'), bbox_inches='tight')
    plt.close()

# ========================================================
# 4. Training Dynamics
# ========================================================
def plot_training_dynamics():
    print("Generating Training Dynamics Plot...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    representatives = {
        'Baseline': ('KD-Baseline', '#696969', '-'),
        'Single-Teacher': ('KD-SimCLR', '#0173B2', '-'),
        'Ensemble': ('KD-Ensemble-All', '#DE8F05', '-'),
        'Progressive': ('KD-Prog-SimCLR-BYOL-DINO', '#CC78BC', '-')
    }
    
    for label, (fname, color, style) in representatives.items():
        df = load_full_log(fname)
        if df is not None:
            ax.plot(df['epoch'], df['test_acc'], 
                   label=label, linewidth=2, 
                   color=color, linestyle=style, alpha=0.9)
    
    ax.set_title('Training Convergence Curves', fontweight='bold', pad=15)
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=True, fancybox=False, 
             edgecolor='black', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Training_Dynamics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'Training_Dynamics.pdf'), bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Custom markers and colors for each method
    markers = ['o', 's', '^', 'D']
    colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    
    for i, (method, marker, color) in enumerate(zip(df['Method'], markers, colors)):
        row = df.iloc[i]
        ax.scatter(row['Teacher Accuracy'], row['Student Accuracy'], 
                  s=200, marker=marker, color=color, 
                  edgecolors='black', linewidth=1.2,
                  label=method, alpha=0.85, zorder=3)
    
    # Trend line
    if len(df) > 1:
        z = np.polyfit(df['Teacher Accuracy'], df['Student Accuracy'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['Teacher Accuracy'].min(), df['Teacher Accuracy'].max(), 100)
        ax.plot(x_trend, p(x_trend), "--", color='red', 
               alpha=0.5, linewidth=1.5, label='Linear Fit', zorder=1)
    

    ax.set_title('Teacher-Student Performance Correlation', fontweight='bold', pad=15)
    ax.set_xlabel('Teacher Test Accuracy (%)', fontweight='bold')
    ax.set_ylabel('Student Test Accuracy (%)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=True, fancybox=False,
             edgecolor='black', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'Teacher_Student_Correlation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'Teacher_Student_Correlation.pdf'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY PLOTS (CVPR STANDARD)")
    print("=" * 60)
    try: 
        plot_master_leaderboard()
        print("✓ Master Leaderboard generated")
    except Exception as e: 
        print(f"✗ Error in Master Leaderboard: {e}")
    
    try: 
        plot_strategy_comparison()
        print("✓ Strategy Comparison generated")
    except Exception as e: 
        print(f"✗ Error in Strategy Comparison: {e}")
    
    try: 
        plot_teacher_quality()
        print("✓ Teacher Quality generated")
    except Exception as e: 
        print(f"✗ Error in Teacher Quality: {e}")
    
    try: 
        plot_training_dynamics()
        print("✓ Training Dynamics generated")
    except Exception as e: 
        print(f"✗ Error in Training Dynamics: {e}")
    
    try: 
        plot_teacher_student_correlation()
        print("✓ Teacher-Student Correlation generated")
    except Exception as e: 
        print(f"✗ Error in Teacher-Student Correlation: {e}")
    
    print("=" * 60)
    print("All plots saved in PNG and PDF formats in /plots folder")
    print("=" * 60)