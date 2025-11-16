import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_supervised_metrics(log_path, save_name):
    """
    Loads a supervised training log and plots
    - Accuracy (Train vs Test Top-1 vs Test Top-5)
    - Loss (Train vs Test)
    """
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Log file not found: {log_path}. Skipping.")
        return

    # --- Set up the plot style ---
    sns.set_theme(style="whitegrid")
    
    # --- Create the figure with two subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # --- 1. Accuracy Plot ---
    sns.lineplot(x='epoch', y='train_acc', data=df, label='Train Accuracy', ax=ax1, color='blue')
    sns.lineplot(x='epoch', y='test_acc', data=df, label='Test Top-1 Acc.', ax=ax1, color='orange')
    
    # --- New Plot Line ---
    if 'test_top5_acc' in df.columns:
        sns.lineplot(x='epoch', y='test_top5_acc', data=df, label='Test Top-5 Acc.', ax=ax1, color='green', linestyle='--')
    # --- End New ---

    ax1.set_title(f'Model Accuracy ({save_name})', fontsize=16)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # --- 2. Loss Plot ---
    sns.lineplot(x='epoch', y='train_loss', data=df, label='Train Loss', ax=ax2, color='blue')
    
    if 'test_loss' in df.columns: # SSL logs don't have test_loss
        sns.lineplot(x='epoch', y='test_loss', data=df, label='Test Loss', ax=ax2, color='orange')
        
    ax2.set_title(f'Model Loss ({save_name})', fontsize=16)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # --- Final Touches ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Supervised Training Metrics: {save_name}', fontsize=20)
    
    # --- Save the Figure ---
    save_path = os.path.join('plots', f"{save_name}.png")
    fig.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    
    print("--- Generating All Training Plots ---")
    
    # --- Find all log files in the 'logs' directory ---
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
    else:
        for filename in os.listdir(log_dir):
            if filename.endswith('.csv'):
                log_path = os.path.join(log_dir, filename)
                save_name = filename.replace('.csv', '')
                
                print(f"Plotting {log_path}...")
                # We can use the same function, it will just skip plots
                # for data that isn't there (e.g., SSL logs)
                plot_supervised_metrics(
                    log_path=log_path,
                    save_name=save_name
                )
    
    print("--- Plot generation complete! ---")