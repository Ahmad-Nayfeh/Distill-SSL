import pandas as pd
import os
import glob

# --- Configuration ---
LOGS_DIR = 'logs'
OUTPUT_FILE = 'final_results_summary.csv'

def generate_master_csv():
    print(f"--- Scanning {LOGS_DIR} for experiment logs ---")
    
    all_rows = []
    csv_files = glob.glob(os.path.join(LOGS_DIR, '*.csv'))
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        if filename == OUTPUT_FILE: continue
            
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"[Skipped] {filename} - Error: {e}")
            continue

        # --- FILTER: Skip SSL Logs (they don't have test_acc) ---
        if 'test_acc' not in df.columns:
            continue

        # --- ROW SELECTION ---
        # Priority: Epoch 50 -> Epoch 30 -> Last Available
        if 50 in df['epoch'].values:
            row = df[df['epoch'] == 50].iloc[0]
        elif 30 in df['epoch'].values:
            row = df[df['epoch'] == 30].iloc[0]
        else:
            row = df.iloc[-1] # Fallback to last row

        # --- DATA EXTRACTION & CLEANING ---
        exp_name = filename.replace('.csv', '').replace('KD-', '').replace('_baseline', '')
        
        # Calculate percentages
        train_top1 = row['train_acc'] * 100
        test_top1 = row['test_acc'] * 100
        test_top5 = row['test_top5_acc'] * 100
        
        # Create clean row
        final_row = {
            'Experiment Name': exp_name,
            'Test Top-1 (%)': f"{test_top1:.2f}",
            'Test Top-5 (%)': f"{test_top5:.2f}"
        }
        
        all_rows.append(final_row)

    # --- SAVE ---
    if all_rows:
        master_df = pd.DataFrame(all_rows)
        
        # Sort alphabetically or by custom logic? Alphabetical is safest.
        master_df = master_df.sort_values('Experiment Name')
        
        # Define Final Column Order
        cols = ['Experiment Name', 'Test Top-1 (%)', 'Test Top-5 (%)']
        master_df = master_df[cols]
        
        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCESS! Clean summary saved to: {OUTPUT_FILE}")
        print("-" * 60)
        print(master_df.to_string(index=False))
        print("-" * 60)
    else:
        print("\nNo valid logs found.")

if __name__ == '__main__':
    generate_master_csv()