import pandas as pd
from pathlib import Path

def process_and_label_data():
    root_dir = Path(__file__).resolve().parent.parent
    raw_path = root_dir / "data" / "raw" / "behavioral_logs.csv"
    processed_dir = root_dir / "data" / "processed"
    output_path = processed_dir / "training_data_labeled.csv"

    if not raw_path.exists():
        print(f"Error: {raw_path} not found!")
        return
        
    df = pd.read_csv(raw_path)
    df['is_real'] = -1

    df.loc[df['age_frames'] > 25, 'is_real'] = 1
    df.loc[(df['age_frames'] < 5) & (df['velocity_px'] > 20), 'is_real'] = 0
    df.loc[df['confidence'] < 0.2, 'is_real'] = 0

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Success! Labeled data saved to: {output_path}")

if __name__ == "__main__":
    process_and_label_data()