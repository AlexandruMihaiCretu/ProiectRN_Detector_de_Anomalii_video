import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "training_data_labeled.csv"
TRAIN_DIR = ROOT_DIR / "data" / "train"
VAL_DIR = ROOT_DIR / "data" / "validation"
TEST_DIR = ROOT_DIR / "data" / "test"

if not PROCESSED_DATA_PATH.exists():
    print(f"Error: {PROCESSED_DATA_PATH} not found!")
else:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = df[df['is_real'] != -1] 

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    df_real = train_df[train_df.is_real == 1]
    df_glitch = train_df[train_df.is_real == 0]

    df_glitch_upsampled = resample(df_glitch, replace=True, n_samples=len(df_real), random_state=42)
    balanced_train_df = pd.concat([df_real, df_glitch_upsampled])

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    balanced_train_df.to_csv(TRAIN_DIR / "train.csv", index=False)
    val_df.to_csv(VAL_DIR / "val.csv", index=False)
    test_df.to_csv(TEST_DIR / "test.csv", index=False)

    print(f"Done! Training set now has {len(balanced_train_df)} balanced rows.")
    print(f"Data saved to {ROOT_DIR / 'data'}")