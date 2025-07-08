import pandas as pd
from sklearn.model_selection import train_test_split
import os


output_folder = 'data'

os.makedirs(output_folder, exist_ok=True)

try:
    df = pd.read_csv('generated_data/metadata.csv')


    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['activity_level']
    )


    train_path = os.path.join(output_folder, 'train_metadata.csv')
    val_path = os.path.join(output_folder, 'validation_metadata.csv')


    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Successfully created '{train_path}' and '{val_path}'")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

except FileNotFoundError:
    print("Error: 'metadata.csv' not found. Please run the data_label.py script first.")