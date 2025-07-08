# predict_activity.py

import torch
import torch.nn as nn
import pandas as pd
import json
import os
from spacetimeformer.spacetimeformer_model.nn import Spacetimeformer


#Re-define the Model and Data Preparation

class DeviceClassifier(nn.Module):
    def __init__(self, base_model_config, num_classes=3):
        super().__init__()
        self.spacetimeformer = Spacetimeformer(**base_model_config)
        self.classification_head = nn.Sequential(
            nn.Linear(base_model_config["d_yc"], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, enc_y, enc_x, dec_y, dec_x):
        _, recon_output, _, _ = self.spacetimeformer(enc_y=enc_y, enc_x=enc_x, dec_y=dec_y, dec_x=dec_x)
        pooled_output = torch.mean(recon_output, dim=1)
        logits = self.classification_head(pooled_output)
        return logits


def prepare_model_inputs(time_series_df, context_len, pred_len):
    """Processes a DataFrame into tensors for the model."""
    df = time_series_df.copy()


    if len(df) < context_len:
        pad_len = context_len - len(df)
        df = pd.concat([df] + [df.iloc[-1:]] * pad_len, ignore_index=True)

    enc_df = df.iloc[:context_len]
    enc_vals = enc_df[['used_ram_kb', 'free_ram_kb', 'cpu_load_percent', 'gpu_load_percent']].values
    enc_y = torch.tensor(enc_vals, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    enc_x = torch.arange(context_len, dtype=torch.float32).view(1, -1, 1)  # Add batch dimension


    dec_y = torch.zeros((1, pred_len, enc_y.shape[-1]), dtype=torch.float32)
    dec_x = torch.arange(context_len, context_len + pred_len, dtype=torch.float32).view(1, -1, 1)

    return enc_y, enc_x, dec_y, dec_x


#Load Model and Make Prediction
if __name__ == '__main__':
    # --- SETUP ---
    CONTEXT_LEN = 100
    PRED_LEN = 10
    MODEL_PATH = 'device_classifier_model.pth'
    JSON_FILE_TO_PREDICT = 'unlabeled_data/new_sample.json'


    config = {
        "d_yc": 4, "d_yt": 4, "d_x": 1, "d_model": 256, "n_heads": 8, "e_layers": 3,
        "d_layers": 1, "d_ff": 512, "dropout_emb": 0.1, "pos_emb_type": "t2v", "start_token_len": 0
    }


    print(f"Loading trained model from {MODEL_PATH}...")
    inference_model = DeviceClassifier(base_model_config=config, num_classes=3)
    inference_model.load_state_dict(torch.load(MODEL_PATH))
    inference_model.eval()
    print("Model loaded successfully.")


    print(f"\nLoading data from {JSON_FILE_TO_PREDICT}...")
    try:
        with open(JSON_FILE_TO_PREDICT, 'r') as f:
            data = json.load(f)

        ts_df = pd.DataFrame(data.get("time_series", []))
        if ts_df.empty:
            raise ValueError("JSON file does not contain time_series data.")

        # Prepare the data into tensors
        enc_y, enc_x, dec_y, dec_x = prepare_model_inputs(ts_df, CONTEXT_LEN, PRED_LEN)
        print("Data prepared for inference.")


        with torch.no_grad():  # Disable gradient calculation for efficiency
            logits = inference_model(enc_y, enc_x, dec_y, dec_x)

            # Get the predicted class index by finding the max logit
            _, predicted_class_index = torch.max(logits.data, 1)


        class_names = ["Low", "Medium", "High"]
        predicted_class_name = class_names[predicted_class_index.item()]

        print("\n" + "=" * 30)
        print(f"Predicted Activity Level: {predicted_class_name}")
        print("=" * 30)

    except FileNotFoundError:
        print(f"\nError: The file '{JSON_FILE_TO_PREDICT}' was not found.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")