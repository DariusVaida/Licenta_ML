# train_transformer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
from spacetimeformer.spacetimeformer_model.nn import Spacetimeformer


#The Classifier Model Definition
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


#The PyTorch Dataset Class
class DeviceTimeSeriesDataset(Dataset):
    def __init__(self, metadata_path, data_folder_path, context_len, pred_len):
        self.metadata_df = pd.read_csv(metadata_path)
        self.data_folder_path = data_folder_path
        self.context_len = context_len
        self.pred_len = pred_len
        self.total_len = context_len + pred_len
        self.class_map = {"Low": 0, "Medium": 1, "High": 2}

    def __len__(self):
        return len(self.metadata_df)

    def prepare_model_inputs(self, df):
        # This function should be inside the class or accessible to it
        enc_df = df.iloc[:self.context_len]
        enc_vals = enc_df[['used_ram_kb', 'free_ram_kb', 'cpu_load_percent', 'gpu_load_percent']].values
        enc_y = torch.tensor(enc_vals, dtype=torch.float32)
        enc_x = torch.arange(self.context_len, dtype=torch.float32).view(-1, 1)
        dec_y = torch.zeros((self.pred_len, enc_y.shape[-1]), dtype=torch.float32)
        dec_x = torch.arange(self.context_len, self.total_len, dtype=torch.float32).view(-1, 1)
        return enc_y, enc_x, dec_y, dec_x

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        filename = row['filename']
        activity_label = row['activity_level']

        json_path = os.path.join(self.data_folder_path, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data["time_series"])

        if len(df) < self.total_len:
            pad_len = self.total_len - len(df)
            df = pd.concat([df] + [df.iloc[-1:]] * pad_len, ignore_index=True)

        enc_y, enc_x, dec_y, dec_x = self.prepare_model_inputs(df)
        label = torch.tensor(self.class_map[activity_label], dtype=torch.long)

        return enc_y, enc_x, dec_y, dec_x, label


# --- 3. Main Training & Validation Script ---
if __name__ == '__main__':
    # Hyperparameters
    CONTEXT_LEN = 100
    PRED_LEN = 10  # Still needed for placeholders
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4

    # Create Datasets and DataLoaders
    # Make sure you have split 'metadata.csv' into 'train_metadata.csv' and 'val_metadata.csv'
    train_dataset = DeviceTimeSeriesDataset(metadata_path='generated_data/data/train_metadata.csv', data_folder_path='collected_data/',
                                            context_len=CONTEXT_LEN, pred_len=PRED_LEN)
    val_dataset = DeviceTimeSeriesDataset(metadata_path='generated_data/data/validation_metadata.csv', data_folder_path='collected_data/',
                                          context_len=CONTEXT_LEN, pred_len=PRED_LEN)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Configuration
    config = {
        "d_yc": 4, "d_yt": 4, "d_x": 1, "d_model": 256, "n_heads": 8, "e_layers": 3,
        "d_layers": 1, "d_ff": 512, "dropout_emb": 0.1, "pos_emb_type": "t2v", "start_token_len": 0
    }

    # Instantiate Model, Loss Function, and Optimizer
    model = DeviceClassifier(base_model_config=config, num_classes=3)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for enc_y, enc_x, dec_y, dec_x, labels in train_dataloader:
            optimizer.zero_grad()
            logits = model(enc_y, enc_x, dec_y, dec_x)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for enc_y, enc_x, dec_y, dec_x, labels in val_dataloader:
                logits = model(enc_y, enc_x, dec_y, dec_x)
                loss = loss_function(logits, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = (correct_predictions / total_samples) * 100

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    print("\n--- Training Complete ---")

    # Save the trained model's state
    torch.save(model.state_dict(), 'device_classifier_model.pth')
    print("Model saved to device_classifier_model.pth")