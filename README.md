## Workflow and Usage

Follow these steps in order to process your data and train the model.

### Step 1: Data Collection

Use the Go-based data collector tool to profile one or more Android devices. This will generate a set of JSON files.

-   Place all the generated raw JSON files into the `collected_data/` directory.

### Step 2: Label Generation

The `data_labeler.py` script is the first step in the ML pipeline. It reads the raw JSON files and performs two main tasks:
1.  It uses a pre-trained Random Forest model (which it builds from external benchmark data) to assign a `hardware_tier` (`Low-Tier`, `Mid-Tier`, `High-Tier`).
2.  It uses a heuristic based on CPU load and core variance to assign an `activity_level` (`Low`, `Medium`, `High`).

This script creates the `labeled_data` directory, saves the newly enriched JSON files there, and generates a `metadata.csv` summary file.

-   **Command:**
    ```bash
    python scripts/data_labeler.py
    ```

### Step 3: Create Training/Validation Sets

The `prepare_data.py` script takes the `metadata.csv` file and splits it into training and validation sets for the model. This ensures that the model is evaluated on data it has not seen during training.

-   **Command:**
    ```bash
    python scripts/prepare_data.py
    ```
-   **Output:** Creates `train_metadata.csv` and `validation_metadata.csv` inside the `data/` directory.

### Step 4: Train the Spacetimeformer Model

The `train_transformer.py` script uses the prepared metadata and the labeled JSON files to train the Spacetimeformer model. The script will print the training progress for each epoch.

-   **Command:**
    ```bash
    python scripts/train_transformer.py
    ```
-   **Output:** Saves the trained model weights to a file named `device_classifier_model.pth`.

### Step 5: Inference on New Data

Once the model is trained, you can use the `predict_activity.py` script to classify a new, unlabeled device profile.

1.  Place your new JSON file in a directory (e.g., `unlabeled_data/`).
2.  Update the `JSON_FILE_TO_PREDICT` variable inside the script to point to your new file.
3.  Run the script.

-   **Command:**
    ```bash
    python scripts/predict_activity.py
    ```
-   **Output:** The script will print the predicted activity level to the console.

## Scripts Description

-   `data_labeler.py`: The main data processing script. It loads raw JSONs, applies the dual-labeling logic, and saves the enriched data and a metadata summary.
-   `prepare_data.py`: A utility script to split the metadata into training and validation sets, ensuring the model can be evaluated fairly.
-   `train_transformer.py`: The main training script. It defines the PyTorch `Dataset` and `DataLoader`, instantiates the Spacetimeformer model, and runs the training and validation loops.
-   `predict_activity.py`: An example script showing how to load the trained model and use it for inference on a single, new data file.

## Model Details

This project uses a dual-labeling approach to enrich the dataset:
-   **Hardware Tier**: An objective classification of the device's hardware potential, determined by a Random Forest model trained on external benchmark data.
-   **Activity Level**: A classification of the device's workload intensity during the profiling window, determined by a heuristic.

The primary machine learning model is a **Spacetimeformer**, a modern Transformer architecture designed for multivariate time-series data. It is trained to predict the **Activity Level** based on the patterns it observes in the time-series sensor data.
