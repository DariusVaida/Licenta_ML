import numpy as np
import pandas as pd
import json
import os
from rapidfuzz import process, fuzz
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


#STEP 1: LOAD PREREQUISITE DATA
print("Loading specification and benchmark datasets...")
try:
    spec_data = pd.read_csv('../specs_data/mobile.csv')
    gpu_benchmark_data = pd.read_csv('../specs_data/GPU_benchmarks.csv')
    cpu_benchmark_data = pd.read_csv('../specs_data/cpu_data.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure your specs_data/ folder is set up correctly.")
    exit()

#PREPARE DATA AND TRAIN THE STATIC HARDWARE CLASSIFIER
print("Preparing data and training the hardware classifier...")

print("Cleaning and renaming columns from mobile.csv...")
spec_data.rename(columns={
    'GPU Clock:': 'GPU Clock',
    'RAM Capacity (converted)': 'RAM Capacity'
}, inplace=True)
spec_data['GPU Clock'] = pd.to_numeric(spec_data['GPU Clock'].str.replace(' MHz', '', regex=False), errors='coerce')
spec_data['RAM Capacity'] = pd.to_numeric(spec_data['RAM Capacity'].str.replace(' GiB RAM', '', regex=False),
                                          errors='coerce')

#Prepare CPU benchmark data + calculate scores
cpu_benchmark_data = cpu_benchmark_data[['cpu_Name', 'single_core_score', 'multi_core_score']]
cpu_benchmark_data.dropna(subset=['single_core_score', 'multi_core_score'], inplace=True)
cpu_benchmark_data = cpu_benchmark_data.drop_duplicates(subset=['cpu_Name'])
max_single = cpu_benchmark_data['single_core_score'].max()
max_multi = cpu_benchmark_data['multi_core_score'].max()
cpu_benchmark_data['CPU_Composite_Score'] = ((cpu_benchmark_data['single_core_score'] / max_single) * 60 + (
            cpu_benchmark_data['multi_core_score'] / max_multi) * 40)

# C. Prepare GPU benchmark data + calculate scores
gpu_benchmark_data = gpu_benchmark_data[['gpuName', 'G3Dmark', 'G2Dmark']]
gpu_benchmark_data.dropna(subset=['G3Dmark', 'G2Dmark'], inplace=True)
max_g3d = gpu_benchmark_data['G3Dmark'].max()
max_g2d = gpu_benchmark_data['G2Dmark'].max()
gpu_benchmark_data['GPU_Composite_Score'] = (
            (gpu_benchmark_data['G3Dmark'] / max_g3d) * 90 + (gpu_benchmark_data['G2Dmark'] / max_g2d) * 10)

# D. Create the mapping dictionaries for fuzzy matching
cpu_score_map = dict(
    zip(cpu_benchmark_data['cpu_Name'].str.upper().str.strip(), cpu_benchmark_data['CPU_Composite_Score']))
gpu_score_map = dict(
    zip(gpu_benchmark_data['gpuName'].str.upper().str.strip(), gpu_benchmark_data['GPU_Composite_Score']))


def get_composite_score(name, choices_map, scorer=fuzz.token_set_ratio, cutoff=80):
    if pd.isna(name): return None
    result = process.extractOne(name, choices_map.keys(), scorer=scorer, score_cutoff=cutoff)
    return choices_map[result[0]] if result else None


spec_data['CPU_Composite_Score'] = spec_data['CPU'].str.upper().str.strip().apply(
    lambda x: get_composite_score(x, cpu_score_map))
spec_data['GPU_Composite_Score'] = spec_data['Graphical Controller'].str.upper().str.strip().apply(
    lambda x: get_composite_score(x, gpu_score_map))

# E. Define feature lists
categorical_features = ['CPU', 'RAM Type', 'Graphical Controller']
numeric_features = [
    'RAM Capacity', 'Nominal Battery Capacity', 'Memory Capacity', 'CPU Clock',
    'GPU Clock', 'CPU_Composite_Score', 'GPU_Composite_Score'
]

# F. Fill missing values on the spec_data DataFrame
print("Filling missing values...")
for col in numeric_features:
    spec_data[col] = spec_data[col].fillna(spec_data[col].mean())
for col in categorical_features:
    spec_data[col] = spec_data[col].fillna('Unknown')


# G. Define the performance score function
def compute_performance_score(row):
    """Calculates a weighted performance score from static specs."""
    score = (
            row['RAM Capacity'] * 0.35 +
            row['CPU Clock'] * 0.2 +
            row['GPU Clock'] * 0.15 +
            row['CPU_Composite_Score'] * 0.3 +
            row['Memory Capacity'] * 0.10 +
            row['Nominal Battery Capacity'] * 0.10 +
            row['GPU_Composite_Score'] * 0.2
    )
    return round(score, 2)


# H. Create the Performance_Score and Hardware_Tier labels
spec_data['Performance_Score'] = spec_data.apply(compute_performance_score, axis=1)
spec_data.dropna(subset=['Performance_Score'], inplace=True)
scores = spec_data[['Performance_Score']].dropna()

# Use K-Means to find 3 natural clusters (hardware tiers)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scores)
spec_data.loc[scores.index, 'Hardware_Tier_Label'] = kmeans.labels_

# Map the cluster labels (0, 1, 2)
# First, find the mean score for each cluster to order them correctly
cluster_centers = kmeans.cluster_centers_.flatten()
label_order = np.argsort(cluster_centers)
label_mapping = {label_order[0]: 'Low-Tier', label_order[1]: 'Mid-Tier', label_order[2]: 'High-Tier'}

spec_data['Hardware_Tier'] = spec_data['Hardware_Tier_Label'].map(label_mapping)
# I. Create X and y from the DataFrame to train the classifier
X_train_static = spec_data[categorical_features + numeric_features]
y_train_static = spec_data['Hardware_Tier']

# J. Define and train the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

hardware_classifier_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

hardware_classifier_pipeline.fit(X_train_static, y_train_static)
print("Hardware classifier trained successfully.")


#DEFINE THE FINAL LABELING FUNCTIONS FOR NEW DATA
def assign_hardware_tier(static_data_dict, pipeline, training_df_columns):
    """Uses the trained pipeline to predict the hardware tier for new JSON data."""
    df = pd.DataFrame([static_data_dict])
    df = df.rename(columns={
        'cpu_identifier': 'CPU', 'cpu_clock_m_hz': 'CPU Clock', 'ram_type': 'RAM Type',
        'ram_capacity': 'RAM Capacity', 'gpu_renderer': 'Graphical Controller',
        'gpu_clock_m_hz': 'GPU Clock', 'nominal_capacity_mah': 'Nominal Battery Capacity',
        'memory_capacity_gb': 'Memory Capacity'
    })

    df['CPU_Composite_Score'] = df['CPU'].str.upper().str.strip().apply(lambda x: get_composite_score(x, cpu_score_map))
    df['GPU_Composite_Score'] = df['Graphical Controller'].str.upper().str.strip().apply(
        lambda x: get_composite_score(x, gpu_score_map))

    X_predict = df[categorical_features + numeric_features]

    # Fill NaNs using the means from the original training data
    for col in numeric_features:
        fill_value = training_df_columns[col].mean()
        X_predict[col] = X_predict[col].fillna(fill_value)
    for col in categorical_features:
        X_predict[col] = X_predict[col].fillna('Unknown')

    prediction = pipeline.predict(X_predict)
    return prediction[0]


def assign_activity_level(time_series_list):
    """Assigns a label based on overall CPU load and per-core utilization variance."""
    if not time_series_list:
        return "Unknown"

    # Convert time-series to a DataFrame
    ts_df = pd.DataFrame(time_series_list)

    # Expand the nested cpu core utilization dictionary into columns
    core_util_df = pd.json_normalize(ts_df['cpu_core_utilization_percent'])
    ts_df = pd.concat([ts_df.drop(['cpu_core_utilization_percent'], axis=1), core_util_df], axis=1)

    # 1. Get the average overall load
    avg_cpu_load = ts_df['cpu_load_percent'].mean()

    core_cols = [col for col in ts_df.columns if 'cpu' in str(col) and col != 'cpu_load_percent']
    avg_core_variance = ts_df[core_cols].var(axis=1).mean()

    activity_score = avg_cpu_load + (avg_core_variance * 0.5)

    if activity_score > 60:
        return "High"
    elif activity_score > 25:
        return "Medium"
    else:
        return "Low"


print("\nProcessing collected JSON files to generate labels...")
collected_data_folder = 'collected_data/'
metadata_list = []

if not os.path.exists(collected_data_folder) or not os.listdir(collected_data_folder):
    print(f"Warning: The '{collected_data_folder}' folder is empty or does not exist.")
    print("Please add your collected JSON files there to generate labels.")
else:
    for filename in os.listdir(collected_data_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(collected_data_folder, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Create a flat dictionary from the nested JSON for static data
                static_data_to_label = {
                    "cpu_identifier": data.get("cpu", {}).get("identifier"),
                    "ram_type": data.get("raminfo", {}).get("ram_type"),
                    "ram_capacity": data.get("raminfo", {}).get("ram_capacity"),
                    "nominal_capacity_mah": data.get("battery", {}).get("nominal_capacity_mah"),
                    "gpu_renderer": data.get("gpu", {}).get("renderer"),
                    "memory_capacity_gb": data.get("memory_capacity_gb"),
                    "cpu_clock_m_hz": data.get("cpu_clock_m_hz"),
                    "gpu_clock_m_hz": data.get("gpu_clock_m_hz")
                }

                # 1. Assign Hardware Tier using the flattened static data
                hardware_tier = assign_hardware_tier(static_data_to_label, hardware_classifier_pipeline, X_train_static)

                # 2. Assign Activity Level using the time-series
                activity_level = assign_activity_level(data.get('time_series', []))

                # 3. Append to our metadata list
                metadata_list.append({
                    'filename': filename,
                    'hardware_tier': hardware_tier,
                    'activity_level': activity_level
                })
                print(f"Processed {filename}: Hardware={hardware_tier}, Activity={activity_level}")

            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")


if metadata_list:
    output_folder="collected_data/"
    metadata_df = pd.DataFrame(metadata_list)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'metadata.csv')
    metadata_df.to_csv(output_path, index=False)
    print("\nSuccess! Metadata file 'metadata.csv' has been created.")
else:
    print("\nNo files were processed.")