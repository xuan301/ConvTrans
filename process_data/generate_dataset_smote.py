import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# Define the input folder and output file
input_folder = "old_data"
output_folder = "ProcessedData"
output_train_file = os.path.join(output_folder, "ProcessedData_TRAIN.ts")
output_test_file = os.path.join(output_folder, "ProcessedData_TEST.ts")

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get all xlsx files in the folder
files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]
files.sort()  # Ensure files are sorted in the correct order

# Initialize lists to store data and labels
all_data = []
labels = []

# Process each file
for i, file in enumerate(files):
    file_path = os.path.join(input_folder, file)
    data = pd.read_excel(file_path, header=None)  # Read the data from the file
    num_samples, num_timepoints = data.shape

    if file == 'f1.xlsx':
        # For f1.xlsx, use KMeans to create clusters and then apply SMOTE
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        sampling_strategy = {label: 1000 // len(unique) for label in unique}
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        data, _ = smote.fit_resample(data, cluster_labels)
        
        # Select only 1000 samples
        data = data[:1000]
        
    for index, row in data.iterrows():
        all_data.append(row.values)
        labels.append(i + 1)  # Assign the label based on the file index

# Convert lists to numpy arrays for easier manipulation
all_data = np.array(all_data)
labels = np.array(labels)

# Split the data into training and testing sets with a 4:1 ratio
train_data, test_data, train_labels, test_labels = train_test_split(
    all_data, labels, test_size=0.2, stratify=labels, random_state=42)

# Function to write data to a .ts file
def write_ts_file(filename, data, labels, num_timepoints, num_classes):
    with open(filename, 'w') as f:
        # Write metadata
        f.write(f"@problemName ProcessedData\n")
        f.write(f"@timeStamps false\n")
        f.write(f"@missing false\n")
        f.write(f"@univariate false\n")
        f.write(f"@dimensions 1\n")  # Since each file has 1 dimension
        f.write(f"@equalLength true\n")
        f.write(f"@seriesLength {num_timepoints}\n")
        f.write(f"@classLabel true {' '.join(map(str, range(1, num_classes + 1)))}\n")
        f.write(f"@data\n")
        
        # Write the data
        for data, label in zip(data, labels):
            line_data = ",".join(map(str, data))
            f.write(f"{line_data}:{label}\n")

# Get the number of classes
num_classes = len(files)

# Write the training data to a .ts file
write_ts_file(output_train_file, train_data, train_labels, num_timepoints, num_classes)

# Write the testing data to a .ts file
write_ts_file(output_test_file, test_data, test_labels, num_timepoints, num_classes)

print(f"Data successfully processed and saved to {output_train_file} and {output_test_file}")
