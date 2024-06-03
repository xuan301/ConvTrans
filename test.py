import os
import glob
import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import Data_Loader, dataset_class, Setup, Initialization, Data_Verifier
from Models.model import model_factory
from Models.utils import load_model

# Define paths
model_dir = 'Results/Dataset/UEA'
test_set_data_dir = 'test_set/data'
output_dir = 'test_set/pred_result'
ts_temp_dir = 'test_set/ts_temp'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(ts_temp_dir, exist_ok=True)

# Find the latest model and its configuration
model_paths = glob.glob(os.path.join(model_dir, '**', 'checkpoints', 'ProcessedDatamodel_last.pth'), recursive=True)
latest_model_path = max(model_paths, key=os.path.getctime)
config_path = os.path.join(os.path.dirname(os.path.dirname(latest_model_path)), 'configuration.json')

print(f"Latest model: {latest_model_path}")
print(f"Configuration file: {config_path}")

# Load the configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Function to convert Excel files to TS format
def convert_excel_to_ts(excel_path, ts_path, num_timepoints):
    data = pd.read_excel(excel_path, header=None)
    with open(ts_path, 'w') as f:
        f.write(f"@problemName Test\n")
        f.write(f"@timeStamps false\n")
        f.write(f"@missing false\n")
        f.write(f"@univariate false\n")
        f.write(f"@dimensions 1\n")
        f.write(f"@equalLength true\n")
        f.write(f"@seriesLength {num_timepoints}\n")
        f.write(f"@classLabel true 1 2 3 4 5 6 7 8\n")
        f.write(f"@data\n")
        for index, row in data.iterrows():
            line_data = ",".join(map(str, row.values))
            f.write(f"{line_data}:-1\n")

# Convert each Excel file to TS format
for file in os.listdir(test_set_data_dir):
    if file.endswith('.xlsx'):
        excel_path = os.path.join(test_set_data_dir, file)
        ts_path = os.path.join(ts_temp_dir, file.replace('.xlsx', '_TEST.ts'))
        convert_excel_to_ts(excel_path, ts_path, 226)  # Adjust the series length accordingly

# Create a directory structure expected by Data_Loader
test_problem_dir = os.path.join(ts_temp_dir, 'TestProblem')
os.makedirs(test_problem_dir, exist_ok=True)

# Move the TEST.ts files to the new directory and create TRAIN.ts and VAL.ts as copies of TEST.ts
for file in os.listdir(ts_temp_dir):
    if file.endswith('_TEST.ts'):
        os.rename(os.path.join(ts_temp_dir, file), os.path.join(test_problem_dir, 'TestProblem_TEST.ts'))
        # Copy TEST.ts to TRAIN.ts and VAL.ts
        with open(os.path.join(test_problem_dir, 'TestProblem_TEST.ts'), 'r') as f:
            content = f.read()
        with open(os.path.join(test_problem_dir, 'TestProblem_TRAIN.ts'), 'w') as f:
            f.write(content)
        with open(os.path.join(test_problem_dir, 'TestProblem_VAL.ts'), 'w') as f:
            f.write(content)

# Load configuration and update paths
with open(config_path, 'r') as f:
    config = json.load(f)
config['data_dir'] = test_problem_dir
config['output_dir'] = 'Results/Dataset/UEA'

# Setup and Initialization
class Args:
    def __init__(self, config):
        self.__dict__.update(config)

args = Args(config)
config = Setup(args)
device = Initialization(config)
Data_Verifier(config)

# Load Data
print("Loading Data ...")
Data = Data_Loader(config)
test_dataset = dataset_class(Data['test_data'], Data['test_label'])
test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)

# Ensure 'num_labels' and 'Data_shape' are set correctly in the config
config['num_labels'] = 8  # Ensure this matches the number of classes in your model
config['Data_shape'] = Data['train_data'].shape

# Load the model
model = model_factory(config)
model.to(device)
model = load_model(model, latest_model_path)

# Predict
print("Predicting ...")
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        inputs, targets, _ = batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Read the original Excel file and save predictions
for file in os.listdir(test_set_data_dir):
    if file.endswith('.xlsx'):
        excel_path = os.path.join(test_set_data_dir, file)
        data = pd.read_excel(excel_path, header=None)
        data_values = data.values.astype(np.float32)

        # Add predictions as the first column
        result_df = pd.DataFrame(data_values)
        result_df.insert(0, 'PredictedLabel', predictions[:len(data_values)])

        # Save the result
        output_file_path = os.path.join(output_dir, file.replace('.xlsx', '_predresult.xlsx'))
        result_df.to_excel(output_file_path, index=False, header=False)

print(f"Predictions saved to {output_dir}")
