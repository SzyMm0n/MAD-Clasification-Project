# Importing the dataset from Kaggle using KaggleHub
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Path to the dataset file in the Kaggle dataset
file_path = "healthcare-dataset-stroke-data.csv"

# Loading the dataset
initial_dataset = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "fedesoriano/stroke-prediction-dataset",
  file_path,
)
