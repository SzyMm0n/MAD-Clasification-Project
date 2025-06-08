# config.py

# Ścieżki
DATASET_PATH = 'data/healthcare-dataset-stroke-data.csv'
MODEL_SAVE_DIR = 'models/'
RESULTS_DIR = 'results/' # Możesz tu zapisywać wykresy, tabele itp.

# Parametry
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'stroke'

# Nazwy plików modeli
LOG_REG_MODEL_NAME = "logistic_regression_model.joblib"
RF_MODEL_NAME = "random_forest_model.joblib"
SVM_MODEL_NAME = "svm_model.joblib"
VOTING_MODEL_NAME = "voting_classifier_model.joblib"

# Możesz dodać inne konfiguracje, np. parametry dla GridSearchCV