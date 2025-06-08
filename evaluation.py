# evaluation.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from config import RESULTS_DIR
import os

# Upewnij się, że katalog results istnieje
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_predictions(model, X_data) -> tuple[pd.Series, pd.Series]:
    """Generuje predykcje klasy i prawdopodobieństwa."""
    y_pred = model.predict(X_data)
    y_proba = model.predict_proba(X_data)[:, 1] # Prawdopodobieństwa dla klasy pozytywnej (udar)
    return y_pred, y_proba

def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    """Oblicza zestaw metryk ewaluacyjnych."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba)
    }
    return metrics

def print_evaluation_summary(model_name: str, y_true, y_pred, y_proba):
    """Drukuje podsumowanie ewaluacji: metryki, macierz pomyłek, raport klasyfikacyjny."""
    print(f"\n--- Ocena dla modelu: {model_name} ---")
    
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
        
    print("\nMacierz pomyłek:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nRaport klasyfikacyjny:")
    print(classification_report(y_true, y_pred, target_names=['Brak udaru (0)', 'Udar (1)']))
    
    return metrics, cm

def plot_confusion_matrix_heatmap(cm, model_name: str, save_plot: bool = False):
    """Rysuje macierz pomyłek jako heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Brak udaru (0)', 'Udar (1)'], 
                yticklabels=['Brak udaru (0)', 'Udar (1)'])
    plt.title(f'Macierz pomyłek dla {model_name}')
    plt.xlabel('Przewidziane')
    plt.ylabel('Rzeczywiste')
    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(RESULTS_DIR, f'cm_{model_name.replace(" ", "_")}.png'))
    plt.show()

def plot_roc_curve_custom(y_true, y_proba, model_name: str, ax=None, save_plot: bool = False):
    """Rysuje krzywą ROC. Może rysować na istniejącej osi (ax) dla porównań."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_standalone = True
    else:
        plot_standalone = False
        
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    if plot_standalone:
        ax.plot([0, 1], [0, 1], 'k--') # Linia losowego zgadywania
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Krzywa ROC dla {model_name}')
        ax.legend(loc='lower right')
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(RESULTS_DIR, f'roc_{model_name.replace(" ", "_")}.png'))
        plt.show()
    return ax # Zwraca oś dla ewentualnego dalszego rysowania

if __name__ == '__main__':
    # Przykładowe użycie - wymaga wytrenowanego modelu i danych testowych
    # Załóżmy, że mamy:
    # trained_model_pipeline (wytrenowany pipeline z model_training.py)
    # X_test_df, y_test_series (z preprocessing.py)

    # Dummy data for demonstration
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split as tts
    from sklearn.linear_model import LogisticRegression as LR
    
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=5, random_state=42)
    _, X_test_dummy, _, y_test_dummy = tts(X_dummy, y_dummy, test_size=0.3, random_state=42)
    
    # Dummy model
    dummy_model = LR(solver='liblinear').fit(X_dummy, y_dummy) # Trenujemy na całym dummy X, y
    
    print("--- Przykład użycia evaluation.py z danymi dummy ---")
    y_pred_dummy, y_proba_dummy = get_predictions(dummy_model, X_test_dummy)
    metrics_dummy, cm_dummy = print_evaluation_summary("Dummy Model", y_test_dummy, y_pred_dummy, y_proba_dummy)
    plot_confusion_matrix_heatmap(cm_dummy, "Dummy Model", save_plot=True)
    plot_roc_curve_custom(y_test_dummy, y_proba_dummy, "Dummy Model", save_plot=True)

    print("\nPrzykładowe metryki:")
    print(pd.DataFrame([metrics_dummy]))