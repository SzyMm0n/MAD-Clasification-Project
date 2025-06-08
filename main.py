# main.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from config import (MODEL_SAVE_DIR, RANDOM_STATE, TARGET_COLUMN,
                    LOG_REG_MODEL_NAME, RF_MODEL_NAME, SVM_MODEL_NAME, VOTING_MODEL_NAME)
from data_loader import load_data
from eda import (display_basic_info, plot_numerical_distributions, plot_categorical_distributions,
                 plot_target_distribution, plot_bivariate_numerical, plot_bivariate_categorical,
                 plot_correlation_matrix)
from preprocessing import handle_gender_other, split_data, get_feature_types, create_preprocessor
from model_training import (define_base_models, create_full_model_pipeline, train_model,
                            define_voting_classifier_pipeline)
from evaluation import (get_predictions, print_evaluation_summary, 
                        plot_confusion_matrix_heatmap, plot_roc_curve_custom)

def run_eda(df: pd.DataFrame):
    """Uruchamia podstawowe funkcje EDA."""
    print("\nRozpoczynanie Eksploracyjnej Analizy Danych (EDA)...")
    display_basic_info(df.copy())
    
    num_feats, cat_feats = get_feature_types(df.drop(TARGET_COLUMN, axis=1)) # Na danych bez targetu
    
    plot_numerical_distributions(df.copy(), num_feats, save_plots=True)
    plot_categorical_distributions(df.copy(), cat_feats, save_plots=True)
    plot_target_distribution(df.copy(), save_plots=True)
    plot_bivariate_numerical(df.copy(), num_feats, save_plots=True)
    plot_bivariate_categorical(df.copy(), cat_feats, save_plots=True)
    
    features_for_corr_plot = df.select_dtypes(include=pd.np.number).columns.tolist()
    plot_correlation_matrix(df.copy(), features_for_corr_plot, save_plots=True)
    print("Zakończono EDA.")

def main_pipeline():
    """Główna funkcja uruchamiająca cały pipeline projektu."""
    
    # 0. Utwórz katalogi, jeśli nie istnieją
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 1. Wczytaj dane
    raw_df = load_data()
    if raw_df.empty:
        print("Nie udało się wczytać danych. Zakończenie pracy.")
        return

    # 2. Uruchom EDA (opcjonalnie, można zakomentować po pierwszym uruchomieniu)
    # run_eda(raw_df)

    # 3. Przetwarzanie wstępne
    print("\n--- Rozpoczynanie Przetwarzania Wstępnego ---")
    processed_df = handle_gender_other(raw_df)
    X_train_df, X_test_df, y_train_series, y_test_series = split_data(processed_df)
    
    # Identyfikacja typów cech na podstawie zbioru treningowego (po podziale!)
    numerical_features, categorical_features = get_feature_types(X_train_df)
    preprocessor_obj = create_preprocessor(numerical_features, categorical_features)
    print("Zakończono definiowanie preprocesora.")
    # Preprocesor zostanie dopasowany wewnątrz pipeline'u modelu na danych treningowych.

    # 4. Trenowanie i Ewaluacja Modeli Podstawowych
    print("\n--- Trenowanie i Ewaluacja Modeli Podstawowych ---")
    base_models_dict = define_base_models()
    trained_models = {}
    evaluation_results = {}
    
    # Oś do wspólnego wykresu ROC
    fig_roc_all, ax_roc_all = plt.subplots(figsize=(10, 8))
    ax_roc_all.plot([0, 1], [0, 1], 'k--', label='Losowe zgadywanie') # Linia bazowa

    model_filenames = {
        "Logistic Regression": LOG_REG_MODEL_NAME,
        "Random Forest": RF_MODEL_NAME,
        "Support Vector Machine": SVM_MODEL_NAME
    }

    for model_name, model_instance in base_models_dict.items():
        print(f"\n--- Model: {model_name} ---")
        model_pipeline = create_full_model_pipeline(preprocessor_obj, model_instance, use_smote=True)
        
        trained_pipeline = train_model(model_pipeline, X_train_df, y_train_series) # Trenuj na DataFrame
        trained_models[model_name] = trained_pipeline
        
        # Ewaluacja na zbiorze testowym (DataFrame)
        y_pred_test, y_proba_test = get_predictions(trained_pipeline, X_test_df)
        metrics, cm = print_evaluation_summary(model_name, y_test_series, y_pred_test, y_proba_test)
        evaluation_results[model_name] = metrics
        
        plot_confusion_matrix_heatmap(cm, model_name, save_plot=True)
        plot_roc_curve_custom(y_test_series, y_proba_test, model_name, ax=ax_roc_all) # Rysuj na wspólnej osi

        # Zapisz model
        model_path = os.path.join(MODEL_SAVE_DIR, model_filenames[model_name])
        joblib.dump(trained_pipeline, model_path)
        print(f"Model {model_name} zapisany w: {model_path}")

    # 5. Trenowanie i Ewaluacja Modelu Hybrydowego (Voting Classifier)
    print("\n--- Model Hybrydowy: Voting Classifier ---")
    # Użyj instancji modeli bazowych (nie wytrenowanych pipeline'ów) dla VotingClassifier
    vc_pipeline = define_voting_classifier_pipeline(
        preprocessor_obj,
        base_models_dict["Logistic Regression"], # Surowe instancje
        base_models_dict["Random Forest"],
        base_models_dict["Support Vector Machine"],
        use_smote=True
    )
    
    trained_vc_pipeline = train_model(vc_pipeline, X_train_df, y_train_series)
    trained_models["Voting Classifier"] = trained_vc_pipeline
    
    y_pred_vc_test, y_proba_vc_test = get_predictions(trained_vc_pipeline, X_test_df)
    metrics_vc, cm_vc = print_evaluation_summary("Voting Classifier", y_test_series, y_pred_vc_test, y_proba_vc_test)
    evaluation_results["Voting Classifier"] = metrics_vc
    
    plot_confusion_matrix_heatmap(cm_vc, "Voting Classifier", save_plot=True)
    plot_roc_curve_custom(y_test_series, y_proba_vc_test, "Voting Classifier", ax=ax_roc_all)
    
    model_path_vc = os.path.join(MODEL_SAVE_DIR, VOTING_MODEL_NAME)
    joblib.dump(trained_vc_pipeline, model_path_vc)
    print(f"Model Voting Classifier zapisany w: {model_path_vc}")

    # Finalizacja wspólnego wykresu ROC
    ax_roc_all.set_xlabel('False Positive Rate')
    ax_roc_all.set_ylabel('True Positive Rate')
    ax_roc_all.set_title('Krzywe ROC dla Porównywanych Modeli')
    ax_roc_all.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join("results", 'roc_curves_comparison.png')) # Zapisz wspólny wykres
    plt.show()

    # 6. Porównanie wyników
    print("\n--- Porównanie Wyników Ewaluacji Modeli ---")
    results_summary_df = pd.DataFrame(evaluation_results).T.sort_values(by="Recall", ascending=False)
    print(results_summary_df)
    results_summary_df.to_csv(os.path.join("results", "model_evaluation_summary.csv"))
    print("Podsumowanie wyników zapisane do pliku CSV.")

    # 7. Wybór najlepszego modelu i demonstracja użycia
    best_model_name = results_summary_df.index[0] # Model z najwyższym Recall
    print(f"\nNajlepszy model (wg Recall): {best_model_name}")
    
    # Załaduj najlepszy model (dla demonstracji)
    best_model_filename_map = {
        "Logistic Regression": LOG_REG_MODEL_NAME,
        "Random Forest": RF_MODEL_NAME,
        "Support Vector Machine": SVM_MODEL_NAME,
        "Voting Classifier": VOTING_MODEL_NAME
    }
    best_model_path = os.path.join(MODEL_SAVE_DIR, best_model_filename_map[best_model_name])
    
    try:
        loaded_best_model = joblib.load(best_model_path)
        print(f"Pomyślnie załadowano najlepszy model: {best_model_path}")

        # Przygotuj przykładowe nowe dane (jeden wiersz DataFrame)
        # Upewnij się, że kolumny są w tej samej kolejności co w X_train_df
        sample_patient_data = pd.DataFrame({
            'gender': ['Female'], 'age': [72.0], 'hypertension': [1], 'heart_disease': [0],
            'ever_married': ['Yes'], 'work_type': ['Self-employed'], 
            'Residence_type': ['Urban'], 'avg_glucose_level': [105.22], 'bmi': [28.89], # Mediana BMI z EDA
            'smoking_status': ['formerly smoked']
        }, columns=X_train_df.columns) # Użyj oryginalnych kolumn X_train_df
        
        print("\nPrzykładowe dane nowego pacjenta:")
        print(sample_patient_data)

        pred_class = loaded_best_model.predict(sample_patient_data)
        pred_proba = loaded_best_model.predict_proba(sample_patient_data)

        print(f"\nPredykcja dla nowego pacjenta ({best_model_name}):")
        print(f"Przewidziana klasa: {pred_class[0]} (0: brak udaru, 1: udar)")
        print(f"Prawdopodobieństwo udaru: {pred_proba[0][1]:.4f}")

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono zapisanego najlepszego modelu: {best_model_path}")
    except Exception as e:
        print(f"Wystąpił błąd podczas ładowania lub używania najlepszego modelu: {e}")
        
    print("\n--- Zakończono Główny Pipeline ---")

if __name__ == '__main__':
    main_pipeline()