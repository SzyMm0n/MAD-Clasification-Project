# model_training.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline # Pipeline dla imblearn
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE

def define_base_models(class_weight='balanced') -> dict:
    """Definiuje podstawowe modele klasyfikacyjne."""
    log_reg = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight=class_weight, max_iter=1000)
    random_forest = RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weight)
    svm_clf = SVC(probability=True, random_state=RANDOM_STATE, class_weight=class_weight)
    
    return {
        "Logistic Regression": log_reg,
        "Random Forest": random_forest,
        "Support Vector Machine": svm_clf
    }

def create_full_model_pipeline(preprocessor, model_instance, use_smote: bool = True):
    """
    Tworzy pełny pipeline dla modelu, zawierający preprocesor, opcjonalnie SMOTE i klasyfikator.
    """
    steps = []
    steps.append(('preprocessor', preprocessor))
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        steps.append(('smote', smote))
    
    steps.append(('classifier', model_instance))
    
    return ImbPipeline(steps=steps)


def train_model(pipeline, X_train, y_train):
    """Trenuje podany pipeline modelu."""
    print(f"Trenowanie modelu: {pipeline.steps[-1][0]}...") # Nazwa ostatniego kroku (klasyfikatora)
    pipeline.fit(X_train, y_train)
    print("Model wytrenowany.")
    return pipeline


def define_voting_classifier_pipeline(
    preprocessor, 
    log_reg_model, 
    rf_model, 
    svm_model, 
    use_smote: bool = True
):
    """Definiuje VotingClassifier z podanymi modelami bazowymi i preprocesorem."""
    
    # Tworzymy pipeline'y dla modeli bazowych, które będą użyte w VotingClassifier
    # Każdy z nich musi zawierać preprocesor, ponieważ VotingClassifier nie przepuszcza danych przez globalny preprocesor
    # do każdego estymatora indywidualnie, jeśli sam nie jest częścią pipeline'u z preprocesorem na początku.
    # Jednak nasz główny pipeline *będzie* miał preprocesor na początku.
    
    # Estymatory dla VotingClassifier (już wytrenowane lub do wytrenowania wewnątrz VC)
    # W tym przypadku, chcemy, aby VotingClassifier trenował je od zera z preprocesowanymi danymi.
    # Dlatego przekazujemy instancje modeli, a nie pipeline'y.
    # Preprocessing i SMOTE będą obsługiwane przez główny pipeline VotingClassifier.

    voting_clf_instance = VotingClassifier(
        estimators=[
            ('lr', log_reg_model), # Przekazujemy surowe instancje modeli
            ('rf', rf_model),
            ('svc', svm_model)
        ],
        voting='soft'
    )

    steps = []
    steps.append(('preprocessor', preprocessor)) # Preprocessing danych wejściowych
    
    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        steps.append(('smote', smote)) # SMOTE na danych treningowych
        
    steps.append(('voting_classifier', voting_clf_instance)) # Sam VotingClassifier
    
    return ImbPipeline(steps=steps)


if __name__ == '__main__':
    # Przykładowe użycie (wymaga danych i preprocesora z preprocessing.py)
    from data_loader import load_data
    from preprocessing import handle_gender_other, split_data, get_feature_types, create_preprocessor
    
    raw_df = load_data()
    if not raw_df.empty:
        processed_df = handle_gender_other(raw_df)
        X_train_df, X_test_df, y_train_series, y_test_series = split_data(processed_df)
        
        num_feats, cat_feats = get_feature_types(X_train_df)
        preprocessor_obj = create_preprocessor(num_feats, cat_feats)

        # Preprocessing danych (dopasowanie na treningowym, transformacja obu)
        # UWAGA: Tutaj przekazujemy X_train_df (DataFrame) do fit_transform.
        # Preprocesor zadziała na oryginalnych danych.
        
        base_models_dict = define_base_models()
        
        for model_name, model_inst in base_models_dict.items():
            print(f"\n--- Tworzenie i trenowanie: {model_name} ---")
            # Tworzenie pipeline dla każdego modelu z preprocesorem
            # preprocessor_obj jest już zdefiniowany i będzie dopasowany wewnątrz `train_model`
            # gdy `pipeline.fit()` jest wywoływane.
            model_pipeline = create_full_model_pipeline(preprocessor_obj, model_inst, use_smote=True)
            
            # Trenowanie (X_train_df jest DataFrame, preprocessor sobie z tym poradzi)
            trained_pipeline = train_model(model_pipeline, X_train_df, y_train_series)
            
            # Tutaj można by dodać szybką predykcję na X_test_df dla weryfikacji
            # y_pred_test = trained_pipeline.predict(X_test_df)
            # print(f"Przykładowe predykcje dla {model_name} na X_test (pierwsze 5): {y_pred_test[:5]}")

        # Przykład dla Voting Classifier
        vc_pipeline = define_voting_classifier_pipeline(
            preprocessor_obj,
            base_models_dict["Logistic Regression"],
            base_models_dict["Random Forest"],
            base_models_dict["Support Vector Machine"],
            use_smote=True
        )
        print("\n--- Tworzenie i trenowanie: Voting Classifier ---")
        trained_vc_pipeline = train_model(vc_pipeline, X_train_df, y_train_series)
        # y_pred_vc_test = trained_vc_pipeline.predict(X_test_df)
        # print(f"Przykładowe predykcje dla Voting Classifier na X_test (pierwsze 5): {y_pred_vc_test[:5]}")