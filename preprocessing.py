# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from config import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

def handle_gender_other(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa wiersze z 'gender' == 'Other', jeśli jest ich bardzo mało."""
    if 'gender' in df.columns and 'Other' in df['gender'].unique():
        count_other = df[df['gender'] == 'Other'].shape[0]
        print(f"Liczba wystąpień 'Other' w 'gender': {count_other}")
        if 0 < count_other <= 5: # Arbitralny mały próg
            print(f"Usuwanie {count_other} wierszy z 'gender' == 'Other'.")
            df = df[df['gender'] != 'Other'].copy() # Użyj .copy() aby uniknąć SettingWithCopyWarning
            print(f"Nowe wymiary danych po usunięciu 'Other': {df.shape}")
    return df

def get_feature_types(df: pd.DataFrame) -> tuple[list, list]:
    """Identyfikuje cechy numeryczne i kategoryczne, wykluczając kolumnę celu."""
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COLUMN in numerical_features:
        numerical_features.remove(TARGET_COLUMN)
    
    categorical_features = df.select_dtypes(include='object').columns.tolist()
    # Upewnijmy się, że kolumna celu nie jest na liście kategorycznych (choć nie powinna)
    if TARGET_COLUMN in categorical_features:
        categorical_features.remove(TARGET_COLUMN)
        
    print(f"Zidentyfikowane cechy numeryczne (predyktory): {numerical_features}")
    print(f"Zidentyfikowane cechy kategoryczne: {categorical_features}")
    return numerical_features, categorical_features

def create_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """Tworzy preprocesor ColumnTransformer dla cech numerycznych i kategorycznych."""
    numerical_transformer = SimpleImputer(strategy='median') # Krok 1: Imputacja medianą
    # StandardScaler będzie dodany w pipeline modelu lub po imputacji globalnie

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
    
    # UWAGA: StandardScaler jest zwykle stosowany PO imputacji.
    # Można go włączyć tutaj lub osobno. Dla uproszczenia, imputacja jest tu, skalowanie będzie w pipeline modelu.
    # Alternatywnie, można zdefiniować pełny pipeline numeryczny:
    numerical_pipeline_full = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Dodajemy StandardScaler tutaj
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline_full, numerical_features), # Używamy pełnego pipeline numerycznego
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Pozostawia inne kolumny (np. już przetworzone) bez zmian
    )
    return preprocessor

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Dzieli dane na zbiory X_train, X_test, y_train, y_test."""
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Rozmiar X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Proporcje klas w y_train:\n{y_train.value_counts(normalize=True)}")
    print(f"Proporcje klas w y_test:\n{y_test.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    from data_loader import load_data
    
    raw_df = load_data()
    if not raw_df.empty:
        processed_df = handle_gender_other(raw_df)
        
        X_train_df, X_test_df, y_train_series, y_test_series = split_data(processed_df)
        
        num_feats, cat_feats = get_feature_types(X_train_df) # Użyj X_train do identyfikacji cech
        
        preprocessor_obj = create_preprocessor(num_feats, cat_feats)
        
        # Dopasuj preprocesor TYLKO na danych treningowych
        X_train_processed_np = preprocessor_obj.fit_transform(X_train_df)
        X_test_processed_np = preprocessor_obj.transform(X_test_df)
        
        # Konwersja z powrotem do DataFrame (opcjonalnie, dla wglądu)
        # Nazwy kolumn po OneHotEncoding mogą być skomplikowane
        try:
            feature_names_out = preprocessor_obj.get_feature_names_out()
            X_train_processed_df = pd.DataFrame(X_train_processed_np, columns=feature_names_out, index=X_train_df.index)
            X_test_processed_df = pd.DataFrame(X_test_processed_np, columns=feature_names_out, index=X_test_df.index)
            print("\nPrzetworzone X_train (pierwsze 5 wierszy):")
            print(X_train_processed_df.head())
        except Exception as e:
            print(f"Nie można pobrać nazw cech z preprocesora: {e}")
            print("X_train_processed jest tablicą NumPy.")

        print(f"\nKształt przetworzonego X_train: {X_train_processed_np.shape}")
        print(f"Kształt przetworzonego X_test: {X_test_processed_np.shape}")