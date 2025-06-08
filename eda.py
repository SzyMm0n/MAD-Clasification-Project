# eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_COLUMN, RESULTS_DIR
import os

# Upewnij się, że katalog results istnieje
os.makedirs(RESULTS_DIR, exist_ok=True)

def display_basic_info(df: pd.DataFrame):
    """Wyświetla podstawowe informacje o DataFrame."""
    print("\n--- Podstawowe informacje o danych ---")
    print("Pierwsze 5 wierszy:")
    print(df.head())
    print("\nOstatnie 5 wierszy:")
    print(df.tail())
    print(f"\nWymiary zbioru danych: {df.shape}")
    print("\nInformacje o typach danych i brakach:")
    df.info()
    print("\nStatystyki opisowe dla cech numerycznych:")
    print(df.describe())
    print("\nLiczba unikalnych wartości w każdej kolumnie:")
    for col in df.columns:
        print(f"Kolumna '{col}': {df[col].nunique()} unikalnych wartości")

def plot_numerical_distributions(df: pd.DataFrame, numerical_features: list, save_plots: bool = False):
    """Rysuje histogramy i boxploty dla cech numerycznych."""
    print("\n--- Rozkłady cech numerycznych ---")
    for col in numerical_features:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram dla {col}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot dla {col}')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(RESULTS_DIR, f'dist_{col}.png'))
        plt.show()

def plot_categorical_distributions(df: pd.DataFrame, categorical_features: list, save_plots: bool = False):
    """Rysuje wykresy słupkowe dla cech kategorycznych."""
    print("\n--- Rozkłady cech kategorycznych ---")
    for col in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order = df[col].value_counts().index)
        plt.title(f'Częstość dla {col}')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(RESULTS_DIR, f'count_{col}.png'))
        plt.show()

def plot_target_distribution(df: pd.DataFrame, save_plots: bool = False):
    """Rysuje rozkład zmiennej celu."""
    print(f"\n--- Rozkład zmiennej celu ({TARGET_COLUMN}) ---")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[TARGET_COLUMN])
    plt.title(f'Rozkład zmiennej celu ({TARGET_COLUMN})')
    if save_plots:
        plt.savefig(os.path.join(RESULTS_DIR, f'target_dist_{TARGET_COLUMN}.png'))
    plt.show()
    print(df[TARGET_COLUMN].value_counts(normalize=True))

def plot_bivariate_numerical(df: pd.DataFrame, numerical_features: list, save_plots: bool = False):
    """Analiza dwuwymiarowa: cechy numeryczne vs zmienna celu."""
    print("\n--- Analiza dwuwymiarowa: Cechy numeryczne vs Udar ---")
    for col in numerical_features:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(x=TARGET_COLUMN, y=col, data=df)
        plt.title(f'{col} vs {TARGET_COLUMN}')

        plt.subplot(1, 2, 2)
        sns.histplot(data=df, x=col, hue=TARGET_COLUMN, kde=True, multiple="stack")
        plt.title(f'Rozkład {col} w zależności od {TARGET_COLUMN}')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(RESULTS_DIR, f'bivar_num_{col}_vs_{TARGET_COLUMN}.png'))
        plt.show()

def plot_bivariate_categorical(df: pd.DataFrame, categorical_features: list, save_plots: bool = False):
    """Analiza dwuwymiarowa: cechy kategoryczne vs zmienna celu."""
    print("\n--- Analiza dwuwymiarowa: Cechy kategoryczne vs Udar ---")
    for col in categorical_features:
        plt.figure(figsize=(12, 7))
        sns.countplot(x=col, hue=TARGET_COLUMN, data=df, order = df[col].value_counts().index)
        plt.title(f'{col} vs {TARGET_COLUMN}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(RESULTS_DIR, f'bivar_cat_{col}_vs_{TARGET_COLUMN}.png'))
        plt.show()

        cross_tab = pd.crosstab(df[col], df[TARGET_COLUMN], normalize='index') * 100
        print(f"\nProcentowy udział udaru dla kategorii w '{col}':")
        print(cross_tab)
        cross_tab.plot(kind='bar', stacked=True, figsize=(10,6))
        plt.title(f'Procentowy udział udaru dla {col}')
        plt.ylabel('Procent')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(RESULTS_DIR, f'bivar_cat_percent_{col}_vs_{TARGET_COLUMN}.png'))
        plt.show()


def plot_correlation_matrix(df: pd.DataFrame, features_for_corr: list, save_plots: bool = False):
    """Rysuje macierz korelacji."""
    print("\n--- Macierz korelacji ---")
    # Tymczasowe kodowanie dla wizualizacji - ostrożnie z interpretacją
    temp_df_for_corr = df.copy()
    for col in temp_df_for_corr.columns:
        if temp_df_for_corr[col].dtype == 'object':
            temp_df_for_corr[col] = temp_df_for_corr[col].astype('category').cat.codes
    
    valid_cols_for_corr = [col for col in features_for_corr if col in temp_df_for_corr.columns and pd.api.types.is_numeric_dtype(temp_df_for_corr[col])]

    if not valid_cols_for_corr:
        print("Nie znaleziono odpowiednich kolumn numerycznych do obliczenia macierzy korelacji.")
        return

    correlation_matrix = temp_df_for_corr[valid_cols_for_corr].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Macierz Korelacji')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(RESULTS_DIR, 'correlation_matrix.png'))
    plt.show()

if __name__ == '__main__':
    # Przykładowe użycie funkcji EDA
    from data_loader import load_data
    from preprocessing import get_feature_types # Załóżmy, że ta funkcja istnieje
    
    raw_df = load_data()
    if not raw_df.empty:
        display_basic_info(raw_df.copy()) # Przekaż kopię, aby uniknąć modyfikacji oryginału w funkcjach EDA

        # Wstępna identyfikacja typów cech (może być udoskonalona)
        numerical_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = raw_df.select_dtypes(include='object').columns.tolist()
        if TARGET_COLUMN in numerical_cols:
            numerical_cols.remove(TARGET_COLUMN)

        plot_numerical_distributions(raw_df.copy(), numerical_cols, save_plots=True)
        plot_categorical_distributions(raw_df.copy(), categorical_cols, save_plots=True)
        plot_target_distribution(raw_df.copy(), save_plots=True)
        plot_bivariate_numerical(raw_df.copy(), numerical_cols, save_plots=True)
        plot_bivariate_categorical(raw_df.copy(), categorical_cols, save_plots=True)
        
        # Cechy do macierzy korelacji (wszystkie numeryczne + zmienna celu)
        features_for_corr_plot = raw_df.select_dtypes(include=np.number).columns.tolist()
        plot_correlation_matrix(raw_df.copy(), features_for_corr_plot, save_plots=True)