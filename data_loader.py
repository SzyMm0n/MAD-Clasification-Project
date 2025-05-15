# data_loader.py
import pandas as pd
from config import DATASET_PATH

def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Wczytuje zbiór danych z podanej ścieżki.
    Usuwa kolumnę 'id' jeśli istnieje.
    """
    try:
        df = pd.read_csv(path)
        print(f"Wczytano dane z: {path}")
        if 'id' in df.columns:
            df.drop('id', axis=1, inplace=True)
            print("Usunięto kolumnę 'id'.")
        return df
    except FileNotFoundError:
        print(f"Błąd: Plik '{path}' nie został znaleziony.")
        return pd.DataFrame() # Zwróć pusty DataFrame w przypadku błędu

if __name__ == '__main__':
    # Prosty test wczytywania
    data = load_data()
    if not data.empty:
        print("\nPierwsze 5 wierszy wczytanych danych:")
        print(data.head())