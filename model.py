import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys

def train_and_predict():
    try:
        print("Ładowanie danych z pliku titanic_data.csv...")
        data = pd.read_csv('titanic_data.csv')
        print("Lokalny plik titanic_data.csv załadowany pomyślnie.")

        # Sprawdź, czy dane mają wystarczającą liczbę próbek
        if len(data) < 2:
            raise ValueError("Zbiór danych jest za mały do treningu (wymagane minimum 2 próbki).")

        # Przygotuj cechy i etykiety (użyj małych liter zgodnie z titanic_data.csv)
        X = data[['pclass', 'sex', 'age', 'fare', 'family_size', 'embarked_Q', 'embarked_S']]
        y = data['survived']
        print("Cechy i etykiety przygotowane.")

        # Podziel dane na treningowe i testowe
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("Dane podzielone na zbiór treningowy i testowy.")

        # Trenuj model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model wytrenowany pomyślnie.")

        # Zapisz model
        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)
        print("Model zapisany jako model.pkl")

        # Wykonaj predykcje
        predictions = model.predict(X_test)
        print("Predykcje zakończone.")

        return predictions, y_test
    except Exception as e:
        print(f"Błąd w funkcji train_and_predict: {e}", file=sys.stderr)
        raise

def get_accuracy(predictions, y_test):
    try:
        print("Obliczanie dokładności...")
        accuracy = np.mean(predictions == y_test) * 100
        print(f"Dokładność: {accuracy}%")
        return accuracy
    except Exception as e:
        print(f"Błąd w funkcji get_accuracy: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    print("Uruchamianie model.py jako główny skrypt...")
    preds, y_test = train_and_predict()
    accuracy = get_accuracy(preds, y_test)
    print(f"Skrypt zakończony. Ostateczna dokładność: {accuracy}%")
