import numpy as np
from model import train_and_predict, get_accuracy

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """
    Test 2 (na maksymalną ocenę 5): Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    preds, y_test = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == len(y_test), "Predictions length should match test samples length."

def test_predictions_value_range():
    """
    Test 3 (na maksymalną ocenę 5): Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie: Dla modelu Titanic mamy 2 klasy (0, 1).
    """
    preds, _ = train_and_predict()
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in range [0, 1]."

def test_model_accuracy():
    """
    Test 4 (na maksymalną ocenę 5): Sprawdza, czy model osiąga co najmniej 70% dokładności (przykładowy warunek, można dostosować do potrzeb).
    """
    preds, y_test = train_and_predict()
    accuracy = get_accuracy(preds, y_test)
    assert accuracy >= 70, f"Model accuracy {accuracy}% is below 70%."

