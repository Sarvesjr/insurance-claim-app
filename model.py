import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import Tuple

def create_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic dataset for training claim prediction models.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with synthetic insurance claim data
    """
    np.random.seed(42)

    data = {
        'age': np.random.randint(18, 70, n_samples),
        'car_age': np.random.randint(0, 15, n_samples),
        'car_value': np.random.randint(200000, 5000000, n_samples),
        'previous_claims': np.random.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'damage_percent': np.random.uniform(5, 95, n_samples),
        'insurance_type': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }

    df = pd.DataFrame(data)

    approval_score = (
        (100 - df['damage_percent']) * 0.4 +
        (15 - df['car_age']) * 3 +
        (3 - df['previous_claims']) * 10 +
        df['insurance_type'] * 15 +
        np.random.uniform(-10, 10, n_samples)
    )

    df['claim_approved'] = (approval_score > 40).astype(int)

    df['claim_amount'] = df.apply(
        lambda row: (
            row['damage_percent'] * row['car_value'] / 100 *
            (0.8 if row['insurance_type'] == 1 else 0.5) *
            np.random.uniform(0.9, 1.1)
        ) if row['claim_approved'] == 1 else 0,
        axis=1
    )

    return df


def train_claim_approval_model() -> RandomForestClassifier:
    """
    Train Random Forest model to predict claim approval.

    Returns:
        Trained RandomForestClassifier
    """
    df = create_synthetic_dataset(1000)

    features = ['age', 'car_age', 'car_value', 'previous_claims', 'damage_percent', 'insurance_type']
    X = df[features]
    y = df['claim_approved']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    return model


def train_claim_amount_model() -> RandomForestRegressor:
    """
    Train Random Forest model to predict claim amount.

    Returns:
        Trained RandomForestRegressor
    """
    df = create_synthetic_dataset(1000)
    df_approved = df[df['claim_approved'] == 1]

    features = ['age', 'car_age', 'car_value', 'previous_claims', 'damage_percent', 'insurance_type']
    X = df_approved[features]
    y = df_approved['claim_amount']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    return model


def predict_claim(
    approval_model: RandomForestClassifier,
    amount_model: RandomForestRegressor,
    age: int,
    car_age: int,
    car_value: float,
    previous_claims: int,
    damage_percent: float,
    insurance_type: int
) -> Tuple[bool, float]:
    """
    Predict claim approval and amount.

    Args:
        approval_model: Trained approval classifier
        amount_model: Trained amount regressor
        age: User age
        car_age: Car age in years
        car_value: Estimated car value
        previous_claims: Number of previous claims
        damage_percent: Damage percentage from image analysis
        insurance_type: 0 for Third-party, 1 for Comprehensive

    Returns:
        Tuple of (is_approved, claim_amount)
    """
    features = np.array([[age, car_age, car_value, previous_claims, damage_percent, insurance_type]])

    approval_pred = approval_model.predict(features)[0]
    is_approved = bool(approval_pred)

    if is_approved:
        claim_amount = amount_model.predict(features)[0]
        claim_amount = max(0, claim_amount)
    else:
        claim_amount = 0.0

    return is_approved, claim_amount
