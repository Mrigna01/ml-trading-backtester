from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["return_1d", "return_5d", "return_10d", "sma_ratio",
            "rsi_14", "volatility_10d", "volume_ratio", "bb_position"]

def walk_forward_train(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[FEATURES]
    y = df["label"]

    results = []
    all_preds = pd.Series(index=df.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        all_preds.iloc[test_idx] = preds

        results.append({
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0)
        })

    return all_preds, results


def walk_forward_train_regression(df, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = df[FEATURES]
    y = df["future_return"]  # regress on the actual continuous return, not the binary label

    results = []
    all_preds = pd.Series(index=df.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        all_preds.iloc[test_idx] = preds

        results.append({
            "mae": mean_absolute_error(y_test, preds),
            "mean_predicted_return": preds.mean(),
            "mean_actual_return": y_test.mean()
        })

    return all_preds, results
