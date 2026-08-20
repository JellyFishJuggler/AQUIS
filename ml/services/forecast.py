import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def run_forecast(observations):
    values = [float(o["groundwater_level"]) for o in observations if o.get("groundwater_level") is not None]
    timestamps = [o["timestamp"] for o in observations if o.get("groundwater_level") is not None]

    if len(values) < 10:
        return {"status": "insufficient_data", "count": len(values)}

    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)

    split = int(len(values) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    models = {}
    for name, pred in [("linear_regression", lr_pred), ("random_forest", rf_pred)]:
        models[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }

    best_name = min(models, key=lambda k: models[k]["rmse"])
    best_model = lr if best_name == "linear_regression" else rf

    future_steps = 12
    future_X = np.arange(len(values), len(values) + future_steps).reshape(-1, 1)
    future_pred = best_model.predict(future_X)

    predictions = []
    for i, val in enumerate(future_pred):
        predictions.append({
            "step": i + 1,
            "prediction": round(float(val), 4),
        })

    return {
        "station_id": observations[0].get("station_id"),
        "status": "success",
        "model": best_name,
        "models_compared": models,
        "predictions": predictions,
        "training_size": split,
        "evaluation_size": len(values) - split,
    }
