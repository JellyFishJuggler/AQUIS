import numpy as np
from sklearn.ensemble import IsolationForest


def run_anomaly_detection(observations):
    values = [float(o["groundwater_level"]) for o in observations if o.get("groundwater_level") is not None]

    if len(values) < 20:
        return {"status": "insufficient_data", "count": len(values)}

    X = np.array(values).reshape(-1, 1)

    clf = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    labels = clf.fit_predict(X)
    scores = clf.decision_function(X)

    anomalies = []
    for i, (label, score) in enumerate(zip(labels, scores)):
        if label == -1:
            anomalies.append({
                "index": i,
                "groundwater_level": values[i],
                "anomaly_score": round(float(score), 6),
                "is_anomaly": True,
            })

    normal_scores = scores[labels == 1]
    mean_score = float(np.mean(normal_scores))
    std_score = float(np.std(normal_scores))

    return {
        "status": "success",
        "total_observations": len(values),
        "anomaly_count": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(values), 4),
        "baseline_mean": round(mean_score, 6),
        "baseline_std": round(std_score, 6),
        "anomalies": anomalies,
    }
