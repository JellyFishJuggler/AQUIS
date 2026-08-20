import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


STATUS_ORDER = {"SAFE": 0, "SEMI_CRITICAL": 1, "CRITICAL": 2, "OVER_EXPLOITED": 3}


def run_risk_assessment(records):
    if len(records) < 2:
        return {"status": "insufficient_data", "count": len(records)}

    sorted_records = sorted(records, key=lambda r: r.get("assessment_year", ""))

    current = sorted_records[-1]
    prev = sorted_records[-2] if len(sorted_records) >= 2 else current

    current_status = current.get("resource_category") or current.get("status") or "SAFE"
    current_stage = float(current.get("stage_of_extraction") or current.get("extraction_rate_pct") or 0)
    prev_stage = float(prev.get("stage_of_extraction") or prev.get("extraction_rate_pct") or 0)

    stage_change = current_stage - prev_stage

    risk_score = 0.0
    risk_category = "low"

    if current_status == "OVER_EXPLOITED":
        risk_score = 0.9
        risk_category = "critical"
    elif current_status == "CRITICAL":
        risk_score = 0.7
        risk_category = "high"
    elif current_status == "SEMI_CRITICAL":
        risk_score = 0.4
        risk_category = "medium"
    else:
        risk_score = 0.1
        risk_category = "low"

    if stage_change > 10:
        risk_score = min(1.0, risk_score + 0.2)
    elif stage_change > 5:
        risk_score = min(1.0, risk_score + 0.1)

    if len(sorted_records) >= 3:
        stages = [float(r.get("stage_of_extraction") or r.get("extraction_rate_pct") or 0) for r in sorted_records]
        trend = np.polyfit(range(len(stages)), stages, 1)[0]
        if trend > 5:
            risk_score = min(1.0, risk_score + 0.15)

    return {
        "status": "success",
        "unit_id": current.get("assessment_unit_id"),
        "current_status": current_status,
        "current_stage_of_extraction": current_stage,
        "previous_stage_of_extraction": prev_stage,
        "stage_change": round(stage_change, 2),
        "risk_score": round(risk_score, 4),
        "risk_category": risk_category,
        "years_analyzed": len(sorted_records),
        "assessment_years": [r.get("assessment_year") for r in sorted_records],
    }


def train_risk_model(records):
    if len(records) < 20:
        return {"status": "insufficient_data", "message": "Need at least 20 records for training"}

    features_list = []
    labels = []

    by_unit = {}
    for r in records:
        uid = r.get("assessment_unit_id")
        if uid not in by_unit:
            by_unit[uid] = []
        by_unit[uid].append(r)

    for uid, unit_records in by_unit.items():
        sorted_recs = sorted(unit_records, key=lambda r: r.get("assessment_year", ""))
        if len(sorted_recs) < 2:
            continue
        for i in range(1, len(sorted_recs)):
            prev_r = sorted_recs[i - 1]
            curr_r = sorted_recs[i]
            prev_stage = float(prev_r.get("stage_of_extraction") or prev_r.get("extraction_rate_pct") or 0)
            curr_stage = float(curr_r.get("stage_of_extraction") or curr_r.get("extraction_rate_pct") or 0)
            prev_extraction = float(prev_r.get("total_extraction") or prev_r.get("total_extraction_ham") or 0)
            curr_extraction = float(curr_r.get("total_extraction") or curr_r.get("total_extraction_ham") or 0)
            prev_recharge = float(prev_r.get("recharge") or prev_r.get("recharge_rainfall_ham") or 0)
            curr_recharge = float(curr_r.get("recharge") or curr_r.get("recharge_rainfall_ham") or 0)

            features_list.append([
                prev_stage, curr_stage,
                curr_stage - prev_stage,
                curr_extraction - prev_extraction,
                curr_recharge - prev_recharge,
                curr_extraction, curr_recharge,
            ])
            curr_status = curr_r.get("resource_category") or curr_r.get("status") or "SAFE"
            labels.append(STATUS_ORDER.get(curr_status, 0))

    if len(features_list) < 20:
        return {"status": "insufficient_data", "message": "Not enough paired records"}

    X = np.array(features_list)
    y = np.array(labels)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    clf.fit(X, y)

    importances = clf.feature_importances_
    feature_names = [
        "prev_stage", "curr_stage", "stage_change",
        "extraction_change", "recharge_change",
        "curr_extraction", "curr_recharge",
    ]
    feature_importance = dict(zip(feature_names, [round(float(v), 4) for v in importances]))

    return {
        "status": "success",
        "task": "risk",
        "model_type": "random_forest_classifier",
        "training_samples": len(features_list),
        "features": feature_names,
        "feature_importance": feature_importance,
        "classes": list(STATUS_ORDER.keys()),
        "message": "Model trained. Persist model with joblib for production use.",
    }
