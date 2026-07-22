"""
Run this LOCALLY (not on Streamlit Cloud) to repair `xgboost_rul_model.pkl`.

WHY: the dashboard fails with
    Check failed: std::isalpha(header[1]): Invalid serialization file
This means the bytes in `xgboost_rul_model.pkl` are not a format the
installed XGBoost (3.3.0 on Streamlit Cloud) can read. The safest permanent
fix is to re-export the model into XGBoost's own JSON format, which is
stable across versions and never has this problem again.

You don't need to already know the original xgboost version — this script
tries several ways to open the file and reports which one worked.

HOW TO USE
----------
1. Put this script in the same folder as your `xgboost_rul_model.pkl`.
2. Create a clean virtual environment and install a candidate xgboost
   version, e.g.:
       python -m venv venv && source venv/bin/activate      # (or venv\\Scripts\\activate on Windows)
       pip install xgboost==1.7.6 scikit-learn joblib
   If step 3 below fails, try another version — common candidates to try,
   in order: 1.7.6, 2.0.3, 2.1.4, 1.6.2, 1.5.2, 3.0.0.
3. Run:
       python fix_model_format.py xgboost_rul_model.pkl
4. On success this creates `xgboost_rul_model.json` next to the original
   file. Update MODEL_PATH in app.py to point at the .json file, and change
   the loader to use xgb.Booster().load_model(...) (the app.py you were
   given already tries this automatically as a fallback, so simply
   replacing the .pkl with the new .json — same filename base — and
   re-deploying is enough).
5. Also pin the SAME xgboost version that worked here in requirements.txt,
   as a belt-and-suspenders measure (the JSON format shouldn't need it, but
   it keeps behavior identical to training).
"""

import sys
import pickle
import joblib


def try_strategies(path):
    attempts = []

    # Strategy 1: joblib.load (most common for sklearn-API XGBRegressor)
    try:
        model = joblib.load(path)
        return model, "joblib.load"
    except Exception as e:
        attempts.append(f"joblib.load failed: {e}")

    # Strategy 2: plain pickle.load
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model, "pickle.load"
    except Exception as e:
        attempts.append(f"pickle.load failed: {e}")

    # Strategy 3: maybe it's already a native XGBoost dump under a .pkl name
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(path)
        return booster, "xgb.Booster.load_model (already native format!)"
    except Exception as e:
        attempts.append(f"xgb.Booster.load_model failed: {e}")

    print("\n".join(attempts))
    return None, None


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_model_format.py xgboost_rul_model.pkl")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Attempting to open '{path}' ...")
    model, strategy = try_strategies(path)

    if model is None:
        print(
            "\nAll strategies failed with the xgboost version currently installed "
            "in this environment.\n"
            "-> Try again after installing a DIFFERENT xgboost version "
            "(pip install xgboost==<version>) — see the candidate list in the "
            "docstring at the top of this script.\n"
            "-> Run: pip show xgboost   to confirm what's active right now."
        )
        sys.exit(1)

    print(f"Loaded successfully using: {strategy}")

    # Extract the underlying Booster regardless of what we loaded
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif hasattr(model, "save_model"):
        booster = model  # already a raw Booster
    else:
        print(f"Loaded object is of type {type(model)} and has no get_booster()/"
              f"save_model() — please inspect it manually.")
        sys.exit(1)

    out_path = path.rsplit(".", 1)[0] + ".json"
    booster.save_model(out_path)
    print(f"\nSaved version-stable model to: {out_path}")

    try:
        import xgboost as xgb
        print(f"(Environment xgboost version used for this export: {xgb.__version__})")
    except Exception:
        pass


if __name__ == "__main__":
    main()
