import pandas as pd
import numpy as np

# Load files

imp_logistic = pd.read_csv("feature_importance_logistic.csv")
imp_xgb = pd.read_csv("feature_importance_xgb.csv")
imp_svm = pd.read_csv("feature_importance_svm.csv")

imps = [imp_logistic, imp_xgb, imp_svm]

# Normalize importance to sum to 1
imps_normalized = []
for imp in imps:
    imp["importance"] = imp["importance"] / imp["importance"].sum()
    imps_normalized.append(imp)

overall_importance = pd.concat(imps_normalized, axis=0, ignore_index=True)
overall_importance = overall_importance.groupby("feature").mean().reset_index()

# Get the final ranking
overall_importance = overall_importance.sort_values("importance", ascending=False)
overall_importance["rank"] = np.arange(1, len(overall_importance) + 1)

# Save the final ranking to a CSV file
overall_importance.to_csv("final_feature_importance.csv", index=False)
