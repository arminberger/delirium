import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load files
auc_logistic = pd.read_csv("auc_list_logistic.csv")
auc_xgb = pd.read_csv("auc_list_xgb.csv")
auc_svm = pd.read_csv("auc_list_svm.csv")

# Plot the AUC values against the number of features (row number)
plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(1, len(auc_logistic) + 1),
    auc_logistic["auc"],
    label="Logistic Regression",
)
plt.plot(np.arange(1, len(auc_xgb) + 1), auc_xgb["auc"], label="XGBoost")
plt.plot(np.arange(1, len(auc_svm) + 1), auc_svm["auc"], label="SVM")
plt.xlabel("Number of Features")
plt.ylim(0.65, 0.85)
plt.ylabel("AUC")
plt.title("AUC vs Number of Features")
plt.legend()
plt.grid(True)
plt.show()
