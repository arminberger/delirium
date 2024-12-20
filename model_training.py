import os
import pandas as pd
import numpy as np
from torch.nn.functional import threshold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from constants import FEATURE_GROUPS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    auc,
    roc_curve,
    average_precision_score,
)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier
from ensemble import EnsembleModel

### Define constants
data_path = "processed_data"
df_imputed = pd.read_csv(os.path.join(data_path, "df_imputed.csv"))
df_not_imputed = pd.read_csv(os.path.join(data_path, "df_not_imputed.csv"))
df_imputed_complex = pd.read_csv(os.path.join(data_path, "df_imputed_complex.csv"))
df_used = df_imputed_complex.copy()
feature_importances = pd.read_csv("Eval/final_feature_importance.csv")
###


def process_data_for_training(
    df,
    drop_pre_op_delirium=True,
    drop_below_age_60=False,
    delirium_cutoff=2,
    standardize=True,
):
    """

    :param df:
    :return:
    """
    ### First we start by dropping columns that are not useful or should not be used for training
    # drop all the columns with HV or DP in the name as they have post-op information
    df = df.copy()
    hv_columns = [col for col in df.columns if "HV" in col]
    df.drop(hv_columns, axis=1, inplace=True)
    dp_columns = [col for col in df.columns if "DP" in col]
    df.drop(dp_columns, axis=1, inplace=True)
    # drop all the columns with post-op in the name
    post_op_columns = [col for col in df.columns if "post-op" in col]
    df.drop(post_op_columns, axis=1, inplace=True)
    # drop column containing length_of_patient in the name as they encode the length of stay which is only known after
    length_of_patient_columns = [
        col for col in df.columns if "length_of_patient" in col
    ]
    df.drop(length_of_patient_columns, axis=1, inplace=True)
    # Drop surgical time columns, as they are not known before surgery
    surgical_time_columns = [col for col in df.columns if "surgical_time" in col]
    df.drop(surgical_time_columns, axis=1, inplace=True)

    ### Extract the target columns and drop all other AP columns, as AP columns are only known after surgery. Drop  other delirium columns as well
    # Get the target columns
    target_substrings = [
        "_disorientation",
        "_inappropriate_behavior",
        "_inappropriate_communication",
        "illusions/hallucinations",
        "_psychomotor_retardation",
    ]
    target_cols = [
        col
        for col in df.columns
        if any(sub in col.lower() for sub in target_substrings)
    ]
    # We want the delirium AP columns to be the target columns
    target_cols_ap = [col for col in target_cols if "AP" in col]

    # Drop other AP cols
    ap_columns = [
        col for col in df.columns if ("AP" in col and col not in target_cols_ap)
    ]
    df.drop(ap_columns, axis=1, inplace=True)

    # Drop pre-op delirium cols
    pre_op_delirium_cols = [col for col in target_cols if "BI" in col]
    if drop_pre_op_delirium:
        # Drop all the datapoints that have pre-op delirium
        # Sum the pre-op delirium columns to get a single column
        df["pre_op_delirium"] = df[pre_op_delirium_cols].sum(axis=1)
        df = df.loc[df["pre_op_delirium"] < delirium_cutoff]
        # Drop the pre-op delirium col again
        df.drop("pre_op_delirium", axis=1, inplace=True)

    df.drop(pre_op_delirium_cols, axis=1, inplace=True)

    ### Drop columns that are not useful for training
    # Can drop elective column since it is a function of acute
    df.drop("(5:_setting)_Elective", axis=1, inplace=True)
    # Can drop these anesthesia columns since they are not useful in most cases
    anes_to_drop = [
        "6:_anaesthesia_Upper extremity",
        "6:_anaesthesia_Lower extremity",
        "6:_anaesthesia_Epidural",
        "6:_anaesthesia_SEVO",
        "6:_anaesthesia_Intraop. nociception monitored",
    ]
    anes_cols_to_drop = [
        col for col in df.columns if any(sub in col for sub in anes_to_drop)
    ]
    df.drop(anes_cols_to_drop, axis=1, inplace=True)

    ### Drop rows that have age <60 if drop_below_age_60 is True
    if drop_below_age_60:
        # Drop rows that have age <60
        df = df[df["age"] >= 60]

    y_df = df[target_cols_ap]
    x_df = df.drop(columns=target_cols_ap)

    # Standardization stats
    sstats = {}
    # Standardize the data
    if standardize:
        # Standardize the columns that are not 0-1
        for col in x_df.columns:
            # Check if col values are not 0-1
            if x_df[col].min() < 0 or x_df[col].max() > 1:
                print(f"Standardizing column: {col}")
                sstats[col] = (x_df[col].mean(), x_df[col].std())
                x_df[col] = (x_df[col] - x_df[col].mean()) / x_df[col].std()

    pd.DataFrame(sstats).to_csv("standardization_stats.csv")
    y_reduced_df = y_df.sum(axis=1)
    y_binary_df = y_reduced_df.apply(lambda x: 1 if x >= delirium_cutoff else 0)
    # Make a dataset with 4 classes: 0->0, 1->1, 2->2, 3+->3
    y_4_classes_df = y_reduced_df.apply(
        lambda x: 3 if x >= 3 else x
    )  # 3+ deliriums are all in class 3
    print(x_df.columns)
    # Check if there are strong correlations between features of x_df
    corr_matrix = x_df.corr().abs()
    # Check what combinations of columns with "6" in the name exist
    six_cols = [col for col in x_df.columns if "6:_an" in col]
    # Group the columns with "6" in the name by their values
    six_cols_grouped = x_df[six_cols].groupby(six_cols)
    # Get count of each group
    six_cols_grouped_count = six_cols_grouped.size()
    # Extract 15 most common groups
    six_cols_grouped_count = six_cols_grouped_count.sort_values(ascending=False)
    six_cols_grouped_count = six_cols_grouped_count.head(15)
    # Print the most common groups
    print(corr_matrix)

    # Print how many patients there are in the final dataset and how many of those have delirium
    print(f"Number of patients: {len(x_df)}")
    print(f"Number of patients with delirium: {len(y_binary_df[y_binary_df == 1])}")
    return x_df, y_df, y_reduced_df, y_binary_df, y_4_classes_df


def recursive_feature_elimination(
    X_train,
    y_train,
    n_features_to_select=10,
    random_state=42,
    model_type="rf",
    oversample=False,
):
    if model_type == "rf":
        # Initialize the Random Forest model
        est = RandomForestClassifier(
            n_estimators=100, random_state=random_state, max_depth=2
        )
    elif model_type == "logistic":
        est = LogisticRegression(penalty="l1", C=1, solver="liblinear", max_iter=1000)

    if oversample:
        # Use SMOTE to oversample the minority class
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Initialize RFE (Recursive Feature Elimination) with the Random Forest model
    # The number of features to select can be set to any number, or left to choose all
    rfe = RFE(
        estimator=est, n_features_to_select=n_features_to_select, step=1
    )  # You can change n_features_to_select

    # Fit RFE
    rfe.fit(X_train, y_train)

    # Check selected features
    selected_features = rfe.support_  # Boolean mask of selected features
    ranking = rfe.ranking_  # Ranking of features (1 means selected)

    # Optionally, you can print or analyze these:
    selected_features = X_train.columns[selected_features]
    ranking = [(name, int(rank)) for name, rank in zip(X_train.columns, ranking)]
    print("Selected Features (Boolean Mask):", selected_features)
    print("Feature Ranking:", ranking)

    # Include back the groups (if one of the features in the group is selected, all are selected)
    for group, features in FEATURE_GROUPS.items():
        if any(feature in selected_features for feature in features):
            selected_features = selected_features.union(features)

    return selected_features


import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score


def fit_xgb_model(X_train, y_train, random_state=42):
    """
    Fits an XGBoost model using cross-validation to tune hyperparameters.

    Args:
        X_train (np.array or pd.DataFrame): Training features.
        y_train (np.array or pd.Series): Training labels.
        random_state (int): Random seed for reproducibility.

    Returns:
        xgb.XGBClassifier: Best XGBoost model found by GridSearchCV.
    """
    # Determine if the task is binary or multi-class classification
    num_classes = len(np.unique(y_train))
    if num_classes > 2:
        objective = "multi:softprob"
        num_class = num_classes
    else:
        objective = "binary:logistic"
        num_class = None
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Define the XGBoost classifier
    model = xgb.XGBClassifier(
        objective=objective,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        num_class=num_class,
        scale_pos_weight=scale_pos_weight if num_classes == 2 else None,
    )

    # Define the parameter grid for tuning
    param_grid = {
        "max_depth": [1, 2],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50],
        "min_child_weight": [1, 2],
        "gamma": [0.1, 0.3],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_alpha": [0.1, 1],
        "reg_lambda": [0.1, 1],
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1,
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score (F1):", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_


def fit_logistic_regression_model(X_train, y_train, oversampling=False):
    """
    Fit a logistic regression model using cross-validation with F1-score as the evaluation metric.

    Parameters:
    - X_train: Feature matrix for training.
    - y_train: Labels for training.
    - oversampling: Whether to use SMOTE for oversampling the minority class.

    Returns:
    - Best model found via GridSearchCV.
    """
    # Define the parameter grid
    param_grid = {
        "logisticregression__C": [0.01, 0.1, 1, 10],  # Regularization strength
        "logisticregression__penalty": ["l1", "l2"],  # Regularization types
        "logisticregression__solver": ["liblinear", "saga"],  # Compatible solvers
    }

    # Initialize SMOTE (if oversampling is enabled)
    if oversampling:
        smote = SMOTE(sampling_strategy=0.5, random_state=42)

    # Create a pipeline with SMOTE (if applicable) and logistic regression
    pipeline = Pipeline(
        [
            (
                "oversampling",
                smote if oversampling else "passthrough",
            ),  # SMOTE or no-op
            ("logisticregression", LogisticRegression(max_iter=1000)),
        ]
    )

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",  # Use F1-score as the evaluation metric
        cv=5,  # 5-fold cross-validation
        verbose=1,  # To show progress
        n_jobs=-1,  # Use all processors for parallel processing
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score (F1):", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_.steps[-1][1]


def fit_neural_network_model(X_train, y_train, random_state=42):
    """
    Fits an MLP neural network using cross-validation with SMOTE for oversampling.

    Args:
        X_train (np.array or pd.DataFrame): Training features.
        y_train (np.array or pd.Series): Training labels.
        random_state (int): Random seed for reproducibility.

    Returns:
        Best MLPClassifier model found by GridSearchCV.
    """
    # Define SMOTE for oversampling
    smote = SMOTE(sampling_strategy=0.5, random_state=random_state)

    # Define the MLPClassifier
    model = MLPClassifier(
        max_iter=500,  # Allow more iterations for convergence
        random_state=random_state,
        early_stopping=True,  # Stops training if validation score doesn't improve
    )

    # Define the parameter grid
    param_grid = {
        "mlpclassifier__hidden_layer_sizes": [(100,), (300, 100)],
        "mlpclassifier__activation": ["logistic", "relu"],
        "mlpclassifier__alpha": [0.0001, 0.001],  # Regularization strength
        "mlpclassifier__learning_rate": ["adaptive"],
        "mlpclassifier__learning_rate_init": [0.001, 0.01],
        "mlpclassifier__solver": ["adam"],  # 'adam' works well for small datasets
    }

    # Create a pipeline with SMOTE and MLPClassifier
    pipeline = Pipeline([("oversampling", smote), ("mlpclassifier", model)])

    # Define a scoring function (F1-score in this case)
    f1_scorer = make_scorer(
        f1_score, average="weighted"
    )  # Use 'binary' for binary classification

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=f1_scorer,  # Use F1-score as the evaluation metric
        cv=5,  # 5-fold cross-validation
        verbose=1,  # To show progress
        n_jobs=-1,  # Use all processors for parallel processing
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score (F1):", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_.steps[-1][1]


def fit_svm_model(X_train, y_train):
    """
    Fit an SVM model using cross-validation to find the optimal parameters.

    Parameters:
    - X_train: Feature matrix for training.
    - y_train: Labels for training.

    Returns:
    - Best model found via GridSearchCV.
    """
    # Define the parameter grid
    param_grid = {
        "C": [0.01, 0.05, 0.1, 0.2, 0.5, 0.7],
        "kernel": ["linear"],
        # "degree": [2, 3],  # Only applicable for 'poly' kernel
        "class_weight": [None, "balanced"],
    }

    # Initialize the SVM model
    model = SVC(probability=True)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",  # You can change this to another metric if needed
        cv=5,  # 5-fold cross-validation
        verbose=1,  # To show progress
        n_jobs=-1,  # Use all processors for parallel processing
    )

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Return the best model
    return grid_search.best_estimator_


def evaluate_model(
    model, X_test, y_test, plot_roc=False, threshold=0.5, plot_precision_recall=True
):
    # Make predictions
    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    # Calculate AUC-ROC
    auc_score = roc_auc_score(y_test, y_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    # Calculate AU-PRC
    auc_pr = average_precision_score(y_test, y_prob)

    # Collect classification report variables
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    class_0_f1 = report["0"]["f1-score"]
    class_1_f1 = report["1"]["f1-score"]
    class_0_recall = report["0"]["recall"]
    class_1_recall = report["1"]["recall"]
    class_0_precision = report["0"]["precision"]
    class_1_precision = report["1"]["precision"]
    accuracy = report["accuracy"]

    if plot_roc:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC: {auc_score:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    if plot_precision_recall:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f"AUC-PR: {auc_pr:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.show()

    return {
        "auc": auc_score,
        "auc_pr": auc_pr,
        "NoDelirium_f1": class_0_f1,
        "Delirium_f1": class_1_f1,
        "NoDelirium_recall": class_0_recall,
        "Delirium_recall": class_1_recall,
        "NoDelirium_precision": class_0_precision,
        "Delirium_precision": class_1_precision,
        "accuracy": accuracy,
    }


def run_pipeline(
    df,
    drop_pre_op_delirium=True,
    drop_below_age_60=False,
    delirium_cutoff=2,
    perform_recursive_feature_elimination=True,
    only_one_recursive_feature_elimination=False,
    seeds=[42, 69, 420, 69420, 6942069, 42069, 420420],
    rfe_num_features=10,
    rfe_model_type="rf",
    model_type="logistic",
    log_and_square_features=False,
    poly_features=0,
    verbose=True,
    oversample=False,
    threshold=0.5,
    standardize_data=True,
    keep_features=1000,
):
    random_seed = seeds[0]
    # Copy the dataframe to avoid modifying the original
    df = df.copy()
    # Process data for training
    x_df, _, _, y_binary_df, y_multiclass = process_data_for_training(
        df,
        drop_pre_op_delirium=drop_pre_op_delirium,
        drop_below_age_60=drop_below_age_60,
        delirium_cutoff=delirium_cutoff,
        standardize=standardize_data,
    )
    # Keep only the top keep_features features
    cols_to_keep = feature_importances["feature"].head(keep_features)
    x_df = x_df[cols_to_keep]

    # Split data into train and test sets for RFE
    X_train, X_test, y_train, y_test = train_test_split(
        x_df, y_binary_df, test_size=0.2, random_state=random_seed
    )
    if only_one_recursive_feature_elimination and perform_recursive_feature_elimination:
        selected_features = recursive_feature_elimination(
            X_train,
            y_train,
            n_features_to_select=rfe_num_features,
            random_state=random_seed,
            model_type=rfe_model_type,
        )

    models = []
    eval_results_list = []
    feature_importance_list = []
    for seed in seeds:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            x_df, y_binary_df, test_size=0.2, random_state=seed
        )
        print(f"\n\nRunning pipeline with random seed: {seed}")
        # RFE
        if (
            not only_one_recursive_feature_elimination
            and perform_recursive_feature_elimination
        ):
            selected_features = recursive_feature_elimination(
                X_train,
                y_train,
                n_features_to_select=rfe_num_features,
                random_state=seed,
                model_type=rfe_model_type,
            )
        if perform_recursive_feature_elimination:
            X_train = X_train[selected_features]
            X_test = X_test[selected_features]
        if log_and_square_features:
            # Add log and square features
            X_train = X_train.copy()
            X_test = X_test.copy()
            for col in X_train.columns:
                X_train[f"log_{col}"] = np.log1p(X_train[col])
                X_train[f"square_{col}"] = X_train[col] ** 2
                X_test[f"log_{col}"] = np.log1p(X_test[col])
                X_test[f"square_{col}"] = X_test[col] ** 2
        if poly_features > 0:
            # Add polynomial features and convert back to DataFrame
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(degree=poly_features, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            X_train = pd.DataFrame(X_train_poly)
            X_test = pd.DataFrame(X_test_poly)

        importance_df = None
        if model_type == "xgb":
            model = fit_xgb_model(X_train, y_train, random_state=seed)
            # Calibrate the model
            eval_calibration(model, X_test, y_test)
            model = calibrate_model(model, X_train, y_train)
            eval_calibration(model, X_test, y_test)
            eval_results = evaluate_model(model, X_test, y_test, threshold=threshold)
            # Feature importance
            importance_df = pd.DataFrame(
                {
                    "feature": X_train.columns,
                    "importance": model.estimator.feature_importances_,
                }
            )

        elif model_type == "logistic":
            model = fit_logistic_regression_model(
                X_train, y_train, oversampling=oversample
            )

            eval_results = evaluate_model(model, X_test, y_test, threshold=threshold)
            # Calculate feature importance
            feature_importance = np.abs(model.coef_[0])
            feature_importance = feature_importance / feature_importance.sum()
            importance_df = pd.DataFrame(
                {"feature": X_train.columns, "importance": feature_importance}
            )
            # Calculate sign of feature importance
            pd.DataFrame(np.sign(model.coef_[0]), index=X_train.columns).to_csv(
                "feature_signs.csv"
            )


        elif model_type == "neural_network":
            model = fit_neural_network_model(X_train, y_train)
            eval_results = evaluate_model(model, X_test, y_test, threshold=threshold)
        elif model_type == "svm":
            model = fit_svm_model(X_train, y_train)
            eval_calibration(model, X_test, y_test)
            # Calibrate the model
            model = calibrate_model(model, X_train, y_train)
            eval_calibration(model, X_test, y_test)
            eval_results = evaluate_model(model, X_test, y_test, threshold=threshold)
            # Calculate feature importance
            try:
                feature_importance = np.abs(model.coef_[0])
                feature_importance = feature_importance / feature_importance.sum()
                importance_df = pd.DataFrame(
                    {"feature": X_train.columns, "importance": feature_importance}
                )
            except:
                importance_df = None
        elif model_type == "constant":
            # Use dummy model that always predicts 0
            model = DummyClassifier(strategy="constant", constant=1)
            model.fit(X_train, y_train)
            eval_results = evaluate_model(model, X_test, y_test, threshold=threshold)

        feature_importance_list.append(importance_df)
        models.append(model)
        eval_results_list.append(eval_results)

    if verbose:
        auc_list = [eval_results["auc"] for eval_results in eval_results_list]
        auc_pr_list = [eval_results["auc_pr"] for eval_results in eval_results_list]
        f1_non_delirium_list = [
            eval_results["NoDelirium_f1"] for eval_results in eval_results_list
        ]
        f1_delirium_list = [
            eval_results["Delirium_f1"] for eval_results in eval_results_list
        ]
        recall_non_delirium_list = [
            eval_results["NoDelirium_recall"] for eval_results in eval_results_list
        ]
        recall_delirium_list = [
            eval_results["Delirium_recall"] for eval_results in eval_results_list
        ]
        precision_non_delirium_list = [
            eval_results["NoDelirium_precision"] for eval_results in eval_results_list
        ]
        precision_delirium_list = [
            eval_results["Delirium_precision"] for eval_results in eval_results_list
        ]
        accuracy_list = [eval_results["accuracy"] for eval_results in eval_results_list]
        # Use +- symbol sign for std
        print(f"Mean AUC: {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
        print(f"Mean AUC-PR: {np.mean(auc_pr_list):.3f} ± {np.std(auc_pr_list):.3f}")
        print(
            f"Mean F1 Score (Non-Delirium): {np.mean(f1_non_delirium_list):.3f} ± {np.std(f1_non_delirium_list):.3f}"
        )
        print(
            f"Mean F1 Score (Delirium): {np.mean(f1_delirium_list):.3f} ± {np.std(f1_delirium_list):.3f}"
        )
        print(
            f"Mean Recall (Non-Delirium): {np.mean(recall_non_delirium_list):.3f} ± {np.std(recall_non_delirium_list):.3f}"
        )
        print(
            f"Mean Precision (Non-Delirium): {np.mean(precision_non_delirium_list):.3f} ± {np.std(precision_non_delirium_list):.3f}"
        )
        print(
            f"Mean Recall (Delirium): {np.mean(recall_delirium_list):.3f} ± {np.std(recall_delirium_list):.3f}"
        )
        print(
            f"Mean Precision (Delirium): {np.mean(precision_delirium_list):.3f} ± {np.std(precision_delirium_list):.3f}"
        )
        print(
            f"Mean Accuracy: {np.mean(accuracy_list):.3f} ± {np.std(accuracy_list):.3f}"
        )

        # Save model to disk
        import joblib

        joblib.dump(models[0], f"model_{model_type}.pkl")

    # Average feature importance
    feature_importance_df = None
    if model_type == "xgb" or model_type == "logistic" or model_type == "svm":
        try:
            feature_importance_df = (
                pd.concat(feature_importance_list)
                .groupby("feature")
                .mean()
                .reset_index()
            )
            feature_importance_df = feature_importance_df.sort_values(
                "importance", ascending=False
            )
            # Print feature importance
            print(feature_importance_df.head(20))
        except:
            print("Feature importance could not be calculated")

    return models, eval_results_list, feature_importance_df

def eval_calibration(model, X_test, y_test):
    # Print calibration curve
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    prob_pos = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    plt.figure(figsize=(6, 6))

    plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Calibration Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

    plt.xlabel("Mean Predicted Value")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()

    plt.show()

def calibrate_model(base_model, X_calib, y_calib, calibration_method="sigmoid"):
    """
    Calibrates a model using Platt scaling.

    Args:
        model: The model to be calibrated.
        X_calib: Calibration features.
        y_calib: Calibration labels.

    Returns:
        Calibrated model.
    """
    # Calibrate the model using Platt scaling
    calibrated_model = CalibratedClassifierCV(
        base_model, method=calibration_method, cv="prefit"
    )
    calibrated_model.fit(X_calib, y_calib)

    # Evaluate calibration
    y_pred = calibrated_model.predict_proba(X_calib)[:, 1]
    brier_score = brier_score_loss(y_calib, y_pred)
    print("Brier Score (lower is better):", brier_score)
    return calibrated_model





def train_ensemble_model(df, random_state=42):
    """
    Trains an ensemble model using a combination of logistic regression, XGBoost, and SVM.

    Args:
        df (pd.DataFrame): Input dataframe
        random_state (int): Random seed for reproducibility.

    Returns:
        EnsembleModel: Trained ensemble model with predict and predict_proba methods
    """
    # Copy the dataframe to avoid modifying the original
    df = df.copy()

    # Process data for training
    x_df, _, _, y_binary_df, y_multiclass = process_data_for_training(
        df,
        drop_pre_op_delirium=True,
        drop_below_age_60=True,
        delirium_cutoff=2,
        standardize=True,
    )

    cols_to_keep = [
        "6:_anaesthesia_Spinal",
        "6:_anaesthesia_General anaesthesia",
        "27:_laryngeal_mask_BI_Tube",
        "6:_anaesthesia_EEG used (is protective)",
        "4:_surgical_speciality_Gyn/Obs",
        "6:_anaesthesia_Cont. Temp. Monitoring",
        "25:_noise_reduction_performed_BI",
        "21: patient_uses_dentures_BI",
        "16:_general_well-being_nrs _0-10_BI",
        "19:_patient_uses_visual_aids_BI",
    ]
    cols_to_keep = [
        col for col in x_df.columns if any(sub in col for sub in cols_to_keep)
    ]
    x_df = x_df[cols_to_keep]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_df, y_binary_df, test_size=0.2, random_state=random_state
    )

    # Split train set into train and calibration sets
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state
    )

    # Define the models
    logistic_model = fit_logistic_regression_model(X_train, y_train, oversampling=True)
    xgb_model = fit_xgb_model(X_train, y_train, random_state=random_state)
    svm_model = fit_svm_model(X_train, y_train)

    # Calibrate the models

    logistic_model = calibrate_model(logistic_model, X_calib, y_calib)
    xgb_model = calibrate_model(
        xgb_model, X_calib, y_calib, calibration_method="isotonic"
    )
    svm_model = calibrate_model(svm_model, X_calib, y_calib)

    # Create an ensemble model
    ensemble_model = EnsembleModel(
        models=[logistic_model, xgb_model, svm_model],
        # Optional: you can adjust weights if needed
        weights=[0.33, 0.33, 0.33],
    )

    # Evaluate the ensemble model
    eval_results = evaluate_model(ensemble_model, X_test, y_test, threshold=0.2)

    return ensemble_model, eval_results


if __name__ == "__main__":
    if False:
        auc_list = []
        for i in range(55):
            models, eval_results_list, imp = run_pipeline(
                df_used,
                drop_pre_op_delirium=True,
                drop_below_age_60=True,
                perform_recursive_feature_elimination=False,
                only_one_recursive_feature_elimination=True,
                rfe_num_features=5,
                model_type="svm",
                log_and_square_features=False,
                poly_features=0,
                rfe_model_type="logistic",
                oversample=False,
                threshold=0.2,
                standardize_data=True,
                keep_features=i + 1,
            )
            auc_list.append(
                np.mean([eval_results["auc"] for eval_results in eval_results_list])
            )
        # Save auc list
        auc_df = pd.DataFrame(auc_list, columns=["auc"])
        auc_df.to_csv("auc_list_svm.csv", index=False)

    if True:
        models, eval_results_list, imp = run_pipeline(
            df_used,
            drop_pre_op_delirium=True,
            drop_below_age_60=True,
            perform_recursive_feature_elimination=False,
            only_one_recursive_feature_elimination=True,
            rfe_num_features=5,
            model_type="xgb",
            log_and_square_features=False,
            poly_features=0,
            rfe_model_type="constant",
            oversample=False,
            threshold=0.1,
            standardize_data=True,
            keep_features=14,
        )
        # Save imp to disk
        # imp.to_csv("feature_importance_logistic_wsgn.csv", index=False)

    """
    models = []
    eval_results_list = []
    for seed in seeds:
        model, eval_results = train_ensemble_model(df_used, random_state=seed)
        models.append(model)
        eval_results_list.append(eval_results)
    print(
        f"Mean AUC: {np.mean([eval_results['auc'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['auc'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean AUC-PR: {np.mean([eval_results['auc_pr'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['auc_pr'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean F1 Score (Non-Delirium): {np.mean([eval_results['NoDelirium_f1'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['NoDelirium_f1'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean F1 Score (Delirium): {np.mean([eval_results['Delirium_f1'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['Delirium_f1'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean Recall (Non-Delirium): {np.mean([eval_results['NoDelirium_recall'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['NoDelirium_recall'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean Precision (Non-Deliriium): {np.mean([eval_results['NoDelirium_precision'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['NoDelirium_precision'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean Recall (Delirium): {np.mean([eval_results['Delirium_recall'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['Delirium_recall'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean Precision (Delirium): {np.mean([eval_results['Delirium_precision'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['Delirium_precision'] for eval_results in eval_results_list]):.3f}"
    )
    print(
        f"Mean Accuracy: {np.mean([eval_results['accuracy'] for eval_results in eval_results_list]):.3f} ± {np.std([eval_results['accuracy'] for eval_results in eval_results_list]):.3f}"
    )"""
