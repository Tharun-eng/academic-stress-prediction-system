# ============================================================
# ACADEMIC STRESS PREDICTION SYSTEM
# Final Production Version
# ============================================================

print("=" * 80)
print("ACADEMIC STRESS PREDICTION SYSTEM")
print("Professional Machine Learning Pipeline")
print("Final Production Version")
print("=" * 80)

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from imblearn.over_sampling import SMOTE

# ============================================================
# OPTIONAL : XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

    xgboost_available = True

    print("✓ XGBoost Installed")

except ImportError:

    xgboost_available = False

    print("✗ XGBoost Not Installed")
    print("Using Remaining Ensemble Models")

# ============================================================
# CREATE PROJECT DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# ============================================================
# CLEAN PREVIOUS OUTPUTS
# ============================================================

print("\nCleaning Previous Outputs...")

for folder in ["charts", "models"]:

    if os.path.exists(folder):

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            try:

                if os.path.isfile(path):
                    os.remove(path)

                elif os.path.isdir(path):
                    shutil.rmtree(path)

            except Exception:
                pass

print("✓ Previous Reports Removed")
print("✓ Previous Charts Removed")
print("✓ Previous Models Removed")

# ============================================================
# GLOBAL VARIABLES
# ============================================================

label_encoders = {}

trained_models = {}

results = []

feature_importance_df = None

best_model = None

best_model_name = ""

best_accuracy = 0

best_cv_score = 0

# ============================================================
# RANDOM STATE
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

print("\nProject Initialized Successfully")

print("=" * 80)
print("SECTION 1 COMPLETED SUCCESSFULLY")
print("=" * 80)

# ============================================================
# SECTION 2 : DATASET LOADING & EDA
# ============================================================

print("\n")
print("=" * 80)
print("SECTION 2 : DATASET LOADING & EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/student_stress_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("\n✓ Dataset Loaded Successfully")

# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset Shape")
print("-" * 40)
print(df.shape)

print("\nDataset Columns")
print("-" * 40)
print(df.columns.tolist())

print("\nData Types")
print("-" * 40)
print(df.dtypes)

print("\nFirst Five Records")
print("-" * 40)
print(df.head())

print("\nLast Five Records")
print("-" * 40)
print(df.tail())

# ============================================================
# STATISTICAL SUMMARY
# ============================================================

print("\nStatistical Summary")
print("-" * 40)
print(df.describe())

# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing Values")
print("-" * 40)
print(df.isnull().sum())

# ============================================================
# DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records")
print("-" * 40)
print(duplicates)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nStress Level Distribution")
print("-" * 40)
print(df["Stress_Level"].value_counts())

print("\nStress Level Percentage")
print("-" * 40)
print(round(df["Stress_Level"].value_counts(normalize=True) * 100, 2))

# ============================================================
# STRESS SCORE ANALYSIS (EDA ONLY)
# ============================================================

print("\nStress Score Statistics")
print("-" * 40)
print(df["Stress_Score"].describe())

print("\nNOTE")
print("-" * 40)
print("Stress_Score will NOT be used as an input feature.")
print("It is retained only for exploratory analysis.")

# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# remove Stress_Score from correlation inputs
numerical_columns = [
    col for col in numerical_columns
    if col != "Stress_Score"
]

print("\nNumerical Features")
print("-" * 40)

for column in numerical_columns:
    print(column)

# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df[numerical_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "charts/feature_correlation_heatmap.png",
    dpi=300
)

plt.close()

print("\n✓ feature_correlation_heatmap.png Saved")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,6))

df["Stress_Level"].value_counts().plot(
    kind="bar",
    color=["green","orange","red"]
)

plt.title("Stress Level Distribution")
plt.xlabel("Stress Level")
plt.ylabel("Students")

plt.tight_layout()

plt.savefig(
    "charts/class_distribution.png",
    dpi=300
)

plt.close()

print("✓ class_distribution.png Saved")

# ============================================================
# STRESS LEVEL PIE CHART
# ============================================================

plt.figure(figsize=(6,6))

df["Stress_Level"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")

plt.title("Stress Level Percentage")

plt.tight_layout()

plt.savefig(
    "charts/stress_distribution_pie.png",
    dpi=300
)

plt.close()

print("✓ stress_distribution_pie.png Saved")

# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(15,10))

df[numerical_columns].hist(
    bins=20,
    figsize=(15,10)
)

plt.tight_layout()

plt.savefig(
    "charts/feature_distribution.png",
    dpi=300
)

plt.close()

print("✓ feature_distribution.png Saved")

# ============================================================
# DATASET QUALITY CHECK
# ============================================================

print("\nDataset Quality Check")
print("-" * 40)

print(f"Rows                 : {df.shape[0]}")
print(f"Columns              : {df.shape[1]}")
print(f"Missing Values       : {df.isnull().sum().sum()}")
print(f"Duplicate Records    : {duplicates}")

print("\nDataset Ready for Preprocessing")

print("\n")
print("=" * 80)
print("SECTION 2 COMPLETED SUCCESSFULLY")
print("=" * 80)

# ============================================================
# SECTION 3 : DATA PREPROCESSING
# ============================================================

print("\n")
print("=" * 80)
print("SECTION 3 : DATA PREPROCESSING")
print("=" * 80)

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nHandling Missing Values...")

if df.isnull().sum().sum() > 0:

    for column in df.columns:

        if df[column].dtype == "object":
            df[column].fillna(df[column].mode()[0], inplace=True)

        else:
            df[column].fillna(df[column].median(), inplace=True)

print("✓ Missing Values Handled")

# ============================================================
# DROP STRESS_SCORE
# ============================================================

print("\nRemoving Stress_Score from Training Features...")

if "Stress_Score" in df.columns:
    df = df.drop(columns=["Stress_Score"])

print("✓ Stress_Score Removed")

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

print("\nEncoding Categorical Features...")

categorical_columns = df.select_dtypes(include="object").columns.tolist()

categorical_columns.remove("Stress_Level")

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

    print(f"{column:<25} Encoded")

# ============================================================
# ENCODE TARGET VARIABLE
# ============================================================

target_encoder = LabelEncoder()

df["Stress_Level"] = target_encoder.fit_transform(df["Stress_Level"])

label_encoders["Stress_Level"] = target_encoder

print("\n✓ Target Variable Encoded")

print("\nTarget Mapping")

for label, value in zip(
    target_encoder.classes_,
    target_encoder.transform(target_encoder.classes_)
):

    print(f"{label:<10} -> {value}")

# ============================================================
# FEATURES & TARGET
# ============================================================

print("\nPreparing Features and Target...")

X = df.drop(columns=["Stress_Level"])

y = df["Stress_Level"]

feature_names = X.columns.tolist()

print(f"\nFeature Shape : {X.shape}")
print(f"Target Shape  : {y.shape}")

print("\nFeature Names")

for feature in feature_names:
    print("-", feature)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\nCreating Train/Test Split...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=RANDOM_STATE

)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ============================================================
# FEATURE SCALING
# ============================================================

print("\nApplying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("✓ Feature Scaling Completed")

# ============================================================
# APPLY SMOTE
# ============================================================

print("\nApplying SMOTE (Training Data Only)...")

smote = SMOTE(random_state=RANDOM_STATE)

X_train_balanced, y_train_balanced = smote.fit_resample(

    X_train,

    y_train

)

print("✓ SMOTE Applied Successfully")

print("\nBalanced Class Distribution")

print(pd.Series(y_train_balanced).value_counts())

# ============================================================
# CROSS VALIDATION
# ============================================================

print("\nPreparing Stratified K-Fold...")

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)

print("✓ Stratified 5-Fold Ready")

# ============================================================
# SAVE PREPROCESSING OBJECTS
# ============================================================

print("\nSaving Preprocessing Objects...")

joblib.dump(

    scaler,

    "models/scaler.pkl"

)

joblib.dump(

    label_encoders,

    "models/label_encoders.pkl"

)

joblib.dump(

    feature_names,

    "models/features.pkl"

)

print("✓ scaler.pkl Saved")

print("✓ label_encoders.pkl Saved")

print("✓ features.pkl Saved")

# ============================================================
# RESULTS CONTAINERS
# ============================================================

results = []

trained_models = {}

print("\nResults Containers Initialized")

print("\n")
print("=" * 80)
print("SECTION 3 COMPLETED SUCCESSFULLY")
print("=" * 80)

# ============================================================
# SECTION 4 : MODEL TRAINING
# ============================================================

print("\n")
print("="*80)
print("SECTION 4 : MODEL TRAINING")
print("="*80)

from xgboost import XGBClassifier

# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

}

if xgboost_available:

    models["XGBoost"] = XGBClassifier(

        n_estimators=200,

        learning_rate=0.10,

        max_depth=5,

        subsample=0.80,

        colsample_bytree=0.80,

        eval_metric="mlogloss",

        random_state=RANDOM_STATE

    )

print("\nModels Loaded Successfully")

# ============================================================
# TRAIN EACH MODEL
# ============================================================

for model_name, model in models.items():

    print("\n" + "="*60)
    print(f"Training : {model_name}")
    print("="*60)

    # ---------------------------------------
    # Logistic Regression
    # ---------------------------------------

    if model_name == "Logistic Regression":

        model.fit(

            X_train_scaled,

            y_train

        )

        prediction = model.predict(

            X_test_scaled

        )

        cv_score = cross_val_score(

            model,

            X_train_scaled,

            y_train,

            cv=skf,

            scoring="accuracy"

        ).mean()

    # ---------------------------------------
    # Ensemble Models
    # ---------------------------------------

    else:

        model.fit(

            X_train_balanced,

            y_train_balanced

        )

        prediction = model.predict(

            X_test

        )

        cv_score = cross_val_score(

            model,

            X_train_balanced,

            y_train_balanced,

            cv=skf,

            scoring="accuracy"

        ).mean()

    # ============================================================
    # EVALUATION METRICS
    # ============================================================

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    trained_models[model_name] = model

    joblib.dump(
        model,
        f"models/{model_name.lower().replace(' ','_')}.pkl"
    )

    # ============================================================
    # STORE RESULTS
    # ============================================================

    results.append({

        "Model": model_name,

        "Accuracy": round(accuracy,4),

        "Precision": round(precision,4),

        "Recall": round(recall,4),

        "F1 Score": round(f1,4),

        "Cross Validation": round(cv_score,4)

    })

    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Cross Validation : {cv_score:.4f}")

# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\n")
print("="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

print(results_df)

# ============================================================
# BEST ENSEMBLE MODEL
# ============================================================

ensemble_results = results_df[
    results_df["Model"] != "Logistic Regression"
]

best_row = ensemble_results.iloc[0]

best_model_name = best_row["Model"]

best_model = trained_models[best_model_name]

best_accuracy = best_row["Accuracy"]

best_cv_score = best_row["Cross Validation"]

print("\n")
print("="*80)
print("BEST DEPLOYMENT MODEL")
print("="*80)

print(f"Model              : {best_model_name}")
print(f"Accuracy           : {best_accuracy:.4f}")
print(f"Cross Validation   : {best_cv_score:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("-"*60)

if best_model_name == "Logistic Regression":

    final_prediction = best_model.predict(X_test_scaled)

else:

    final_prediction = best_model.predict(X_test)

print(classification_report(
    y_test,
    final_prediction,
    target_names=target_encoder.classes_,
    zero_division=0
))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    final_prediction
)

print("\nConfusion Matrix")
print("-"*60)

print(cm)

confusion_matrix_df = pd.DataFrame(
    cm,
    index=target_encoder.classes_,
    columns=target_encoder.classes_
)

confusion_matrix_df.to_csv(
    "models/confusion_matrix.csv"
)

# ============================================================
# SAVE DEPLOYMENT MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/stress_model.pkl"
)

print("\n✓ stress_model.pkl Saved")
print("✓ model_comparison.csv Saved")
print("✓ confusion_matrix.csv Saved")

print("\n")
print("="*80)
print("SECTION 4 COMPLETED SUCCESSFULLY")
print("="*80)

# ============================================================
# SECTION 5 : HYPERPARAMETER TUNING
# ============================================================

print("\n")
print("="*80)
print("SECTION 5 : HYPERPARAMETER TUNING")
print("="*80)

# ============================================================
# RANDOM FOREST TUNING
# ============================================================

print("\nOptimizing Random Forest...")

rf_parameters = {

    "n_estimators":[100,200,300],

    "max_depth":[10,20,30,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4]

}

rf_search = RandomizedSearchCV(

    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE
    ),

    param_distributions=rf_parameters,

    n_iter=10,

    cv=5,

    scoring="accuracy",

    random_state=RANDOM_STATE,

    n_jobs=-1

)

rf_search.fit(

    X_train_balanced,

    y_train_balanced

)

best_rf = rf_search.best_estimator_

print("✓ Random Forest Optimized")

# ============================================================
# EXTRA TREES TUNING
# ============================================================

print("\nOptimizing Extra Trees...")

et_parameters = {

    "n_estimators":[100,200,300],

    "max_depth":[10,20,30,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4]

}

et_search = RandomizedSearchCV(

    estimator=ExtraTreesClassifier(
        random_state=RANDOM_STATE
    ),

    param_distributions=et_parameters,

    n_iter=10,

    cv=5,

    scoring="accuracy",

    random_state=RANDOM_STATE,

    n_jobs=-1

)

et_search.fit(

    X_train_balanced,

    y_train_balanced

)

best_et = et_search.best_estimator_

print("✓ Extra Trees Optimized")

# ============================================================
# XGBOOST TUNING
# ============================================================

if xgboost_available:

    print("\nOptimizing XGBoost...")

    xgb_parameters={

        "n_estimators":[100,200,300],

        "learning_rate":[0.05,0.1,0.2],

        "max_depth":[3,5,7],

        "subsample":[0.8,1.0]

    }

    xgb_search = RandomizedSearchCV(

        estimator=XGBClassifier(

            eval_metric="mlogloss",

            random_state=RANDOM_STATE

        ),

        param_distributions=xgb_parameters,

        n_iter=10,

        cv=5,

        scoring="accuracy",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    xgb_search.fit(

        X_train_balanced,

        y_train_balanced

    )

    best_xgb = xgb_search.best_estimator_

    print("✓ XGBoost Optimized")

print("\n")
print("="*80)
print("SECTION 5A COMPLETED SUCCESSFULLY")
print("="*80)


# ============================================================
# SECTION 5B : FINAL MODEL EVALUATION
# ============================================================

print("\n")
print("="*80)
print("SECTION 5B : FINAL MODEL EVALUATION")
print("="*80)

optimized_results = []

optimized_models = {

    "Random Forest": best_rf,

    "Extra Trees": best_et

}

if xgboost_available:
    optimized_models["XGBoost"] = best_xgb

# ============================================================
# EVALUATE OPTIMIZED MODELS
# ============================================================

for model_name, model in optimized_models.items():

    print("\n" + "="*60)
    print(f"Evaluating : {model_name}")
    print("="*60)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    cv_score = cross_val_score(

        model,

        X_train_balanced,

        y_train_balanced,

        cv=skf,

        scoring="accuracy"

    ).mean()

    optimized_results.append({

        "Model": model_name,

        "Accuracy": round(accuracy,4),

        "Precision": round(precision,4),

        "Recall": round(recall,4),

        "F1 Score": round(f1,4),

        "Cross Validation": round(cv_score,4)

    })

    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"F1 Score         : {f1:.4f}")
    print(f"Cross Validation : {cv_score:.4f}")

# ============================================================
# CREATE RESULTS TABLE
# ============================================================

optimized_results_df = pd.DataFrame(optimized_results)

optimized_results_df = optimized_results_df.sort_values(

    by="Accuracy",

    ascending=False

).reset_index(drop=True)

print("\n")
print("="*80)
print("OPTIMIZED MODEL COMPARISON")
print("="*80)

print(optimized_results_df)

optimized_results_df.to_csv(

    "models/optimized_model_comparison.csv",

    index=False

)

# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = optimized_results_df.iloc[0]["Model"]

deployment_model = optimized_models[best_model_name]

print("\n")
print("="*80)
print("FINAL DEPLOYMENT MODEL")
print("="*80)

print(f"Selected Model : {best_model_name}")

# ============================================================
# SAVE DEPLOYMENT MODEL
# ============================================================

joblib.dump(

    deployment_model,

    "models/stress_model.pkl"

)

print("\n✓ stress_model.pkl Saved")

# ============================================================
# SAVE FINAL REPORT
# ============================================================

with open("models/training_report.txt","w") as file:

    file.write("ACADEMIC STRESS PREDICTION SYSTEM\n")

    file.write("="*60)

    file.write("\n\nFinal Deployment Model\n")

    file.write("---------------------------\n")

    file.write(f"Model : {best_model_name}\n\n")

    file.write(optimized_results_df.to_string(index=False))

print("✓ training_report.txt Saved")

print("\n")
print("="*80)
print("SECTION 5 COMPLETED SUCCESSFULLY")
print("="*80)


# ============================================================
# SECTION 6 : FINAL MODEL EVALUATION
# ============================================================

print("\n")
print("=" * 80)
print("SECTION 6 : FINAL MODEL EVALUATION")
print("=" * 80)

# ============================================================
# FINAL PREDICTION
# ============================================================

print("\nEvaluating Final Deployment Model...")

# Logistic Regression requires scaled features
if best_model_name == "Logistic Regression":

    final_prediction = deployment_model.predict(X_test_scaled)

else:

    final_prediction = deployment_model.predict(X_test)

# ============================================================
# FINAL METRICS
# ============================================================

final_accuracy = accuracy_score(y_test, final_prediction)

final_precision = precision_score(
    y_test,
    final_prediction,
    average="weighted",
    zero_division=0
)

final_recall = recall_score(
    y_test,
    final_prediction,
    average="weighted",
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    final_prediction,
    average="weighted",
    zero_division=0
)

print("\n")
print("=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print(f"Model              : {best_model_name}")
print(f"Accuracy           : {final_accuracy:.4f}")
print(f"Precision          : {final_precision:.4f}")
print(f"Recall             : {final_recall:.4f}")
print(f"F1 Score           : {final_f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    final_prediction,

    target_names=target_encoder.classes_,

    zero_division=0

)

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)

with open(
    "models/classification_report.txt",
    "w"
) as file:

    file.write(report)

print("\n✓ classification_report.txt Saved")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    final_prediction

)

cm_df = pd.DataFrame(

    cm,

    index=target_encoder.classes_,

    columns=target_encoder.classes_

)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm_df)

cm_df.to_csv(

    "models/confusion_matrix.csv"

)

print("\n✓ confusion_matrix.csv Saved")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(deployment_model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": deployment_model.feature_importances_

    })

    importance_df = importance_df.sort_values(

        by="Importance",

        ascending=False

    )

    print(importance_df.head(10))

    importance_df.to_csv(

        "models/feature_importance.csv",

        index=False

    )

    print("\n✓ feature_importance.csv Saved")

else:

    print("Feature importance is not available for this model.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FINAL DEPLOYMENT SUMMARY")
print("=" * 80)

print(f"Deployment Model : {best_model_name}")
print(f"Accuracy         : {final_accuracy:.4f}")
print(f"Precision        : {final_precision:.4f}")
print(f"Recall           : {final_recall:.4f}")
print(f"F1 Score         : {final_f1:.4f}")

print("\nDeployment Model Evaluation Completed Successfully.")

print("\n")
print("=" * 80)
print("SECTION 6 COMPLETED SUCCESSFULLY")
print("=" * 80)


# ============================================================
# SECTION 7 : PERFORMANCE VISUALIZATIONS
# PART A : PERFORMANCE CHARTS
# ============================================================

print("\n")
print("=" * 80)
print("SECTION 7 : PERFORMANCE CHARTS")
print("=" * 80)

# ============================================================
# PERFORMANCE DATA
# ============================================================

chart_df = optimized_results_df.copy()

# ============================================================
# ACCURACY
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    chart_df["Model"],

    chart_df["Accuracy"]

)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    "charts/model_accuracy.png",

    dpi=300

)

plt.close()

print("✓ model_accuracy.png Saved")

# ============================================================
# PRECISION
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    chart_df["Model"],

    chart_df["Precision"]

)

plt.title("Model Precision Comparison")

plt.ylabel("Precision")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    "charts/precision.png",

    dpi=300

)

plt.close()

print("✓ precision.png Saved")

# ============================================================
# RECALL
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    chart_df["Model"],

    chart_df["Recall"]

)

plt.title("Model Recall Comparison")

plt.ylabel("Recall")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    "charts/recall.png",

    dpi=300

)

plt.close()

print("✓ recall.png Saved")

# ============================================================
# F1 SCORE
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    chart_df["Model"],

    chart_df["F1 Score"]

)

plt.title("Model F1 Score Comparison")

plt.ylabel("F1 Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    "charts/f1_score.png",

    dpi=300

)

plt.close()

print("✓ f1_score.png Saved")

# ============================================================
# CROSS VALIDATION
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(

    chart_df["Model"],

    chart_df["Cross Validation"]

)

plt.title("Cross Validation Accuracy")

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(

    "charts/cross_validation.png",

    dpi=300

)

plt.close()

print("✓ cross_validation.png Saved")

print("\n")
print("=" * 80)
print("SECTION 7A COMPLETED SUCCESSFULLY")
print("=" * 80)

# ============================================================
# SECTION 7B : ADVANCED VISUALIZATIONS
# ============================================================

print("\n")
print("="*80)
print("SECTION 7B : ADVANCED VISUALIZATIONS")
print("="*80)

# ============================================================
# MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(10,6))

comparison = optimized_results_df.set_index("Model")[

    ["Accuracy","Precision","Recall","F1 Score"]

]

comparison.plot(kind="bar")

plt.title("Model Performance Comparison")

plt.ylabel("Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(

    "charts/model_comparison.png",

    dpi=300

)

plt.close()

print("✓ model_comparison.png Saved")

# ============================================================
# BEST MODEL METRICS
# ============================================================

best_metrics = [

    final_accuracy,

    final_precision,

    final_recall,

    final_f1

]

metric_names = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score"

]

plt.figure(figsize=(7,5))

plt.bar(

    metric_names,

    best_metrics

)

plt.title(f"{best_model_name} Performance")

plt.ylim(0,1)

plt.tight_layout()

plt.savefig(

    "charts/best_model_metrics.png",

    dpi=300

)

plt.close()

print("✓ best_model_metrics.png Saved")

# ============================================================
# CONFUSION MATRIX HEATMAP
# ============================================================

plt.figure(figsize=(6,5))

plt.imshow(

    cm,

    interpolation="nearest"

)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(target_encoder.classes_))

plt.xticks(

    tick_marks,

    target_encoder.classes_,

    rotation=45

)

plt.yticks(

    tick_marks,

    target_encoder.classes_

)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(

    "charts/confusion_matrix.png",

    dpi=300

)

plt.close()

print("✓ confusion_matrix.png Saved")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if "importance_df" in locals():

    plt.figure(figsize=(10,6))

    top_features = importance_df.head(10)

    plt.barh(

        top_features["Feature"],

        top_features["Importance"]

    )

    plt.title("Top 10 Important Features")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(

        "charts/feature_importance.png",

        dpi=300

    )

    plt.close()

    print("✓ feature_importance.png Saved")

print("\n")
print("="*80)
print("SECTION 7B COMPLETED SUCCESSFULLY")
print("="*80)

# ============================================================
# SECTION 7C : FINAL PROJECT VISUALIZATIONS
# ============================================================

# ============================================================
# SECTION 7C : FINAL PROJECT VISUALIZATIONS
# ============================================================

print("\n")
print("="*80)
print("SECTION 7C : FINAL PROJECT VISUALIZATIONS")
print("="*80)

# ============================================================
# STRESS LEVEL PIE CHART
# ============================================================

if "Stress_Level" in df.columns:

    stress_count = df["Stress_Level"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        stress_count.values,
        labels=stress_count.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Stress Level Distribution")

    plt.tight_layout()

    plt.savefig(
        "charts/stress_distribution_pie.png",
        dpi=300
    )

    plt.close()

    print("✓ stress_distribution_pie.png Saved")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

if "Stress_Level" in df.columns:

    plt.figure(figsize=(7,5))

    plt.bar(
        stress_count.index,
        stress_count.values
    )

    plt.title("Stress Level Distribution")

    plt.xlabel("Stress Level")

    plt.ylabel("Students")

    plt.tight_layout()

    plt.savefig(
        "charts/class_distribution.png",
        dpi=300
    )

    plt.close()

    print("✓ class_distribution.png Saved")

# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10,8))

corr = df.select_dtypes(include="number").corr()

plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "charts/correlation_heatmap.png",
    dpi=300
)

plt.close()

print("✓ correlation_heatmap.png Saved")

# ============================================================
# TOP 5 MODELS
# ============================================================

top5 = results_df.sort_values(
    by="Accuracy",
    ascending=False
).head(5)

plt.figure(figsize=(8,5))

plt.bar(
    top5["Model"],
    top5["Accuracy"]
)

plt.xticks(rotation=20)

plt.ylabel("Accuracy")

plt.title("Top 5 Models")

plt.tight_layout()

plt.savefig(
    "charts/top5_models.png",
    dpi=300
)

plt.close()

print("✓ top5_models.png Saved")

# ============================================================
# TOP 10 FEATURES
# ============================================================

if hasattr(deployment_model, "feature_importances_"):

    importance = deployment_model.feature_importances_

    feature_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(

        by="Importance",

        ascending=False

    )

    feature_df.to_csv(

        "models/feature_importance.csv",

        index=False

    )

    plt.figure(figsize=(9,6))

    plt.barh(

        feature_df["Feature"].head(10),

        feature_df["Importance"].head(10)

    )

    plt.gca().invert_yaxis()

    plt.title("Top 10 Important Features")

    plt.tight_layout()

    plt.savefig(

        "charts/top10_features.png",

        dpi=300

    )

    plt.close()

    print("✓ top10_features.png Saved")

# ============================================================
# CHART SUMMARY
# ============================================================

print("\n")
print("="*80)
print("ALL PROJECT CHARTS GENERATED")
print("="*80)

chart_list=[

"model_accuracy.png",
"precision.png",
"recall.png",
"f1_score.png",
"cross_validation.png",
"model_comparison.png",
"best_model_metrics.png",
"confusion_matrix.png",
"feature_importance.png",
"stress_distribution_pie.png",
"class_distribution.png",
"correlation_heatmap.png",
"top10_features.png",
"top5_models.png"

]

for chart in chart_list:

    print("✓",chart)

print("\nTotal Charts :",len(chart_list))

print("\n")
print("="*80)
print("SECTION 7 COMPLETED SUCCESSFULLY")
print("="*80)

# ============================================================
# SECTION 8 : SAVE PROJECT FILES & FINAL SUMMARY
# ============================================================

print("\n")
print("="*80)
print("SECTION 8 : SAVE PROJECT FILES & FINAL SUMMARY")
print("="*80)

import os
import shutil

# ============================================================
# ENSURE REQUIRED FOLDERS EXIST
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# ============================================================
# SAVE DEPLOYMENT MODEL
# ============================================================

joblib.dump(deployment_model, "models/stress_model.pkl")
print("✓ stress_model.pkl Saved")

joblib.dump(scaler, "models/scaler.pkl")
print("✓ scaler.pkl Saved")

joblib.dump(label_encoders, "models/label_encoders.pkl")
print("✓ label_encoders.pkl Saved")

joblib.dump(feature_names, "models/features.pkl")
print("✓ features.pkl Saved")

# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("✓ model_comparison.csv Saved")

# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

if hasattr(deployment_model, "feature_importances_"):

    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": deployment_model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(

        by="Importance",

        ascending=False

    )

    feature_importance.to_csv(

        "models/feature_importance.csv",

        index=False

    )

    print("✓ feature_importance.csv Saved")

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df.to_csv(

    "models/confusion_matrix.csv"

)

print("✓ confusion_matrix.csv Saved")

# ============================================================
# CREATE TRAINING REPORT
# ============================================================

with open("models/training_report.txt", "w") as file:

    file.write("="*60 + "\n")
    file.write("ACADEMIC STRESS PREDICTION SYSTEM\n")
    file.write("="*60 + "\n\n")

    file.write(f"Deployment Model : {best_model_name}\n")
    file.write(f"Accuracy         : {final_accuracy:.4f}\n")
    file.write(f"Precision        : {final_precision:.4f}\n")
    file.write(f"Recall           : {final_recall:.4f}\n")
    file.write(f"F1 Score         : {final_f1:.4f}\n\n")

    file.write("MODEL COMPARISON\n")
    file.write(results_df.to_string(index=False))

print("✓ training_report.txt Saved")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("="*80)
print("FINAL TRAINING SUMMARY")
print("="*80)

print(f"Deployment Model : {best_model_name}")
print(f"Accuracy         : {final_accuracy:.4f}")
print(f"Precision        : {final_precision:.4f}")
print(f"Recall           : {final_recall:.4f}")
print(f"F1 Score         : {final_f1:.4f}")

print("\nTop 10 Important Features")

if hasattr(deployment_model, "feature_importances_"):

    print(feature_importance.head(10))

# ============================================================
# GENERATED FILES
# ============================================================

print("\n")
print("="*80)
print("FILES GENERATED")
print("="*80)

generated_files = [

    "models/stress_model.pkl",
    "models/scaler.pkl",
    "models/label_encoders.pkl",
    "models/features.pkl",
    "models/model_comparison.csv",
    "models/feature_importance.csv",
    "models/confusion_matrix.csv",
    "models/classification_report.txt",
    "models/training_report.txt",

    "charts/model_accuracy.png",
    "charts/precision.png",
    "charts/recall.png",
    "charts/f1_score.png",
    "charts/cross_validation.png",
    "charts/model_comparison.png",
    "charts/best_model_metrics.png",
    "charts/confusion_matrix.png",
    "charts/feature_importance.png",
    "charts/stress_distribution_pie.png",
    "charts/class_distribution.png",
    "charts/correlation_heatmap.png",
    "charts/top10_features.png",
    "charts/top5_models.png"

]

for file in generated_files:

    if os.path.exists(file):

        print("✓", file)

# ============================================================
# PIPELINE COMPLETED
# ============================================================

print("\n")
print("="*80)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)

print("\n✓ Dataset Loaded")
print("✓ Data Preprocessed")
print("✓ Features Encoded")
print("✓ Data Split")
print("✓ Models Trained")
print("✓ Hyperparameter Tuning Completed")
print("✓ Final Model Evaluated")
print("✓ Reports Generated")
print("✓ Charts Generated")
print("✓ Deployment Model Saved")
print("✓ Ready for Flask Application")
print("✓ Ready for Project Demonstration")
print("✓ Ready for IEEE Paper")

print("\n")
print("="*80)
print("END OF TRAINING")
print("="*80)