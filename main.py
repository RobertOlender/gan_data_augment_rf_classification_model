# Load Packages.
import numpy as np
import pandas as pd
from table_evaluator import TableEvaluator
from ctgan import CTGAN
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import randint

# ======================#
# ..............Generate start df..................#
# ======================#

np.random.seed(781995)
size = 10_000
df = pd.DataFrame(
    {
        # Create a binary target varaible:
        "target": np.random.choice(["Yes", "No"], p=[0.50, 0.50], size=size),
        # Integer columns with different distributions:
        "int_uniform_dist": np.random.randint(1, 101, size=size),
        "int_normal_dist": np.random.normal(loc=55, scale=15, size=size).astype(int),
        "int_binomial_dist": np.random.binomial(n=12, p=0.5, size=size),
        "int_negative_kurtosis": (np.random.beta(a=0.5, b=0.5, size=size) * 100).astype(
            int
        ),
        "int_positive_kurtosis": np.random.laplace(loc=55, scale=15, size=size).astype(
            int
        ),
        "int_heavy_tailed": np.random.pareto(a=2, size=size).astype(int) * 10,
        "int_bimodal": np.concatenate(
            [
                np.random.normal(loc=30, scale=5, size=size // 2).astype(int),
                np.random.normal(loc=70, scale=5, size=size - size // 2).astype(int),
            ]
        ),
        "int_multimodal_3_peaks": np.concatenate(
            [
                np.random.normal(loc=10, scale=5, size=size // 3).astype(int),
                np.random.normal(loc=50, scale=5, size=size // 3).astype(int),
                np.random.normal(loc=90, scale=5, size=size - 2 * (size // 3)).astype(
                    int
                ),
            ]
        ),
        # Boolean columns with different distributions:
        "bool_normal_dist": np.random.normal(loc=0.5, scale=0.25, size=size) > 0.5,
        "bool_true_bias": np.random.choice([True, False], p=[0.85, 0.15], size=size),
        "bool_false_bias": np.random.choice([True, False], p=[0.15, 0.85], size=size),
    }
)

df.head(10)

# split into train and test set.
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=781995
)
train_df = pd.concat(
    [y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [
        y_test.reset_index(drop=True),
        X_test.reset_index(drop=True),
    ],
    axis=1,
)

train_df.head(5)
test_df.head(5)

# ======================#
# ..........Synthetic Data Generation.......#
# ======================#

# categorical features must be pointed out for CTGAN to work properly.
categorical_features = [
    "target",
    "bool_normal_dist",
    "bool_true_bias",
    "bool_false_bias",
]

# train ctgan
ctgan = CTGAN(verbose=True)
ctgan.fit(train_df, categorical_features, epochs=500)

# generate synthetic data
synthetic_samples = ctgan.sample(2_000)
synthetic_samples.head(5)

# merge synthetic_samples and
train_df_augmented = pd.concat(
    [train_df.reset_index(drop=True), synthetic_samples.reset_index(drop=True)], axis=0
)

X_train_augmented = train_df_augmented.drop(columns="target")
y_train_augmented = train_df_augmented["target"]

# ======================#
# .......Comparing the two data sets......#
# ======================#

print(train_df.shape, synthetic_samples.shape)
table_evaluator = TableEvaluator(df, synthetic_samples, cat_cols=categorical_features)
table_evaluator.visual_evaluation()

# ======================#
# ..........Random Forest Training..........#
# ======================#

param_grid = {
    "n_estimators": randint(800, 1200),
    "max_depth": randint(10, 25),
    "min_samples_split": randint(6, 12),
    "min_samples_leaf": randint(1, 12),
    "max_features": ["sqrt", None],
}

randomized_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=781995),
    param_distributions=param_grid,
    n_iter=200,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=781995,
)

# Training and hyperparameter tuning.
rf_model_real = RandomForestClassifier(random_state=781995)
rf_model_real.fit(X_train, y_train)

rf_model_augmented = RandomForestClassifier(random_state=781995)
rf_model_augmented.fit(X_train_augmented, y_train_augmented)

# Fit the randomsearch model and identify best hyperparameters.
randomized_search.fit(X_train, y_train)
best_model = randomized_search.best_estimator_
best_params = randomized_search.best_params_
print("Best Parameters Found using real training data:", str(best_params))

randomized_search_augmented = randomized_search.fit(
    X_train_augmented, y_train_augmented
)
best_model_augmented = randomized_search_augmented.best_estimator_
best_params_augmented = randomized_search_augmented.best_params_
print(
    "Best Parameters Found using augmented training data:", str(best_params_augmented)
)

# ======================#
# .........Random Forest Prediction........#
# ======================#

# Prediction
best_model = randomized_search.best_estimator_
y_pred = best_model.predict(X_test)

best_model_augmented = randomized_search_augmented.best_estimator_
y_pred_augmented = best_model_augmented.predict(X_test)

# accuracy and AUC-ROC
accuracy = accuracy_score(y_test, y_pred)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test.map({"Yes": 1, "No": 0}), y_pred_prob)

print("\nAccuracy using real data: " + str(accuracy))
print("AUC-ROC using real data: {:.2f}".format(roc_auc))

accuracy_augmented = accuracy_score(y_test, y_pred_augmented)
y_pred_prob_augmented = best_model.predict_proba(X_test)[:, 1]
roc_auc_augmented = roc_auc_score(
    y_test.map({"Yes": 1, "No": 0}), y_pred_prob_augmented
)

print("\nAccuracy using augmented data: " + str(accuracy_augmented))
print("AUC-ROC using augmented data: {:.2f}".format(roc_auc_augmented))
