"""Probability-capable candidate classifiers used by the P2 comparison."""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier


def get_models(seed: int = 42) -> dict:
    """Return six classifiers from distinct model families.

    Every estimator is wrapped in a preprocessing pipeline so imputation and
    scaling are fitted independently inside each CV fold.
    """
    specs = {
        "logistic_regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed), True),
        "random_forest": (RandomForestClassifier(n_estimators=24, max_depth=10, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=-1, random_state=seed), False),
        "hist_gradient_boosting": (HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=31, learning_rate=0.08, l2_regularization=1.0, random_state=seed), False),
        "rbf_svc": (SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed), True),
        "knn": (KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1), True),
        "lightgbm": (LGBMClassifier(n_estimators=80, learning_rate=0.05, num_leaves=31, class_weight="balanced", verbosity=-1, n_jobs=-1, random_state=seed), False),
        "gaussian_nb": (GaussianNB(), True),
    }
    models = {}
    for name, (classifier, scale) in specs.items():
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        models[name] = Pipeline(steps)
    return models
