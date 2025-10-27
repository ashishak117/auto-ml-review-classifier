import argparse
import os
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from utils import load_csv, ensure_dir, save_model, timestamp


def build_pipelines() -> Dict[str, Dict[str, Any]]:
    """
    Define multiple model candidates + hyperparameter grids.
    We keep grids modest so training is fast in CI and on laptops.
    """
    candidates: Dict[str, Dict[str, Any]] = {}

    # Common vectorizer; we’ll grid a few TF-IDF knobs per model
    tfidf = TfidfVectorizer(stop_words="english", strip_accents="unicode")

    # 1) Logistic Regression (strong baseline for sparse text)
    pipe_lr = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
        ]
    )
    grid_lr = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "clf__C": [0.5, 1.0, 2.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["liblinear", "lbfgs"],
    }
    candidates["logreg"] = {"pipeline": pipe_lr, "param_grid": grid_lr}

    # 2) Linear SVM (works great for text)
    pipe_svm = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", LinearSVC()),
        ]
    )
    grid_svm = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "clf__C": [0.5, 1.0, 2.0],
    }
    candidates["linsvc"] = {"pipeline": pipe_svm, "param_grid": grid_svm}

    # 3) Multinomial Naive Bayes (fast, strong on word counts)
    pipe_nb = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("clf", MultinomialNB()),
        ]
    )
    grid_nb = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "clf__alpha": [0.1, 0.5, 1.0],
    }
    candidates["mnb"] = {"pipeline": pipe_nb, "param_grid": grid_nb}

    return candidates


def evaluate(y_true, y_pred) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train AutoML-style text classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default="src/data/restaurant_reviews_sample.csv",
        help="Path to CSV with columns: text,label",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save trained artifacts",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="StratifiedKFold CV splits",
    )
    args = parser.parse_args()

    ensure_dir(args.models_dir)
    df = load_csv(args.data)

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    # Hold-out split for a final unbiased check after CV model selection
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    candidates = build_pipelines()
    best_global = {
        "name": None,
        "estimator": None,
        "cv_score": -np.inf,
        "cv_metric": "f1_weighted",
        "best_params": None,
    }

    # We’ll optimize f1_weighted to balance classes
    scoring = "f1_weighted"

    for name, cfg in candidates.items():
        print(f"\n=== Searching: {name} ===")
        search = GridSearchCV(
            cfg["pipeline"],
            cfg["param_grid"],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        print(f"[{name}] best {scoring}: {search.best_score_:.4f}")
        print(f"[{name}] best params: {search.best_params_}")

        if search.best_score_ > best_global["cv_score"]:
            best_global.update(
                {
                    "name": name,
                    "estimator": search.best_estimator_,
                    "cv_score": float(search.best_score_),
                    "best_params": search.best_params_,
                }
            )

    # Final evaluation on holdout
    best_model = best_global["estimator"]
    y_pred = best_model.predict(X_holdout)
    metrics = evaluate(y_holdout, y_pred)
    print("\n=== Holdout metrics ===")
    print(classification_report(y_holdout, y_pred, digits=4))
    print(metrics)

    # Save model + metadata
    tag = timestamp()
    model_path = os.path.join(args.models_dir, f"best_model_{tag}.joblib")
    meta = {
        "candidate": best_global["name"],
        "cv_metric": best_global["cv_metric"],
        "cv_score": best_global["cv_score"],
        "best_params": best_global["best_params"],
        "holdout_metrics": metrics,
        "data_path": os.path.abspath(args.data),
    }
    save_model(best_model, model_path, meta=meta)
    print(f"\nSaved best model to: {model_path}")
    print("Metadata saved alongside as *_meta.json")


if __name__ == "__main__":
    main()
