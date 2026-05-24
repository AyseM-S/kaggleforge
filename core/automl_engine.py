"""
AutoML Engine Module — Competition-Grade
Uses PyCaret with XGBoost, LightGBM, CatBoost + ensembling.
Supports Quick/Standard/Competition quality modes.
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore")


# Quality mode constants
MODE_QUICK = "quick"         # Compare only (~2 min)
MODE_STANDARD = "standard"   # Compare + tune best (~5 min)
MODE_COMPETITION = "competition"  # Compare + tune top 3 + ensemble (~15 min)


class AutoMLEngine:
    """Competition-grade AutoML engine powered by PyCaret + boosted trees + ensembling."""

    def __init__(self):
        self.experiment = None
        self.best_model = None
        self.all_top_models: list = []
        self.comparison_results: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.DataFrame] = None
        self.problem_type: str = "classification"
        self.setup_done = False
        self.ensemble_method: str = "single"  # "single", "blend", or "stack"
        self._sort_metric: Optional[str] = None

    def run_pipeline(
        self,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        target_column: str,
        id_column: Optional[str],
        problem_type: str = "classification",
        metric: Optional[str] = None,
        quality_mode: str = MODE_STANDARD,
        n_folds: int = 5,
        output_dir: str = "outputs",
    ) -> dict:
        """
        Full AutoML pipeline:
        1. Setup PyCaret experiment
        2. Compare all models (including XGBoost, LightGBM, CatBoost)
        3. Tune top models
        4. Ensemble (blend/stack) in Competition mode
        5. Generate predictions + submission.csv

        Returns dict with results info.
        """
        self.problem_type = problem_type
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "status": "running",
            "best_model_name": None,
            "best_score": None,
            "comparison_html": None,
            "submission_path": None,
            "ensemble_method": "single",
            "logs": [],
        }

        try:
            # Step 1: Setup
            results["logs"].append("Setting up PyCaret experiment...")
            self._setup_experiment(train_df, target_column, n_folds, metric)
            results["logs"].append(f"✅ Setup complete. Problem type: {problem_type}")

            # Step 2: Compare all models
            results["logs"].append("Comparing all models (including XGBoost, LightGBM, CatBoost)...")
            n_select = 3 if quality_mode == MODE_COMPETITION else 1
            self._compare_models(n_select=n_select)
            results["logs"].append(
                f"✅ Model comparison complete. {len(self.comparison_results)} models evaluated."
            )

            # Step 3: Tune (Standard + Competition)
            if quality_mode in (MODE_STANDARD, MODE_COMPETITION):
                if quality_mode == MODE_COMPETITION and len(self.all_top_models) > 1:
                    results["logs"].append(
                        f"Tuning top {len(self.all_top_models)} models..."
                    )
                    self._tune_top_models()
                    results["logs"].append("✅ Top models tuned.")
                else:
                    results["logs"].append("Tuning best model...")
                    self._tune_best_model()
                    results["logs"].append("✅ Best model tuned.")

            # Step 4: Ensemble (Competition mode only)
            if quality_mode == MODE_COMPETITION and len(self.all_top_models) >= 2:
                results["logs"].append("Creating ensemble (blending top models)...")
                self._create_ensemble()
                results["logs"].append(
                    f"✅ Ensemble created using: {self.ensemble_method}"
                )

            results["ensemble_method"] = self.ensemble_method

            # Step 5: Finalize
            results["logs"].append("Finalizing model (training on full dataset)...")
            self._finalize_model()
            results["logs"].append("✅ Model finalized.")

            # Step 6: Generate predictions
            if test_df is not None:
                results["logs"].append("Generating predictions on test set...")
                submission_path = self._generate_submission(
                    test_df, id_column, target_column, output_dir
                )
                results["submission_path"] = submission_path
                results["logs"].append(f"✅ Submission saved to: {submission_path}")

            # Collect results
            results["status"] = "success"
            results["best_model_name"] = self._get_model_name()
            if self.comparison_results is not None:
                results["comparison_df"] = self.comparison_results
                results["best_score"] = self._extract_best_score()

        except Exception as e:
            results["status"] = "error"
            results["logs"].append(f"❌ Error: {str(e)}")
            results["error"] = str(e)

        return results

    # ── Setup ──────────────────────────────────────────────────────

    def _setup_experiment(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        n_folds: int = 5,
        metric: Optional[str] = None,
    ):
        """Initialize PyCaret experiment with all available algorithms."""
        # PyCaret 4.0: session_id, fold, verbose are constructor parameters
        ctor_kwargs = {
            "target": target_column,
            "session_id": 42,
            "fold": n_folds,
            "verbose": False,
            "normalize": True,
        }

        if self.problem_type == "classification":
            from pycaret.classification import ClassificationExperiment
            self.experiment = ClassificationExperiment(**ctor_kwargs)
        else:
            from pycaret.regression import RegressionExperiment
            self.experiment = RegressionExperiment(**ctor_kwargs)
            # Save bounds for automated post-processing clipping
            self._y_min = float(train_df[target_column].min())
            self._y_max = float(train_df[target_column].max())

        # Store the sort metric for use in compare_models
        self._sort_metric = None
        if metric:
            metric_map = {
                "accuracy": "Accuracy",
                "auc": "AUC",
                "f1": "F1",
                "precision": "Precision",
                "recall": "Recall",
                "rmse": "RMSE",
                "mae": "MAE",
                "r2": "R2",
                "mse": "MSE",
                "rmsle": "RMSLE",
                "logloss": "LogLoss",
            }
            self._sort_metric = metric_map.get(metric.lower(), metric)

        # PyCaret 4.0: fit() takes only the DataFrame
        self.experiment.fit(train_df)
        self.setup_done = True

    # ── Compare ────────────────────────────────────────────────────

    def _compare_models(self, n_select: int = 1):
        """Compare all models including XGBoost, LightGBM, CatBoost."""
        if not self.setup_done:
            raise RuntimeError("Experiment not set up yet.")

        compare_kwargs = {
            "verbose": False,
            "n_select": n_select,
            "turbo": True,  # Keep True to avoid extremely slow models like SVM/MLP
        }
        if self._sort_metric:
            compare_kwargs["sort"] = self._sort_metric

        result = self.experiment.compare_models(**compare_kwargs)

        # PyCaret 4.0 returns a CompareResult with .best and .leaderboard
        if hasattr(result, "best"):
            self.best_model = result.best
            self.comparison_results = result.leaderboard
            # Get all selected models
            if hasattr(result, "models") and n_select > 1:
                self.all_top_models = list(result.models[:n_select])
            else:
                self.all_top_models = [result.best]
        else:
            # Fallback for older API
            if isinstance(result, list):
                self.all_top_models = result
                self.best_model = result[0]
            else:
                self.all_top_models = [result]
                self.best_model = result
            self.comparison_results = self.experiment.pull()

    # ── Tune ───────────────────────────────────────────────────────

    def _tune_best_model(self):
        """Tune the single best model's hyperparameters."""
        if self.best_model is None:
            raise RuntimeError("No best model to tune.")

        try:
            result = self.experiment.tune_model(
                self.best_model,
                verbose=False,
                n_iter=20,
            )
            if hasattr(result, "pipeline"):
                self.best_model = result.pipeline
            else:
                self.best_model = result
            self.all_top_models[0] = self.best_model
        except Exception:
            pass

    def _tune_top_models(self):
        """Tune each of the top N models individually."""
        tuned_models = []
        for i, model in enumerate(self.all_top_models):
            try:
                result = self.experiment.tune_model(
                    model,
                    verbose=False,
                    n_iter=15,
                )
                if hasattr(result, "pipeline"):
                    tuned_models.append(result.pipeline)
                else:
                    tuned_models.append(result)
            except Exception:
                tuned_models.append(model)  # Keep untuned if tuning fails

        self.all_top_models = tuned_models
        if tuned_models:
            self.best_model = tuned_models[0]

    # ── Ensemble ───────────────────────────────────────────────────

    def _create_ensemble(self):
        """Create a blended ensemble from top models."""
        if len(self.all_top_models) < 2:
            return

        # Try blending first (simpler, usually works well)
        try:
            result = self.experiment.blend_models(
                self.all_top_models,
                verbose=False,
            )
            if hasattr(result, "pipeline"):
                blend_model = result.pipeline
            elif hasattr(result, "best"):
                blend_model = result.best
            else:
                blend_model = result

            self.best_model = blend_model
            self.ensemble_method = "blend"
            return
        except Exception:
            pass

        # Fallback: try stacking
        try:
            result = self.experiment.stack_models(
                self.all_top_models,
                verbose=False,
            )
            if hasattr(result, "pipeline"):
                stack_model = result.pipeline
            elif hasattr(result, "best"):
                stack_model = result.best
            else:
                stack_model = result

            self.best_model = stack_model
            self.ensemble_method = "stack"
        except Exception:
            # If both fail, keep single best
            self.ensemble_method = "single"

    # ── Finalize ───────────────────────────────────────────────────

    def _finalize_model(self):
        """Finalize the model (train on full dataset including holdout)."""
        if self.best_model is None:
            raise RuntimeError("No model to finalize.")

        result = self.experiment.finalize_model(self.best_model)
        if hasattr(result, "pipeline"):
            self.best_model = result.pipeline
        else:
            self.best_model = result

    # ── Predict ────────────────────────────────────────────────────

    def _generate_submission(
        self,
        test_df: pd.DataFrame,
        id_column: Optional[str],
        target_column: str,
        output_dir: str,
    ) -> str:
        """Generate predictions and save submission CSV."""
        result = self.experiment.predict_model(self.best_model, data=test_df)

        # PyCaret 4.0 returns PredictResult with .predictions DataFrame
        if hasattr(result, "predictions"):
            predictions = result.predictions
        else:
            predictions = result

        # Build submission DataFrame
        submission = pd.DataFrame()

        if id_column and id_column in test_df.columns:
            submission[id_column] = test_df[id_column]

        # PyCaret adds 'prediction_label' for classification/regression
        if "prediction_label" in predictions.columns:
            submission[target_column] = predictions["prediction_label"]
        elif "Label" in predictions.columns:
            submission[target_column] = predictions["Label"]
        else:
            pred_col = [c for c in predictions.columns if c not in test_df.columns]
            if pred_col:
                submission[target_column] = predictions[pred_col[0]]

        # Post-Processing: Clip regression predictions to train bounds
        if self.problem_type == "regression" and hasattr(self, "_y_min") and hasattr(self, "_y_max"):
            submission[target_column] = np.clip(
                submission[target_column], self._y_min, self._y_max
            )

        submission_path = os.path.join(output_dir, "submission.csv")
        submission.to_csv(submission_path, index=False)
        self.predictions = submission

        return submission_path

    # ── Helpers ────────────────────────────────────────────────────

    def _get_model_name(self) -> str:
        """Get a human-readable name for the best model."""
        model = self.best_model
        if model is None:
            return "N/A"

        if hasattr(model, "steps"):
            model = model.steps[-1][1]

        name = type(model).__name__

        # Map class names to friendly names
        name_map = {
            "XGBClassifier": "XGBoost",
            "XGBRegressor": "XGBoost",
            "LGBMClassifier": "LightGBM",
            "LGBMRegressor": "LightGBM",
            "CatBoostClassifier": "CatBoost",
            "CatBoostRegressor": "CatBoost",
            "RandomForestClassifier": "Random Forest",
            "RandomForestRegressor": "Random Forest",
            "GradientBoostingClassifier": "Gradient Boosting",
            "GradientBoostingRegressor": "Gradient Boosting",
            "ExtraTreesClassifier": "Extra Trees",
            "ExtraTreesRegressor": "Extra Trees",
            "VotingClassifier": "Blended Ensemble",
            "VotingRegressor": "Blended Ensemble",
            "StackingClassifier": "Stacked Ensemble",
            "StackingRegressor": "Stacked Ensemble",
        }

        friendly = name_map.get(name, name)
        if self.ensemble_method != "single":
            friendly = f"{friendly} ({self.ensemble_method})"
        return friendly

    def _extract_best_score(self) -> Optional[float]:
        """Extract the best score from comparison results."""
        if self.comparison_results is None:
            return None
        try:
            first_row = self.comparison_results.iloc[0]
            score_cols = [
                c
                for c in self.comparison_results.columns
                if c not in ["Model", "TT (Sec)", "model_id"]
            ]
            if score_cols:
                return round(float(first_row[score_cols[0]]), 4)
        except Exception:
            pass
        return None

    def get_comparison_text(self) -> str:
        """Get a text representation of model comparison results."""
        if self.comparison_results is None:
            return "No comparison results available."
        return self.comparison_results.to_string()

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the best model if available."""
        if self.best_model is None or self.experiment is None:
            return None

        try:
            model = self.best_model
            if hasattr(model, "steps"):
                model = model.steps[-1][1]

            if hasattr(model, "feature_importances_"):
                try:
                    X_train = self.experiment.X_train
                    feature_names = list(X_train.columns)
                except Exception:
                    try:
                        X_train = self.experiment.get_config("X_train")
                        feature_names = list(X_train.columns)
                    except Exception:
                        feature_names = [
                            f"Feature_{i}"
                            for i in range(len(model.feature_importances_))
                        ]

                fi = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)
                return fi
        except Exception:
            pass

        return None

    @staticmethod
    def get_metric_info(metric_name: Optional[str], problem_type: str) -> dict:
        """Return info about a metric: direction, description, ideal value."""
        # Higher is better
        higher_better = {
            "Accuracy": ("Higher is better", "1.0 = perfect", "↑"),
            "AUC": ("Higher is better", "1.0 = perfect", "↑"),
            "F1": ("Higher is better", "1.0 = perfect", "↑"),
            "Precision": ("Higher is better", "1.0 = perfect", "↑"),
            "Recall": ("Higher is better", "1.0 = perfect", "↑"),
            "R2": ("Higher is better", "1.0 = perfect", "↑"),
            "Kappa": ("Higher is better", "1.0 = perfect", "↑"),
            "MCC": ("Higher is better", "1.0 = perfect", "↑"),
        }
        # Lower is better
        lower_better = {
            "RMSE": ("Lower is better", "0 = perfect", "↓"),
            "MAE": ("Lower is better", "0 = perfect", "↓"),
            "MSE": ("Lower is better", "0 = perfect", "↓"),
            "RMSLE": ("Lower is better", "0 = perfect", "↓"),
            "LogLoss": ("Lower is better", "0 = perfect", "↓"),
        }

        if metric_name and metric_name in higher_better:
            desc, ideal, arrow = higher_better[metric_name]
            return {"direction": desc, "ideal": ideal, "arrow": arrow, "higher_better": True}
        elif metric_name and metric_name in lower_better:
            desc, ideal, arrow = lower_better[metric_name]
            return {"direction": desc, "ideal": ideal, "arrow": arrow, "higher_better": False}
        else:
            # Default based on problem type
            if problem_type == "classification":
                return {"direction": "Higher is better", "ideal": "1.0 = perfect", "arrow": "↑", "higher_better": True}
            else:
                return {"direction": "Lower is better", "ideal": "0 = perfect", "arrow": "↓", "higher_better": False}
