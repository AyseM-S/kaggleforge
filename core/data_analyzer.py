"""
Data Analyzer Module
Profiles uploaded CSV files and creates structured summaries
for the AI brain and AutoML engine to consume.
"""

import pandas as pd
import numpy as np
from typing import Optional


class DataAnalyzer:
    """Analyzes competition data files and extracts useful metadata."""

    def __init__(self):
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.sample_sub_df: Optional[pd.DataFrame] = None
        self.extra_dfs: dict[str, pd.DataFrame] = {}
        self.target_column: Optional[str] = None
        self.id_column: Optional[str] = None
        self.problem_type: Optional[str] = None  # "classification" or "regression"

    def load_file(self, file_bytes, filename: str) -> pd.DataFrame:
        """Load a CSV file from uploaded bytes."""
        import io
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
        return df

    def set_train(self, file_bytes, filename: str = "train.csv"):
        self.train_df = self.load_file(file_bytes, filename)

    def set_test(self, file_bytes, filename: str = "test_x.csv"):
        self.test_df = self.load_file(file_bytes, filename)

    def set_sample_submission(self, file_bytes, filename: str = "sample_submission.csv"):
        self.sample_sub_df = self.load_file(file_bytes, filename)

    def add_extra_file(self, file_bytes, filename: str):
        self.extra_dfs[filename] = self.load_file(file_bytes, filename)

    def detect_target_column(self) -> Optional[str]:
        """Detect the target column by comparing train columns with test columns."""
        if self.train_df is None:
            return None

        if self.test_df is not None:
            # Target = columns in train but NOT in test
            train_cols = set(self.train_df.columns)
            test_cols = set(self.test_df.columns)
            diff = train_cols - test_cols
            if len(diff) == 1:
                self.target_column = diff.pop()
                return self.target_column

        if self.sample_sub_df is not None:
            # Target = columns in sample_submission (excluding ID-like columns)
            sub_cols = list(self.sample_sub_df.columns)
            # Usually first column is ID, rest are targets
            if len(sub_cols) >= 2:
                self.target_column = sub_cols[-1]  # Last column is often the target
                return self.target_column

        # Fallback: last column of train
        self.target_column = self.train_df.columns[-1]
        return self.target_column

    def detect_id_column(self) -> Optional[str]:
        """Detect the ID column from sample submission or test data."""
        if self.sample_sub_df is not None:
            self.id_column = self.sample_sub_df.columns[0]
        elif self.test_df is not None:
            # Look for common ID column names
            for col in self.test_df.columns:
                if col.lower() in ["id", "index", "key", "row_id", "sample_id"]:
                    self.id_column = col
                    break
            if self.id_column is None:
                self.id_column = self.test_df.columns[0]
        return self.id_column

    def detect_problem_type(self) -> str:
        """Detect if this is a classification or regression problem."""
        if self.train_df is None or self.target_column is None:
            return "unknown"

        target = self.train_df[self.target_column]

        # Check if target is categorical/string
        if target.dtype == "object" or target.dtype.name == "category":
            self.problem_type = "classification"
            return self.problem_type

        # Check number of unique values
        n_unique = target.nunique()
        n_total = len(target)

        # If few unique values relative to total, likely classification
        if n_unique <= 20 or (n_unique / n_total) < 0.05:
            # Check if values are integer-like
            if np.all(target.dropna() == target.dropna().astype(int)):
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"
        else:
            self.problem_type = "regression"

        return self.problem_type

    def get_column_summary(self, df: pd.DataFrame, name: str) -> dict:
        """Get a summary of a DataFrame's columns."""
        summary = {
            "name": name,
            "shape": df.shape,
            "columns": [],
        }

        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(df[col].isnull().mean() * 100, 1),
                "n_unique": int(df[col].nunique()),
            }

            if df[col].dtype in ["int64", "float64"]:
                col_info["mean"] = round(float(df[col].mean()), 4)
                col_info["std"] = round(float(df[col].std()), 4)
                col_info["min"] = float(df[col].min())
                col_info["max"] = float(df[col].max())
                col_info["skewness"] = round(float(df[col].skew()), 2)
            elif df[col].dtype == "object":
                top_values = df[col].value_counts().head(5).to_dict()
                col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                if col_info["n_unique"] > 50:
                    col_info["high_cardinality"] = True

            summary["columns"].append(col_info)

        return summary

    def get_full_summary(self) -> dict:
        """Generate a complete data analysis summary."""
        # Detect columns
        self.detect_target_column()
        self.detect_id_column()
        if self.target_column:
            self.detect_problem_type()

        summary = {
            "target_column": self.target_column,
            "id_column": self.id_column,
            "problem_type": self.problem_type,
            "datasets": {},
        }

        if self.train_df is not None:
            summary["datasets"]["train"] = self.get_column_summary(self.train_df, "train")
            # Add target distribution
            if self.target_column and self.target_column in self.train_df.columns:
                target = self.train_df[self.target_column]
                if self.problem_type == "classification":
                    summary["target_distribution"] = {
                        str(k): int(v)
                        for k, v in target.value_counts().head(20).to_dict().items()
                    }
                else:
                    summary["target_stats"] = {
                        "mean": round(float(target.mean()), 4),
                        "std": round(float(target.std()), 4),
                        "min": float(target.min()),
                        "max": float(target.max()),
                        "median": float(target.median()),
                    }

            # Add sample rows
            summary["sample_rows"] = self.train_df.head(3).to_dict(orient="records")

        if self.test_df is not None:
            summary["datasets"]["test"] = self.get_column_summary(self.test_df, "test")

        if self.sample_sub_df is not None:
            summary["datasets"]["sample_submission"] = self.get_column_summary(
                self.sample_sub_df, "sample_submission"
            )

        for name, df in self.extra_dfs.items():
            summary["datasets"][name] = self.get_column_summary(df, name)

        return summary

    def get_summary_text(self) -> str:
        """Generate a human-readable text summary for the AI."""
        summary = self.get_full_summary()
        lines = []

        lines.append(f"=== DATA ANALYSIS SUMMARY ===")
        lines.append(f"Problem Type: {summary['problem_type']}")
        lines.append(f"Target Column: {summary['target_column']}")
        lines.append(f"ID Column: {summary['id_column']}")
        lines.append("")

        for ds_name, ds_info in summary.get("datasets", {}).items():
            lines.append(f"--- {ds_name.upper()} ({ds_info['shape'][0]} rows × {ds_info['shape'][1]} cols) ---")
            for col in ds_info["columns"]:
                null_str = f" | {col['null_pct']}% null" if col["null_count"] > 0 else ""
                skew_str = ""
                if "skewness" in col and abs(col["skewness"]) > 2:
                    skew_str = f" | ⚠️ SKEWED({col['skewness']})"
                hc_str = " | ⚠️ HIGH-CARDINALITY" if col.get("high_cardinality") else ""
                if "mean" in col:
                    lines.append(
                        f"  {col['name']}: {col['dtype']} | "
                        f"mean={col['mean']}, std={col['std']}, "
                        f"range=[{col['min']}, {col['max']}] | "
                        f"{col['n_unique']} unique{null_str}{skew_str}"
                    )
                elif "top_values" in col:
                    top = ", ".join(f"{k}({v})" for k, v in list(col["top_values"].items())[:3])
                    lines.append(
                        f"  {col['name']}: {col['dtype']} | "
                        f"{col['n_unique']} unique | top: {top}{null_str}{hc_str}"
                    )
                else:
                    lines.append(f"  {col['name']}: {col['dtype']} | {col['n_unique']} unique{null_str}")
            lines.append("")

        if "target_distribution" in summary:
            lines.append("--- TARGET DISTRIBUTION ---")
            for k, v in summary["target_distribution"].items():
                lines.append(f"  {k}: {v}")
        elif "target_stats" in summary:
            lines.append("--- TARGET STATISTICS ---")
            for k, v in summary["target_stats"].items():
                lines.append(f"  {k}: {v}")

        if "sample_rows" in summary:
            lines.append("")
            lines.append("--- SAMPLE ROWS (first 3) ---")
            for i, row in enumerate(summary["sample_rows"]):
                lines.append(f"  Row {i}: {row}")

        return "\n".join(lines)
