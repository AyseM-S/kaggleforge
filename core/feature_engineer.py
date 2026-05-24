"""
Feature Engineering Module
Automated feature engineering pipeline for competition data.
Applies smart transformations before feeding data to AutoML.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """Automated feature engineering for competition datasets."""

    def __init__(self):
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._date_cols: list[str] = []
        self._skewed_cols: list[str] = []
        self._high_corr_drops: list[str] = []
        self._low_var_drops: list[str] = []
        self._fitted = False
        self._logs: list[str] = []

        # Stored fit parameters for consistent test transforms
        self._medians: dict = {}
        self._modes: dict = {}
        self._skew_shifts: dict = {}
        self._cap_limits: dict = {}  # (lower, upper) for each numeric col

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        target_column: str,
        test_df: Optional[pd.DataFrame] = None,
        id_column: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Fit on training data and transform both train and test.

        Returns (train_transformed, test_transformed).
        """
        self._logs = []
        self._logs.append("🔧 Starting feature engineering...")

        # Separate target and features
        y = train_df[target_column].copy()
        feature_cols = [c for c in train_df.columns if c != target_column]
        X_train = train_df[feature_cols].copy()
        X_test = test_df.copy() if test_df is not None else None

        # Detect column types
        self._detect_column_types(X_train, id_column)

        # Step 1: Extract date features
        X_train, X_test = self._extract_date_features(X_train, X_test)

        # Step 2: Handle missing values (fit on train, apply to both)
        X_train, X_test = self._handle_missing_values(X_train, X_test)

        # Step 3: Cap outliers
        X_train, X_test = self._cap_outliers(X_train, X_test)

        # Step 4: Log-transform skewed features
        X_train, X_test = self._log_transform_skewed(X_train, X_test)

        # Step 5: Drop low-variance and highly-correlated features
        X_train, X_test = self._drop_redundant_features(X_train, X_test)

        # Step 6: Create domain-specific features (Sleep, Caffeine, Fatigue)
        X_train, X_test = self._create_domain_features(X_train, X_test)

        # Step 7: Create interaction features for top numeric columns
        X_train, X_test = self._create_interactions(X_train, X_test, y)

        # Reassemble train with target
        train_out = X_train.copy()
        train_out[target_column] = y.values

        self._fitted = True
        self._logs.append(
            f"✅ Feature engineering complete. "
            f"Features: {len(feature_cols)} → {len(X_train.columns)}"
        )

        return train_out, X_test

    @property
    def logs(self) -> list[str]:
        return self._logs

    def _detect_column_types(self, df: pd.DataFrame, id_column: Optional[str]):
        """Classify columns into numeric, categorical, date types."""
        for col in df.columns:
            if id_column and col == id_column:
                continue

            if df[col].dtype in ["datetime64[ns]", "datetime64"]:
                self._date_cols.append(col)
            elif df[col].dtype == "object":
                # Check if it looks like a date
                if self._is_date_column(df[col]):
                    self._date_cols.append(col)
                else:
                    self._categorical_cols.append(col)
            elif df[col].dtype in ["int64", "float64", "int32", "float32"]:
                self._numeric_cols.append(col)

        self._logs.append(
            f"  📊 Detected: {len(self._numeric_cols)} numeric, "
            f"{len(self._categorical_cols)} categorical, "
            f"{len(self._date_cols)} date columns"
        )

    @staticmethod
    def _is_date_column(series: pd.Series) -> bool:
        """Check if a string column contains dates."""
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return False
        try:
            pd.to_datetime(sample)
            return True
        except (ValueError, TypeError):
            return False

    def _extract_date_features(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Extract year, month, day, weekday, is_weekend from date columns."""
        if not self._date_cols:
            return X_train, X_test

        for col in self._date_cols:
            for df in [X_train, X_test]:
                if df is None or col not in df.columns:
                    continue
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = dt.dt.year
                df[f"{col}_month"] = dt.dt.month
                df[f"{col}_day"] = dt.dt.day
                df[f"{col}_weekday"] = dt.dt.weekday
                df[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
                df.drop(columns=[col], inplace=True)

            # Update numeric cols list
            for suffix in ["_year", "_month", "_day", "_weekday", "_is_weekend"]:
                new_col = f"{col}{suffix}"
                if new_col not in self._numeric_cols:
                    self._numeric_cols.append(new_col)

        self._logs.append(
            f"  📅 Extracted date features from {len(self._date_cols)} columns"
        )
        return X_train, X_test

    def _handle_missing_values(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Smart missing value imputation using KNN for numeric and mode for categorical."""
        cols_with_missing = [
            c for c in X_train.columns if X_train[c].isnull().sum() > 0
        ]
        if not cols_with_missing:
            return X_train, X_test

        from sklearn.impute import KNNImputer

        numeric_missing = []
        categorical_missing = []

        for col in cols_with_missing:
            missing_pct = X_train[col].isnull().mean()
            # Add missing indicator flag if >5% missing
            if missing_pct > 0.05:
                flag_col = f"{col}_missing"
                X_train[flag_col] = X_train[col].isnull().astype(int)
                if X_test is not None and col in X_test.columns:
                    X_test[flag_col] = X_test[col].isnull().astype(int)

            if X_train[col].dtype in ["int64", "float64", "int32", "float32"]:
                numeric_missing.append(col)
            elif X_train[col].dtype == "object":
                categorical_missing.append(col)

        # Handle Categorical with Mode
        for col in categorical_missing:
            fill_val = X_train[col].mode().iloc[0] if len(X_train[col].mode()) > 0 else "MISSING"
            self._modes[col] = fill_val
            X_train[col] = X_train[col].fillna(fill_val)
            if X_test is not None and col in X_test.columns:
                X_test[col] = X_test[col].fillna(fill_val)

        # Handle Numeric with KNN Imputer
        if numeric_missing:
            # We fit KNN Imputer on all numeric columns, not just missing ones, for better context
            all_numeric = [c for c in self._numeric_cols if c in X_train.columns]
            if len(all_numeric) > 0:
                knn = KNNImputer(n_neighbors=5)
                # Fit and transform train
                X_train_num = X_train[all_numeric]
                X_train[all_numeric] = knn.fit_transform(X_train_num)
                # Transform test
                if X_test is not None:
                    X_test_num = X_test[all_numeric]
                    X_test[all_numeric] = knn.transform(X_test_num)

        self._logs.append(
            f"  🔨 Imputed missing values in {len(cols_with_missing)} columns "
            f"(KNN for numeric, mode for categorical)"
        )
        return X_train, X_test

    def _cap_outliers(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Winsorize numeric features at 1st/99th percentiles."""
        n_capped = 0
        for col in self._numeric_cols:
            if col not in X_train.columns:
                continue
            if X_train[col].dtype not in ["int64", "float64", "int32", "float32"]:
                continue

            q01 = X_train[col].quantile(0.01)
            q99 = X_train[col].quantile(0.99)

            if q01 == q99:
                continue

            self._cap_limits[col] = (q01, q99)
            X_train[col] = X_train[col].clip(q01, q99)
            if X_test is not None and col in X_test.columns:
                X_test[col] = X_test[col].clip(q01, q99)
            n_capped += 1

        if n_capped > 0:
            self._logs.append(
                f"  📏 Capped outliers in {n_capped} numeric columns (1st-99th percentile)"
            )
        return X_train, X_test

    def _log_transform_skewed(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Log-transform highly skewed numeric features."""
        n_transformed = 0
        for col in self._numeric_cols:
            if col not in X_train.columns:
                continue
            if X_train[col].dtype not in ["int64", "float64", "int32", "float32"]:
                continue

            skewness = X_train[col].skew()
            if abs(skewness) > 2.0:  # Only transform highly skewed
                min_val = X_train[col].min()
                shift = 0
                if min_val <= 0:
                    shift = abs(min_val) + 1
                self._skew_shifts[col] = shift

                X_train[col] = np.log1p(X_train[col] + shift)
                if X_test is not None and col in X_test.columns:
                    X_test[col] = np.log1p(X_test[col] + shift)
                self._skewed_cols.append(col)
                n_transformed += 1

        if n_transformed > 0:
            self._logs.append(
                f"  📐 Log-transformed {n_transformed} highly skewed features"
            )
        return X_train, X_test

    def _drop_redundant_features(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Drop near-zero-variance and highly correlated features."""
        drops = set()

        # Drop near-zero-variance (>99% same value)
        for col in X_train.columns:
            if X_train[col].dtype == "object":
                continue
            top_pct = X_train[col].value_counts(normalize=True).iloc[0]
            if top_pct > 0.99:
                drops.add(col)
                self._low_var_drops.append(col)

        # Drop highly correlated features (>0.98)
        numeric_df = X_train.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            for col in upper.columns:
                if any(upper[col] > 0.98):
                    if col not in drops:
                        drops.add(col)
                        self._high_corr_drops.append(col)

        if drops:
            X_train = X_train.drop(columns=list(drops), errors="ignore")
            if X_test is not None:
                X_test = X_test.drop(columns=list(drops), errors="ignore")
            self._logs.append(
                f"  🗑️ Dropped {len(drops)} redundant features "
                f"({len(self._low_var_drops)} low-variance, "
                f"{len(self._high_corr_drops)} highly-correlated)"
            )

        return X_train, X_test

    def _create_interactions(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        y: pd.Series,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Create interaction features for the most important numeric columns."""
        numeric_cols = [
            c for c in self._numeric_cols
            if c in X_train.columns
            and X_train[c].dtype in ["int64", "float64", "int32", "float32"]
        ]

        if len(numeric_cols) < 2:
            return X_train, X_test

        # Find top features by correlation with target (if numeric target)
        if y.dtype in ["int64", "float64", "int32", "float32"]:
            correlations = {}
            for col in numeric_cols:
                try:
                    corr = abs(X_train[col].corr(y))
                    if not np.isnan(corr):
                        correlations[col] = corr
                except Exception:
                    pass

            top_cols = sorted(correlations, key=correlations.get, reverse=True)[
                :min(5, len(correlations))
            ]
        else:
            # For categorical targets, just use the first 5 numeric cols
            top_cols = numeric_cols[:5]

        if len(top_cols) < 2:
            return X_train, X_test

        # Create ratio and product features for top pairs
        n_created = 0
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                if n_created >= 10:  # Limit to 10 interaction features
                    break
                col_a, col_b = top_cols[i], top_cols[j]

                # Ratio feature
                ratio_name = f"{col_a}_div_{col_b}"
                for df in [X_train, X_test]:
                    if df is None:
                        continue
                    denominator = df[col_b].replace(0, np.nan)
                    df[ratio_name] = df[col_a] / denominator
                    df[ratio_name] = df[ratio_name].fillna(0)
                n_created += 1

        if n_created > 0:
            self._logs.append(
                f"  🔗 Created {n_created} interaction features from top correlated columns"
            )
        return X_train, X_test

    def _create_domain_features(
        self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame]
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Create domain-specific features based on the dataset."""
        n_created = 0
        dfs = [X_train]
        if X_test is not None:
            dfs.append(X_test)

        for df in dfs:
            # 1. Kaliteli Uyku Oranı: rem_yuzdesi + derin_uyku_yuzdesi
            if "rem_yuzdesi" in df.columns and "derin_uyku_yuzdesi" in df.columns:
                df["kaliteli_uyku_orani"] = df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]
                if "kaliteli_uyku_orani" not in self._numeric_cols:
                    self._numeric_cols.append("kaliteli_uyku_orani")
                    n_created += 1

            # 2. Kafein / Kilo Oranı: uyku_oncesi_kafein_mg / vucut_kitle_indeksi
            if "uyku_oncesi_kafein_mg" in df.columns and "vucut_kitle_indeksi" in df.columns:
                # Avoid division by zero
                vki = df["vucut_kitle_indeksi"].replace(0, np.nan)
                df["kafein_vki_orani"] = df["uyku_oncesi_kafein_mg"] / vki
                df["kafein_vki_orani"] = df["kafein_vki_orani"].fillna(0)
                if "kafein_vki_orani" not in self._numeric_cols:
                    self._numeric_cols.append("kafein_vki_orani")
                    n_created += 1

            # 3. Fiziksel Yük: gunluk_adim_sayisi * gunluk_calisma_saati
            if "gunluk_adim_sayisi" in df.columns and "gunluk_calisma_saati" in df.columns:
                df["fiziksel_zihinsel_yuk"] = df["gunluk_adim_sayisi"] * df["gunluk_calisma_saati"]
                if "fiziksel_zihinsel_yuk" not in self._numeric_cols:
                    self._numeric_cols.append("fiziksel_zihinsel_yuk")
                    n_created += 1

        if n_created > 0:
            self._logs.append(
                f"  🧠 Created {n_created} domain-specific custom features (Sleep, Caffeine, Load)"
            )
        return X_train, X_test

    def get_summary(self) -> str:
        """Return a text summary of all transformations applied."""
        return "\n".join(self._logs)
