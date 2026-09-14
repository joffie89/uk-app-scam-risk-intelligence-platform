import math

import pandas as pd
import xgboost as xgb

from src.feature_engineering.feature_engineer import FEATURE_COLUMNS


class RiskModel:
    def __init__(
        self,
        test_fraction=0.2,
        decision_threshold=0.5,
        random_seed=42
        ):
        if (
            isinstance(test_fraction, bool)
            or not isinstance(test_fraction, (int, float))
            or not math.isfinite(test_fraction)
            or not 0 < test_fraction < 1
            ):
            raise ValueError("test_fraction must be between 0 and 1")
        if (
            isinstance(decision_threshold, bool)
            or not isinstance(decision_threshold, (int, float))
            or not math.isfinite(decision_threshold)
            or not 0 <= decision_threshold <= 1
            ):
            raise ValueError(
                "decision_threshold must be between 0 and 1"
                )
        if (
            isinstance(random_seed, bool)
            or not isinstance(random_seed, int)
            or random_seed < 0
            ):
            raise ValueError(
                "random_seed must be a non-negative integer"
                )

        self.test_fraction = test_fraction
        self.decision_threshold = decision_threshold
        self.random_seed = random_seed
        self.booster = None
        self.metrics = None
        self.evaluation_results = None
        self.train_transaction_ids = ()
        self.test_transaction_ids = ()
        self.split_timestamp = None

    def train_and_evaluate(self, features, labels, timestamps):
        """Train the model and evaluate it against held-out fraud labels."""
        features = self._validate_features(features)
        labels = self._validate_labels(labels, features.index)
        timestamps = self._validate_timestamps(
            timestamps,
            features.index
            )
        train_ids, test_ids, split_timestamp = (
            self._chronological_split(labels, timestamps)
            )

        train_features = features.loc[train_ids]
        test_features = features.loc[test_ids]
        train_labels = labels.loc[train_ids]
        test_labels = labels.loc[test_ids]

        positive_count = int((train_labels == 1).sum())
        negative_count = int((train_labels == 0).sum())
        scale_pos_weight = negative_count / positive_count

        training_data = xgb.DMatrix(
            train_features.to_numpy(),
            label=train_labels.to_numpy(),
            feature_names=list(FEATURE_COLUMNS)
            )
        parameters = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "seed": self.random_seed,
            "nthread": 1,
            "verbosity": 0
            }
        booster = xgb.train(
            parameters,
            training_data,
            num_boost_round=100
            )

        test_risk_scores = self._predict_with_booster(
            booster,
            test_features
            )
        predicted_labels = (
            test_risk_scores >= self.decision_threshold
            ).astype("int64")
        metrics = self._calculate_metrics(
            test_labels,
            predicted_labels
            )

        self.booster = booster
        self.metrics = metrics
        self.train_transaction_ids = tuple(train_ids)
        self.test_transaction_ids = tuple(test_ids)
        self.split_timestamp = split_timestamp
        self.evaluation_results = pd.DataFrame({
            "actual_label": test_labels.astype("int64"),
            "risk_score": test_risk_scores,
            "predicted_label": predicted_labels
            })
        self.evaluation_results.index.name = features.index.name

        return metrics.copy()

    def predict_risk(self, features):
        """Return fraud-risk probabilities from a trained model."""
        if self.booster is None:
            raise RuntimeError("RiskModel must be trained before prediction")
        features = self._validate_features(features)
        return self._predict_with_booster(self.booster, features)

    def _validate_features(self, features):
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.empty:
            raise ValueError("features must not be empty")
        if features.columns.has_duplicates:
            raise ValueError("feature column names must be unique")

        expected_columns = set(FEATURE_COLUMNS)
        supplied_columns = set(features.columns)
        missing_columns = expected_columns - supplied_columns
        extra_columns = supplied_columns - expected_columns
        if missing_columns or extra_columns:
            raise ValueError(
                "features must contain exactly FEATURE_COLUMNS; "
                f"missing={sorted(missing_columns)}, "
                f"extra={sorted(extra_columns)}"
                )
        if features.index.has_duplicates:
            raise ValueError("feature transaction IDs must be unique")
        if features.index.hasnans:
            raise ValueError("feature transaction IDs must not be missing")
        if any(
            not isinstance(transaction_id, str)
            or not transaction_id
            for transaction_id in features.index
            ):
            raise ValueError(
                "feature transaction IDs must be non-empty strings"
                )

        features = features.loc[:, list(FEATURE_COLUMNS)].copy()
        invalid_type_columns = [
            column
            for column in FEATURE_COLUMNS
            if (
                not pd.api.types.is_numeric_dtype(features[column])
                or pd.api.types.is_bool_dtype(features[column])
                )
            ]
        if invalid_type_columns:
            raise ValueError(
                "features must be numeric; invalid columns: "
                f"{invalid_type_columns}"
                )
        if features.isnull().any().any():
            raise ValueError("features must not contain missing values")
        if any(
            not math.isfinite(float(value))
            for column in FEATURE_COLUMNS
            for value in features[column]
            ):
            raise ValueError("features must contain only finite values")

        return features

    def _validate_labels(self, labels, feature_index):
        if not isinstance(labels, pd.Series):
            raise TypeError("labels must be a pandas Series")
        if labels.empty:
            raise ValueError("labels must not be empty")
        if labels.index.has_duplicates:
            raise ValueError("label transaction IDs must be unique")
        if labels.index.hasnans:
            raise ValueError("label transaction IDs must not be missing")
        if any(
            not isinstance(transaction_id, str)
            or not transaction_id
            for transaction_id in labels.index
            ):
            raise ValueError(
                "label transaction IDs must be non-empty strings"
                )
        if set(labels.index) != set(feature_index):
            raise ValueError(
                "label transaction IDs must exactly match feature IDs"
                )
        if labels.isnull().any():
            raise ValueError("labels must not contain missing values")
        if not (
            pd.api.types.is_bool_dtype(labels)
            or pd.api.types.is_numeric_dtype(labels)
            ):
            raise ValueError("labels must contain only binary values")

        labels = labels.reindex(feature_index)
        if not labels.isin([0, 1, False, True]).all():
            raise ValueError("labels must contain only 0 and 1")
        labels = labels.astype("int64")
        class_counts = labels.value_counts()
        if set(class_counts.index) != {0, 1}:
            raise ValueError("labels must contain both fraud classes")
        if (class_counts < 2).any():
            raise ValueError(
                "each fraud class requires at least two transactions"
                )

        return labels

    def _validate_timestamps(self, timestamps, feature_index):
        if not isinstance(timestamps, pd.Series):
            raise TypeError("timestamps must be a pandas Series")
        if timestamps.empty:
            raise ValueError("timestamps must not be empty")
        if timestamps.index.has_duplicates:
            raise ValueError("timestamp transaction IDs must be unique")
        if timestamps.index.hasnans:
            raise ValueError(
                "timestamp transaction IDs must not be missing"
                )
        if set(timestamps.index) != set(feature_index):
            raise ValueError(
                "timestamp transaction IDs must exactly match feature IDs"
                )
        if timestamps.isnull().any():
            raise ValueError("timestamps must not contain missing values")

        timestamps = timestamps.reindex(feature_index).copy()
        try:
            timestamps = pd.to_datetime(timestamps, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("timestamps must contain valid dates") from error
        if timestamps.isnull().any():
            raise ValueError("timestamps must contain valid dates")

        return timestamps

    def _chronological_split(self, labels, timestamps):
        ordered_transactions = pd.DataFrame({
            "timestamp": timestamps,
            "label": labels,
            "_transaction_id": labels.index
            }).sort_values(
                ["timestamp", "_transaction_id"],
                kind="mergesort"
                )
        target_test_count = round(
            len(ordered_transactions) * self.test_fraction
            )
        target_test_count = max(1, target_test_count)
        target_test_count = min(
            len(ordered_transactions) - 1,
            target_test_count
            )
        split_candidates = []
        timestamp_counts = ordered_transactions.groupby(
            "timestamp",
            sort=False
            ).size()
        remaining_test_count = len(ordered_transactions)
        for position, (split_timestamp, cohort_count) in enumerate(
            timestamp_counts.items()
            ):
            if position:
                train_count = (
                    len(ordered_transactions) - remaining_test_count
                    )
                if train_count >= 2 and remaining_test_count >= 2:
                    split_candidates.append((
                        abs(
                            remaining_test_count - target_test_count
                            ),
                        split_timestamp
                        ))
            remaining_test_count -= int(cohort_count)
        if not split_candidates:
            if len(timestamp_counts) < 2:
                raise ValueError(
                    "chronological evaluation requires distinct timestamps"
                    )
            raise ValueError(
                "chronological evaluation requires at least two "
                "transactions on each side"
                )

        _, split_timestamp = min(
            split_candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1]
                )
            )
        train_mask = (
            ordered_transactions["timestamp"] < split_timestamp
            )
        test_mask = ~train_mask
        train_labels = ordered_transactions.loc[train_mask, "label"]
        test_labels = ordered_transactions.loc[test_mask, "label"]
        if set(train_labels) != {0, 1}:
            raise ValueError(
                "Chronological training data must contain both "
                "fraud classes"
                )
        if set(test_labels) != {0, 1}:
            raise ValueError(
                "Chronological test data must contain both fraud classes"
                )

        train_ids = ordered_transactions.index[train_mask].tolist()
        test_ids = ordered_transactions.index[test_mask].tolist()
        return train_ids, test_ids, split_timestamp

    def _predict_with_booster(self, booster, features):
        prediction_data = xgb.DMatrix(
            features.to_numpy(),
            feature_names=list(FEATURE_COLUMNS)
            )
        risk_scores = pd.Series(
            booster.predict(prediction_data),
            index=features.index,
            name="risk_score",
            dtype="float64"
            )
        if any(
            not math.isfinite(float(score))
            or not 0 <= score <= 1
            for score in risk_scores
            ):
            raise ValueError(
                "model risk scores must be finite probabilities"
                )
        return risk_scores

    def _calculate_metrics(self, actual_labels, predicted_labels):
        true_positives = int((
            (actual_labels == 1)
            & (predicted_labels == 1)
            ).sum())
        false_positives = int((
            (actual_labels == 0)
            & (predicted_labels == 1)
            ).sum())
        false_negatives = int((
            (actual_labels == 1)
            & (predicted_labels == 0)
            ).sum())

        if true_positives + false_positives:
            precision = (
                true_positives
                / (true_positives + false_positives)
                )
        else:
            precision = 0.0
        if true_positives + false_negatives:
            recall = (
                true_positives
                / (true_positives + false_negatives)
                )
        else:
            recall = 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
            }
