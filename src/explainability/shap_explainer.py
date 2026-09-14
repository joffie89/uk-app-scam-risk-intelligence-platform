import numpy as np
import pandas as pd
import shap

from src.feature_engineering.feature_engineer import FEATURE_COLUMNS
from src.models.risk_model import RiskModel


class ShapExplainer:
    def __init__(
        self,
        risk_model,
        background_features,
        background_size=100
        ):
        if not isinstance(risk_model, RiskModel):
            raise TypeError("risk_model must be a RiskModel")
        if risk_model.booster is None:
            raise RuntimeError(
                "RiskModel must be trained before creating explanations"
                )
        if (
            isinstance(background_size, bool)
            or not isinstance(background_size, int)
            or background_size <= 0
            ):
            raise ValueError(
                "background_size must be a positive integer"
                )
        if risk_model.booster.feature_names != list(FEATURE_COLUMNS):
            raise ValueError(
                "RiskModel booster features must match FEATURE_COLUMNS"
                )

        self.risk_model = risk_model
        self.booster = risk_model.booster
        background_features, _ = self._prepare_features(
            background_features
            )
        training_transaction_ids = set(
            risk_model.train_transaction_ids
            )
        if not training_transaction_ids:
            raise ValueError(
                "RiskModel must contain training transaction IDs"
                )
        if not set(background_features.index).issubset(
            training_transaction_ids
            ):
            raise ValueError(
                "background_features must contain only training "
                "transactions"
                )

        background_features = background_features.sort_index()
        if len(background_features) > background_size:
            background_features = background_features.sample(
                n=background_size,
                random_state=risk_model.random_seed
                ).sort_index()
        self.background_features = background_features.copy()
        self.background_transaction_ids = tuple(
            self.background_features.index
            )
        self.tree_explainer = shap.TreeExplainer(
            self.booster,
            data=self.background_features,
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=list(FEATURE_COLUMNS)
            )

    def global_feature_importance(self, features):
        """Return mean absolute SHAP importance over supplied cases."""
        _, _, _, shap_values = self._explain(features)
        feature_importance = pd.DataFrame({
            "feature": list(FEATURE_COLUMNS),
            "mean_absolute_shap_value": shap_values.abs().mean(
                axis=0
                ).to_numpy(),
            "_feature_order": range(len(FEATURE_COLUMNS))
            })
        feature_importance = feature_importance.sort_values(
            ["mean_absolute_shap_value", "_feature_order"],
            ascending=[False, True],
            kind="mergesort"
            ).drop(columns="_feature_order")
        return feature_importance.reset_index(drop=True)

    def explain_transaction(self, features, transaction_id):
        """Return a probability-space explanation for one transaction."""
        if not isinstance(transaction_id, str):
            raise TypeError("transaction_id must be a string")
        if not transaction_id:
            raise ValueError("transaction_id must not be empty")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if transaction_id not in features.index:
            raise ValueError(
                f"Unknown transaction_id: {transaction_id}"
                )

        transaction_features = features.loc[[transaction_id]]
        (
            transaction_features,
            risk_scores,
            base_values,
            shap_values
            ) = self._explain(transaction_features)
        feature_contributions = pd.DataFrame({
            "feature": list(FEATURE_COLUMNS),
            "feature_value": transaction_features.loc[
                transaction_id
                ].to_numpy(),
            "shap_value": shap_values.loc[
                transaction_id
                ].to_numpy()
            })
        feature_contributions["absolute_shap_value"] = (
            feature_contributions["shap_value"].abs()
            )
        feature_contributions = feature_contributions.sort_values(
            "absolute_shap_value",
            ascending=False,
            kind="mergesort"
            ).reset_index(drop=True)

        risk_score = float(risk_scores.loc[transaction_id])
        return {
            "transaction_id": transaction_id,
            "base_value": float(base_values.loc[transaction_id]),
            "risk_score": risk_score,
            "predicted_label": int(
                risk_score >= self.risk_model.decision_threshold
                ),
            "feature_contributions": feature_contributions
            }

    def _prepare_features(self, features):
        self._ensure_current_model()
        risk_scores = self.risk_model.predict_risk(features)
        features = features.loc[
            :,
            list(FEATURE_COLUMNS)
            ].copy()
        return features, risk_scores

    def _explain(self, features):
        features, risk_scores = self._prepare_features(features)
        explanation = self.tree_explainer(
            features,
            check_additivity=False
            )
        shap_values = np.asarray(
            explanation.values,
            dtype="float64"
            )
        if shap_values.shape != (
            len(features),
            len(FEATURE_COLUMNS)
            ):
            raise ValueError(
                "SHAP values must match the feature matrix shape"
                )
        if not np.isfinite(shap_values).all():
            raise ValueError("SHAP values must be finite")

        base_values = np.asarray(
            explanation.base_values,
            dtype="float64"
            )
        if base_values.ndim == 0:
            base_values = np.repeat(base_values, len(features))
        base_values = base_values.reshape(-1)
        if len(base_values) != len(features):
            raise ValueError(
                "SHAP base values must align with transactions"
                )
        if (
            not np.isfinite(base_values).all()
            or (base_values < 0).any()
            or (base_values > 1).any()
            ):
            raise ValueError(
                "SHAP base values must be finite probabilities"
                )

        reconstructed_risk_scores = (
            base_values + shap_values.sum(axis=1)
            )
        if not np.allclose(
            reconstructed_risk_scores,
            risk_scores.to_numpy(),
            rtol=1e-5,
            atol=1e-5
            ):
            raise ValueError(
                "SHAP values do not reconstruct model risk scores"
                )

        base_values = pd.Series(
            base_values,
            index=features.index,
            name="base_value"
            )
        shap_values = pd.DataFrame(
            shap_values,
            index=features.index,
            columns=list(FEATURE_COLUMNS)
            )
        return features, risk_scores, base_values, shap_values

    def _ensure_current_model(self):
        if self.risk_model.booster is not self.booster:
            raise RuntimeError(
                "RiskModel changed after ShapExplainer was created"
                )
