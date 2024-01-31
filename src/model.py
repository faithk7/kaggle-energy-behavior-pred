import lightgbm as lgb
import numpy as np
from sklearn.ensemble import VotingRegressor


class Model:
    def __init__(self):
        self.model_parameters = {
            "n_estimators": 2000,
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "colsample_bytree": 0.89,
            "colsample_bynode": 0.596,
            "lambda_l1": 3.4895,
            "lambda_l2": 1.489,
            "max_depth": 15,
            "num_leaves": 490,
            "min_data_in_leaf": 48,
            "max_bin": 840,
        }

        self.model_consumption = VotingRegressor(
            [
                (
                    f"consumption_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters, random_state=i),
                )
                for i in range(8)
            ]
        )
        self.model_production = VotingRegressor(
            [
                (
                    f"production_lgb_{i}",
                    lgb.LGBMRegressor(**self.model_parameters, random_state=i),
                )
                for i in range(8)
            ]
        )

    def fit(self, df_train_features):
        mask = df_train_features["is_consumption"] == 1
        self.model_consumption.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"],
        )

        mask = df_train_features["is_consumption"] == 0
        self.model_production.fit(
            X=df_train_features[mask].drop(columns=["target"]),
            y=df_train_features[mask]["target"],
        )

    def predict(self, df_features):
        predictions = np.zeros(len(df_features))

        mask = df_features["is_consumption"] == 1
        predictions[mask.values] = self.model_consumption.predict(
            df_features[mask]
        ).clip(0)

        mask = df_features["is_consumption"] == 0
        predictions[mask.values] = self.model_production.predict(
            df_features[mask]
        ).clip(0)

        return predictions
