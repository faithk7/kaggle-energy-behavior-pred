from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pytest import param
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error

from data import DataStorage
from features import FeaturesGenerator
from model import Model, ModelWrapper


# TODO: can add model as a parameter as well, if you want to generalize more, you can add metrics as a parameter well
def train(features, num_fold, base_model):
    train_val_split = get_train_val_split_features(features, num_fold)
    val_scores = []
    fold_cnt = 1

    while True:
        try:
            train_feature, val_feature = next(train_val_split)
            model = ModelWrapper(base_model, clone(base_model))
            model.fit(train_feature)

            y_pred = model.predict(val_feature)
            val_score = mean_absolute_error(val_feature["target"], y_pred)
            val_scores.append(val_score)
            print(f"Training fold {fold_cnt}: val_mae_score: {val_score}")
            fold_cnt += 1
        except StopIteration:
            print(f"Mean val_mae_score: {sum(val_scores) / len(val_scores)}")
            break


# TODO: this can be added to the template
def get_train_val_split_features(features, num_fold):
    num_data = len(features)
    fold_size = num_data // num_fold

    for end_index in range(fold_size, num_data - fold_size, fold_size):
        train_feature = features[:end_index]
        # train_target = target[:end_index]
        val_feature = features[end_index : end_index + fold_size]
        # val_target = target[end_index : end_index + fold_size]
        yield train_feature, val_feature


def main():
    select_model = "default"
    data_storage = DataStorage()

    features = FeaturesGenerator(data_storage=data_storage).generate_features(
        data_storage.df_data
    )

    features = features[features["target"].notnull()]

    N_FOLDS = 10

    match select_model:
        case "default":
            model_parameters = {
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
            base_model = VotingRegressor(
                [(f"lgb_{i}", LGBMRegressor(**model_parameters)) for i in range(8)]
            )
            train(features, N_FOLDS, base_model)
        case "ctb":
            params = {
                "cat_features": [
                    "county",
                    "product_type",
                    "segment",
                    "is_business",
                    "is_consumption",
                ],
                "eval_metric": "MAE",
            }
            base_model = CatBoostRegressor(**params)
            # get all columns that have the type of category
            train(features, N_FOLDS, base_model)
        case "lgb":
            pass
        case "xgb":
            pass
        case _:
            print("not a valid input")


if __name__ == "__main__":
    main()
