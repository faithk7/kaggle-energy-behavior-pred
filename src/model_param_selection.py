import catboost
from sklearn.metrics import mean_absolute_error

from data import DataStorage
from features import FeaturesGenerator


# TODO: can add model as a parameter as well, if you want to generalize more, you can add metrics as a parameter well
def train(features, target, num_fold):
    train_val_split = get_train_val_split_features(features, target, num_fold)
    val_scores = []
    fold_cnt = 1
    while True:
        try:
            train_feature, train_target, val_feature, val_target = next(train_val_split)
            model = catboost.CatBoostRegressor()
            model.fit(train_feature, train_target)

            y_pred = model.predict(val_feature)
            val_score = mean_absolute_error(val_target, y_pred)
            val_scores.append(val_score)
            print(f"Training fold {fold_cnt}: val_ame_score: {val_score}")
            fold_cnt += 1
        except StopIteration:
            print(f"Mean val_ame_score: {sum(val_scores) / len(val_scores)}")


# TODO: this can be added to the template
def get_train_val_split_features(features, target, num_fold):
    num_data = len(features)
    fold_size = (num_data - fold_size) // num_fold

    for end_index in range(fold_size, num_data - fold_size, fold_size):
        train_feature = features[:end_index]
        train_target = target[:end_index]
        val_feature = features[end_index : end_index + fold_size]
        val_target = target[end_index : end_index + fold_size]
        yield train_feature, train_target, val_feature, val_target


def main():
    select_model = "ctb"
    data_storage = DataStorage()

    features = FeaturesGenerator(data_storage=data_storage).generate_features(
        data_storage.df_data
    )

    N_FOLDS = 10

    match select_model:
        case "ctb":
            train(features, data_storage.df_target, N_FOLDS)
        case "lgb":
            pass
        case "xgb":
            pass
        case _:
            print("not a valid input")


if __name__ == "__main__":
    main()
