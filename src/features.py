import datetime
from re import I
from tabnanny import check

import polars as pl
from pl_utils import pl_strtime2timeobj

from tests.utils import check_has_null


def get_features(
    df: pl.DataFrame,
    client: pl.DataFrame,
    targets: pl.DataFrame,
    historical_weather: pl.DataFrame,
    forecast_weather: pl.DataFrame,
    electricity_prices: pl.DataFrame,
    gas_prices: pl.DataFrame,
):
    """
    Generate features for energy behavior prediction.

    Args:
        df (pl.DataFrame): The main train/test dataset, assumed that the target has already been dropped
        client (pl.DataFrame): Client dataset.
        targets (pl.DataFrame): The revealed target dataset.
        historical_weather (pl.DataFrame): Historical weather dataset.
        forecast_weather (pl.DataFrame): Forecast weather dataset.
        electricity_prices (pl.DataFrame): Electricity prices dataset.
        gas_prices (pl.DataFrame): Gas prices dataset.

    Returns:
        feat (pl.DataFrame): The generated features.
    """

    # assert there is no missing data in each dataset
    assert df.is_null().sum().sum() == 0
    assert client.is_null().sum().sum() == 0
    assert targets.is_null().sum().sum() == 0
    assert historical_weather.is_null().sum().sum() == 0
    assert forecast_weather.is_null().sum().sum() == 0
    assert electricity_prices.is_null().sum().sum() == 0
    assert gas_prices.is_null().sum().sum() == 0

    # initialize features
    feat = pl.DataFrame()

    utility_features = get_utility_features(df, electricity_prices, gas_prices)
    forecast_weather_features = get_forecast_weather_features(df, forecast_weather)
    historical_weather_features = get_historical_weather_features(
        df, historical_weather
    )
    client_features = get_client_features(df, client)

    feat = feat.join(utility_features)
    feat = feat.join(forecast_weather_features)
    feat = feat.join(historical_weather_features)
    feat = feat.join(client_features)

    # assertions for feat
    assert feat.is_null().sum().sum() == 0
    return feat


def get_utility_features(df, electricity_prices, gas_prices):
    """
    Generate utility features.

    Args:
        df (pl.DataFrame): The main train/test dataset, assumed that the target has already been dropped
        electricity_prices (pl.DataFrame): Electricity prices dataset, including the historical electricity_prices
        gas_prices (pl.DataFrame): Gas prices dataset, including the historical gas_prices

    Returns:
        feat (pl.DataFrame): The generated features.
    """
    # get the columns of df
    date_differences = [1, 2, 3, 4, 5, 6, 7]  # TODO: can add more date differences

    tmp_df = df.clone()
    tmp_gas_prices = gas_prices.clone()

    # convert the datetime column to datetime type
    # NOTE: I am following the "open-closed principle"!
    tmp_df = pl_strtime2timeobj(tmp_df, "datetime")
    tmp_df = tmp_df.with_columns(tmp_df["datetime_object"].dt.date().alias("date"))
    tmp_gas_prices = pl_strtime2timeobj(tmp_gas_prices, "origin_date", "%Y-%m-%d")
    tmp_gas_prices = tmp_gas_prices.with_columns(
        tmp_gas_prices["origin_date_object"].dt.date()
    )

    assert (
        check_has_null(tmp_gas_prices.null_count()) == False
    ), "tmp_gas_prices does not have any null values"

    # iterating through the date differences and get the date differences from the date column
    tmp_df = tmp_df.with_columns(
        [
            (tmp_df["date"] - pl.duration(days=num_day)).alias(f"date_{num_day}_before")
            for num_day in date_differences
        ]
    )

    # join the gas_prices to the tmp_df using the date
    for date_difference in date_differences:
        tmp_df = tmp_df.join(
            tmp_gas_prices,
            left_on=f"date_{date_difference}_before",
            right_on="origin_date_object",
            how="left",
        )
        duplicate_columns_to_drop = [
            column for column in tmp_df.columns if "_right" in column
        ]
        tmp_df = tmp_df.drop(duplicate_columns_to_drop)
    # ? What is the beginning of the gas price though?
    # (1/29/24) at the end, there was missing data which makes sense because 5-30 does not match any data for the gas price, which
    # ends at 5-29
    tmp_df = tmp_df.fill_null(strategy="mean")

    print(
        electricity_prices.group_by("data_block_id").agg(pl.col("euros_per_mwh").mean())
    )

    # ? Does this apply to the test features as well?
    # the null values for the data block id comes from the data where the data_block_id is 0
    tmp_df = tmp_df.join(
        electricity_prices.group_by("data_block_id")
        .agg(pl.col("euros_per_mwh").mean())
        .rename({"euros_per_mwh": "daily_euros_per_mwh"}),
        on="data_block_id",
        how="left",
    )

    print(tmp_df.null_count())
    # print the column of tmp_df where there are null values
    print(tmp_df.filter(pl.col("daily_euros_per_mwh").is_null()))
    # print(tmp_df.with_columns(pl.all().is_null().name.suffix("_isnull")))

    tmp_df = tmp_df.fill_null(strategy="mean")

    assert (
        check_has_null(tmp_df.null_count()) == False
    ), "tmp_df does not have any null values"

    return tmp_df


def get_forecast_weather_features(df, forecast_weather):
    """
    Generate forecast weather features.

    Args:
        df (pl.DataFrame): The main train/test dataset, assumed that the target has already been dropped
        forecast_weather (pl.DataFrame): Forecast weather dataset.

    Returns:
        feat (pl.DataFrame): The generated features.
    """
    # need to parse the json file containing the longitude & latitude to county

    # join the forecast weather to the to the county to get the county

    #


def get_historical_weather_features(df, historical_weather):
    """
    Generate historical weather features.

    Args:
        df (pl.DataFrame): The main train/test dataset, assumed that the target has already been dropped
        historical_weather (pl.DataFrame): Historical weather dataset.

    Returns:
        feat (pl.DataFrame): The generated features.
    """
    pass


def get_client_features(df, client):
    """
    Generate client features.

    Args:
        df (pl.DataFrame): The main train/test dataset, assumed that the target has already been dropped
        client (pl.DataFrame): Client dataset.

    Returns:
        feat (pl.DataFrame): The generated features.
    """
    pass
