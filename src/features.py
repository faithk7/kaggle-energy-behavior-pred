import datetime

import polars as pl

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
    date_differences = [2]  # TODO: can add more date differences

    tmp_df = df.clone()
    tmp_gas_prices = gas_prices.clone()

    # convert the datetime column to datetime type
    # NOTE: I am following the "open-closed principle"!
    tmp_df = tmp_df.with_columns(
        pl.col("datetime")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        .alias("datetime_object")
    )

    # get the date from the datetime column df
    tmp_df = tmp_df.with_columns(tmp_df["datetime_object"].dt.date().alias("date"))
    tmp_gas_prices = tmp_gas_prices.with_columns(
        pl.col("origin_date")
        .str.strptime(pl.Datetime, "%Y-%m-%d")
        .dt.date()
        .alias("origin_date_object")
    )

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

    print(tmp_df.head(10))

    # TODO: group the electricity_price by the data_block_id and get the mean of euros_per_mwh

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
    pass


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
