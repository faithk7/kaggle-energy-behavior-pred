import sys

import polars as pl
import pytest
from utils import check_has_null

sys.path.insert(
    0, ".."
)  # add parent directory to path so we can import the `src` package

# import function from the `features.py` module in the `src` package
from src.features import get_utility_features

# Sample data for testing
train_df = pl.read_csv("../data/train.csv")
electricity_prices_df = pl.read_csv("../data/electricity_prices.csv")
gas_prices_df = pl.read_csv("../data/gas_prices.csv")


# Define test cases
@pytest.fixture
def sample_data():
    return train_df.drop_nulls(), electricity_prices_df.clone(), gas_prices_df.clone()


def test_get_utility_features(sample_data):
    # Unpack the sample data
    df, electricity_prices, gas_prices = sample_data

    # Call the function to generate features
    features = get_utility_features(df, electricity_prices, gas_prices)

    null_cnt = features.null_count()
    print(null_cnt)
    print(features.filter(features["lowest_price_per_mwh"].is_null())["datetime"])
    assert check_has_null(null_cnt) == False


# Run the tests
if __name__ == "__main__":
    pytest.main()
