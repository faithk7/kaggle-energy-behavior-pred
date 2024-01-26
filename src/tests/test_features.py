import sys

import polars as pl
import pytest

sys.path.insert(
    0, ".."
)  # add parent directory to path so we can import the `src` package

from utils import check_has_null

# import function from the `features.py` module in the `src` package
from src.features import get_utility_features

# Sample data for testing
train_df = pl.read_csv("../data/train.csv")
electricity_prices_df = pl.read_csv("../data/electricity_prices.csv")
gas_prices_df = pl.read_csv("../data/gas_prices.csv")


# Define test cases
@pytest.fixture
def sample_data():
    return train_df.clone(), electricity_prices_df.clone(), gas_prices_df.clone()


def test_get_utility_features(sample_data):
    # Unpack the sample data
    df, electricity_prices, gas_prices = sample_data

    # Call the function to generate features
    features = get_utility_features(df, electricity_prices, gas_prices)

    null_cnt = features.null_count()
    assert check_has_null(null_cnt) == False

    # Check if the returned dataframe has the expected columns
    # expected_columns = [
    #     "county",
    #     "product_type",
    #     "datetime",
    #     "data_block_id",
    #     "date",
    #     "date_2_before",
    #     "lowest_price_per_mwh",
    #     "highest_price_per_mwh",
    # ]
    # assert set(features.columns) == set(expected_columns)

    # Check if there are no missing values in the resulting dataframe
    # assert features.isnull().sum().sum() == 0

    # Add more specific tests based on your function's logic
    # For example, test that date differences are calculated correctly, and data is joined properly.

    # Test any additional logic or conditions in your function as needed


# Run the tests
if __name__ == "__main__":
    pytest.main()
