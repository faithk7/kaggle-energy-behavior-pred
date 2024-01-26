def check_has_null(df):
    """
    Check if the given DataFrame has any null values.

    Args:
        df (polar.DataFrame): The DataFrame to check.

    Returns:
        bool: True if the DataFrame has null values, False otherwise.
    """
    pd_df = df.to_pandas()
    return pd_df.sum().sum() != 0
