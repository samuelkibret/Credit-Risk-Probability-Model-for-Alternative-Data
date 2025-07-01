import pytest
from src.feature_engineering import DateTimeFeatureExtractor

def test_datetime_extractor():
    import pandas as pd
    df = pd.DataFrame({"TransactionStartTime": ["2023-01-01 10:00:00"]})
    transformer = DateTimeFeatureExtractor()
    result = transformer.transform(df)
    assert "transaction_hour" in result.columns
    assert result.loc[0, "transaction_hour"] == 10
