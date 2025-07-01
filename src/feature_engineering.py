# src/feature_engineering.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ====================
# Custom Transformers
# ====================

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X.drop(columns=[self.datetime_col])


class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby("CustomerId").agg({
            "Amount": ["sum", "mean", "count", "std"]
        })
        agg_df.columns = ['total_amount', 'avg_amount', 'tx_count', 'std_amount']
        agg_df = agg_df.reset_index()
        return agg_df


# ====================
# Feature Engineering Pipeline
# ====================

def get_feature_pipeline(categorical_cols, numerical_cols):
    categorical_pipe = Pipeline([
        ("impute_cat", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_pipe = Pipeline([
        ("impute_num", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ])

    full_preprocessor = ColumnTransformer([
        ("cat", categorical_pipe, categorical_cols),
        ("num", numeric_pipe, numerical_cols)
    ])

    return full_preprocessor


# ====================
# Wrapper Function
# ====================

def run_feature_engineering(raw_df):
    # Step 0: Drop ID columns that should not go into the model
    id_cols_to_drop = ['TransactionId', 'BatchId', 'SubscriptionId', 'AccountId']
    raw_df = raw_df.drop(columns=id_cols_to_drop, errors='ignore')
    # Step 1: Extract date features
    datetime_transformer = DateTimeFeatureExtractor()
    df_with_dates = datetime_transformer.transform(raw_df)

    # Step 2: Create aggregates
    agg_transformer = CustomerAggregateFeatures()
    agg_features = agg_transformer.transform(raw_df)

    # Step 3: Join with date features and select final columns
    final_df = pd.merge(df_with_dates, agg_features, on='CustomerId', how='left')

    # Step 4: Define columns
    cat_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                'ProductCategory', 'ChannelId', 'PricingStrategy']
    num_cols = ['Value', 'transaction_hour', 'transaction_day', 'transaction_month',
                'transaction_year', 'total_amount', 'avg_amount', 'tx_count', 'std_amount']

    # Step 5: Apply pipeline
    feature_pipeline = get_feature_pipeline(cat_cols, num_cols)
    transformed_array = feature_pipeline.fit_transform(final_df)

    # Optionally return feature names
    ohe = feature_pipeline.named_transformers_['cat']['onehot']
    feature_names = ohe.get_feature_names_out(cat_cols)
    all_feature_names = list(feature_names) + num_cols
    y = raw_df['is_high_risk']

    return transformed_array, all_feature_names, final_df
