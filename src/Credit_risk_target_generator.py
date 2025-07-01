# src/credit_risk_target_generator.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress KMeans deprecation warning for n_init if it pops up repeatedly
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

class CreditRiskTargetGenerator:
    """
    A class to generate a 'is_high_risk' target variable based on RFM analysis
    and K-Means clustering for credit risk prediction.
    """
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initializes the CreditRiskTargetGenerator.

        Args:
            n_clusters (int): The number of clusters for KMeans. Default is 3.
            random_state (int): Random state for KMeans to ensure reproducibility.
                                Default is 42.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10) # n_init is important for newer sklearn versions
        self.high_risk_cluster_index = None
        self.rfm_df = None # To store the calculated RFM values and clusters

    def calculate_rfm(self, df: pd.DataFrame, customer_id_col: str = 'CustomerId',
                      transaction_time_col: str = 'TransactionStartTime',
                      monetary_value_col: str = 'Value') -> pd.DataFrame:
        """
        Calculates Recency, Frequency, and Monetary (RFM) values for each customer.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.
            customer_id_col (str): Name of the customer ID column.
            transaction_time_col (str): Name of the transaction timestamp column.
            monetary_value_col (str): Name of the monetary value column.

        Returns:
            pd.DataFrame: A DataFrame with CustomerId and their RFM values.
        """
        # Ensure TransactionStartTime is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df[transaction_time_col]):
            df[transaction_time_col] = pd.to_datetime(df[transaction_time_col])

        # Define snapshot date as the day after the last transaction
        snapshot_date = df[transaction_time_col].max() + pd.Timedelta(days=1)
        print(f"Calculating RFM... Snapshot Date: {snapshot_date}")

        rfm_df = df.groupby(customer_id_col).agg(
            Recency=(transaction_time_col, lambda date: (snapshot_date - date.max()).days),
            Frequency=(transaction_time_col, 'nunique'), # Count unique transactions
            Monetary=(monetary_value_col, 'sum') # Sum of monetary value
        ).reset_index()

        self.rfm_df = rfm_df # Store for later use if needed
        return rfm_df

    def cluster_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Scales RFM features and applies K-Means clustering to segment customers.

        Args:
            rfm_df (pd.DataFrame): DataFrame containing RFM values.

        Returns:
            pd.DataFrame: The RFM DataFrame with an added 'Cluster' column.
        """
        # Select RFM features for scaling and clustering
        rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']]

        # Scale features
        print("Scaling RFM features...")
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        rfm_df_scaled = pd.DataFrame(rfm_scaled, columns=rfm_features.columns, index=rfm_features.index)

        # Apply K-Means clustering
        print(f"Applying K-Means clustering with {self.n_clusters} clusters...")
        rfm_df['Cluster'] = self.kmeans.fit_predict(rfm_scaled)

        # Calculate and display cluster centers in original scale
        cluster_centers_original_scale = pd.DataFrame(
            self.scaler.inverse_transform(self.kmeans.cluster_centers_),
            columns=rfm_features.columns
        )
        print("\nCluster Centers (Original Scale):")
        print(cluster_centers_original_scale)

        # Automatically identify high-risk cluster (highest Recency, lowest Frequency, lowest Monetary)
        # This assumes a standard RFM interpretation where lower F/M and higher R mean worse.
        # It's always good to visually inspect for confirmation.
        self.high_risk_cluster_index = cluster_centers_original_scale.iloc[
            (cluster_centers_original_scale['Recency'].rank(ascending=False) +
             cluster_centers_original_scale['Frequency'].rank(ascending=True) +
             cluster_centers_original_scale['Monetary'].rank(ascending=True)).idxmin() # idxmin finds the cluster with the lowest sum of ranks (i.e., worst RFM)
        ].name
        print(f"\nIdentified High-Risk Cluster Index: {self.high_risk_cluster_index}")

        return rfm_df

    def assign_high_risk_label(self, rfm_df_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns the 'is_high_risk' binary label based on the identified high-risk cluster.

        Args:
            rfm_df_clustered (pd.DataFrame): DataFrame with RFM values and 'Cluster' column.

        Returns:
            pd.DataFrame: The DataFrame with the 'is_high_risk' column added.
        """
        if self.high_risk_cluster_index is None:
            raise ValueError("High-risk cluster not identified. Run cluster_customers() first.")

        print(f"Assigning 'is_high_risk' label. High-risk customers are in Cluster {self.high_risk_cluster_index}.")
        rfm_df_clustered['is_high_risk'] = (rfm_df_clustered['Cluster'] == self.high_risk_cluster_index).astype(int)

        print("\nDistribution of 'is_high_risk' label:")
        print(rfm_df_clustered['is_high_risk'].value_counts())

        return rfm_df_clustered[['CustomerId', 'is_high_risk']] # Return only what's needed for merging

    def generate_target(self, df: pd.DataFrame, customer_id_col: str = 'CustomerId',
                        transaction_time_col: str = 'TransactionStartTime',
                        monetary_value_col: str = 'Value') -> pd.DataFrame:
        """
        Orchestrates the entire process of generating the 'is_high_risk' target variable
        and merging it back into the main DataFrame.

        Args:
            df (pd.DataFrame): The main input DataFrame containing all transaction data.
            customer_id_col (str): Name of the customer ID column.
            transaction_time_col (str): Name of the transaction timestamp column.
            monetary_value_col (str): Name of the monetary value column.

        Returns:
            pd.DataFrame: The original DataFrame with the 'is_high_risk' column added.
        """
        print("Starting target variable generation (Task 4)...")

        # 1. Calculate RFM
        rfm_data = self.calculate_rfm(df.copy(), customer_id_col, transaction_time_col, monetary_value_col)

        # 2. Cluster Customers
        rfm_data_clustered = self.cluster_customers(rfm_data)

        # 3. Assign High-Risk Label
        customer_risk_mapping = self.assign_high_risk_label(rfm_data_clustered)

        # 4. Integrate the Target Variable back into the original DataFrame
        print("Merging 'is_high_risk' into the main DataFrame...")
        df_with_target = df.merge(customer_risk_mapping, on=customer_id_col, how='left')

        print("Target variable generation complete.")
        return df_with_target

# Example Usage (for testing the .py file directly, or for reference)
if __name__ == "__main__":
    # Create a dummy DataFrame for testing
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
        'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C1', 'C2', 'C3', 'C1', 'C2', 'C4'],
        'TransactionStartTime': ['2024-01-01', '2024-01-05', '2024-01-10', '2024-02-01',
                                 '2024-02-15', '2024-03-01', '2024-03-05', '2024-04-01',
                                 '2024-04-10', '2024-01-02'],
        'Value': [100, 50, 200, 30, 120, 150, 40, 80, 250, 10]
    }
    sample_df = pd.DataFrame(data)

    print("Sample DataFrame:")
    print(sample_df)

    # Initialize the generator
    target_generator = CreditRiskTargetGenerator(n_clusters=3, random_state=42)

    # Generate the target variable
    df_with_risk = target_generator.generate_target(sample_df.copy()) # Use .copy() to avoid modifying original sample_df

    print("\nDataFrame with 'is_high_risk' column:")
    print(df_with_risk.head())
    print(df_with_risk['is_high_risk'].value_counts())
    print("\nRFM data used for clustering:")
    print(target_generator.rfm_df)