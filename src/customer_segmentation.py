"""
Customer demand segmentation and analysis.
RFM analysis, cohort analysis, and demand pattern clustering.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

class CustomerSegmentationAnalyzer:
    """Analyze customer demand patterns and segments."""
    
    def __init__(self):
        self.segments = {}
        self.rfm_data = None
        self.customer_profiles = {}
        
    def perform_rfm_analysis(self, sales_df, customer_id_col='customer_id', 
                             date_col='date', revenue_col='revenue'):
        """
        Perform RFM (Recency, Frequency, Monetary) analysis.
        
        Args:
            sales_df: Sales DataFrame with customer transactions
            customer_id_col: Column name for customer ID
            date_col: Column name for transaction date
            revenue_col: Column name for revenue
            
        Returns:
            DataFrame with RFM scores and segments
        """
        df = sales_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate RFM metrics
        current_date = df[date_col].max()
        
        rfm = df.groupby(customer_id_col).agg({
            date_col: lambda x: (current_date - x.max()).days,  # Recency
            'units_sold': 'count',  # Frequency
            revenue_col: 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = [customer_id_col, 'recency', 'frequency', 'monetary']
        
        # Calculate RFM scores (1-5 scale)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        
        # Convert to numeric
        rfm['r_score'] = pd.to_numeric(rfm['r_score'])
        rfm['f_score'] = pd.to_numeric(rfm['f_score'])
        rfm['m_score'] = pd.to_numeric(rfm['m_score'])
        
        # Calculate overall RFM score
        rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
        
        # Assign segments
        rfm['segment'] = rfm['rfm_score'].apply(self._assign_rfm_segment)
        
        self.rfm_data = rfm
        
        return rfm
    
    def _assign_rfm_segment(self, score):
        """Assign customer segment based on RFM score."""
        if score >= 13:
            return 'Champions'
        elif score >= 11:
            return 'Loyal Customers'
        elif score >= 9:
            return 'Potential Loyalists'
        elif score >= 7:
            return 'Recent Customers'
        elif score >= 5:
            return 'At Risk'
        else:
            return 'Lost Customers'
    
    def perform_demand_clustering(self, sales_df, n_clusters=4):
        """
        Cluster customers based on demand patterns.
        
        Args:
            sales_df: Sales DataFrame
            n_clusters: Number of customer clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        # Aggregate customer-level metrics
        customer_metrics = sales_df.groupby('customer_id').agg({
            'units_sold': ['sum', 'mean', 'std'],
            'revenue': ['sum', 'mean'],
            'date': ['count', 'min', 'max']
        }).reset_index()
        
        customer_metrics.columns = ['customer_id', 'total_units', 'avg_units', 'std_units',
                                    'total_revenue', 'avg_revenue', 'num_transactions',
                                    'first_purchase', 'last_purchase']
        
        # Calculate additional features
        customer_metrics['purchase_frequency'] = customer_metrics['num_transactions'] / \
            ((pd.to_datetime(customer_metrics['last_purchase']) - 
              pd.to_datetime(customer_metrics['first_purchase'])).dt.days + 1)
        
        customer_metrics['avg_order_value'] = customer_metrics['total_revenue'] / customer_metrics['num_transactions']
        customer_metrics['demand_variability'] = customer_metrics['std_units'] / (customer_metrics['avg_units'] + 1)
        
        # Select features for clustering
        feature_cols = ['total_units', 'avg_units', 'total_revenue', 
                       'num_transactions', 'purchase_frequency', 'avg_order_value']
        
        X = customer_metrics[feature_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer_metrics['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Assign cluster names based on characteristics
        cluster_profiles = customer_metrics.groupby('cluster').agg({
            'total_revenue': 'mean',
            'num_transactions': 'mean',
            'purchase_frequency': 'mean'
        })
        
        # Name clusters
        cluster_names = {}
        for cluster_id in range(n_clusters):
            profile = cluster_profiles.loc[cluster_id]
            
            if profile['total_revenue'] > cluster_profiles['total_revenue'].median() * 1.5:
                if profile['purchase_frequency'] > cluster_profiles['purchase_frequency'].median():
                    cluster_names[cluster_id] = 'High-Value Frequent'
                else:
                    cluster_names[cluster_id] = 'High-Value Occasional'
            elif profile['purchase_frequency'] > cluster_profiles['purchase_frequency'].median():
                cluster_names[cluster_id] = 'Regular Buyers'
            else:
                cluster_names[cluster_id] = 'Infrequent Buyers'
        
        customer_metrics['cluster_name'] = customer_metrics['cluster'].map(cluster_names)
        
        return customer_metrics
    
    def analyze_seasonal_customer_behavior(self, sales_df):
        """Analyze how customer demand varies by season."""
        df = sales_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Seasonal patterns by customer segment
        if self.rfm_data is not None:
            df = df.merge(self.rfm_data[['customer_id', 'segment']], 
                         on='customer_id', how='left')
            
            seasonal_analysis = df.groupby(['season', 'segment']).agg({
                'units_sold': 'sum',
                'revenue': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            seasonal_analysis.columns = ['season', 'segment', 'total_units', 
                                         'total_revenue', 'unique_customers']
            
            return seasonal_analysis
        else:
            # Simple seasonal analysis without segments
            seasonal_analysis = df.groupby('season').agg({
                'units_sold': 'sum',
                'revenue': 'sum',
                'customer_id': 'nunique'
            }).reset_index()
            
            return seasonal_analysis
    
    def predict_customer_churn_risk(self, sales_df, days_threshold=90):
        """
        Identify customers at risk of churning.
        
        Args:
            sales_df: Sales DataFrame
            days_threshold: Days of inactivity to consider at-risk
            
        Returns:
            DataFrame with churn risk customers
        """
        df = sales_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Get last purchase date for each customer
        last_purchase = df.groupby('customer_id')['date'].max().reset_index()
        last_purchase.columns = ['customer_id', 'last_purchase_date']
        
        # Calculate days since last purchase
        current_date = df['date'].max()
        last_purchase['days_since_purchase'] = (current_date - last_purchase['last_purchase_date']).dt.days
        
        # Get customer lifetime value
        customer_value = df.groupby('customer_id').agg({
            'revenue': 'sum',
            'units_sold': 'sum',
            'date': 'count'
        }).reset_index()
        customer_value.columns = ['customer_id', 'lifetime_value', 'total_units', 'num_purchases']
        
        # Merge
        churn_risk = last_purchase.merge(customer_value, on='customer_id')
        
        # Calculate churn risk
        churn_risk['at_risk'] = churn_risk['days_since_purchase'] > days_threshold
        churn_risk['risk_level'] = churn_risk['days_since_purchase'].apply(
            lambda x: 'High' if x > days_threshold * 1.5 
            else 'Medium' if x > days_threshold 
            else 'Low'
        )
        
        # Calculate potential revenue loss
        churn_risk['avg_order_value'] = churn_risk['lifetime_value'] / churn_risk['num_purchases']
        churn_risk['potential_annual_loss'] = churn_risk['avg_order_value'] * (365 / 30)  # Assume monthly purchases
        
        # Return high-risk customers
        at_risk_customers = churn_risk[churn_risk['at_risk']].sort_values('lifetime_value', ascending=False)
        
        return at_risk_customers
    
    def generate_segment_forecasts(self, sales_df, forecast_horizon=30):
        """
        Generate demand forecasts by customer segment.
        
        Args:
            sales_df: Sales DataFrame
            forecast_horizon: Days to forecast
            
        Returns:
            Dict with forecasts by segment
        """
        if self.rfm_data is None:
            raise ValueError("Run perform_rfm_analysis first")
        
        # Merge segment info
        df = sales_df.merge(self.rfm_data[['customer_id', 'segment']], on='customer_id', how='left')
        
        # Calculate daily demand by segment
        segment_daily = df.groupby(['date', 'segment'])['units_sold'].sum().reset_index()
        
        segment_forecasts = {}
        
        for segment in df['segment'].unique():
            segment_data = segment_daily[segment_daily['segment'] == segment].copy()
            
            # Simple moving average forecast
            recent_avg = segment_data.tail(30)['units_sold'].mean()
            
            segment_forecasts[segment] = {
                'forecast_daily_avg': recent_avg,
                'forecast_total': recent_avg * forecast_horizon,
                'historical_avg': segment_data['units_sold'].mean(),
                'trend': 'increasing' if recent_avg > segment_data['units_sold'].mean() else 'decreasing'
            }
        
        return segment_forecasts