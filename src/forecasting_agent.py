"""
Core demand forecasting agent with machine learning capabilities.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DemandForecastingAgent:
    """
    AI Agent for demand forecasting and dynamic replenishment.
    Continuously learns from historical data and adapts predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.forecast_accuracy = {}
        self.learning_history = []
        
    def prepare_features(self, sales_df, external_df=None):
        """
        Engineer features from sales and external data.
        
        Args:
            sales_df: Sales data with date, sku_id, sales columns
            external_df: External factors data
            
        Returns:
            Feature DataFrame
        """
        df = sales_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Time-based features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Holiday indicators
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        
        # Lag features (by SKU)
        for sku in df['sku_id'].unique():
            mask = df['sku_id'] == sku
            df.loc[mask, 'sales_lag_7'] = df.loc[mask, 'sales'].shift(7)
            df.loc[mask, 'sales_lag_14'] = df.loc[mask, 'sales'].shift(14)
            df.loc[mask, 'sales_lag_30'] = df.loc[mask, 'sales'].shift(30)
            
            # Rolling statistics
            df.loc[mask, 'sales_rolling_7'] = df.loc[mask, 'sales'].rolling(7, min_periods=1).mean()
            df.loc[mask, 'sales_rolling_30'] = df.loc[mask, 'sales'].rolling(30, min_periods=1).mean()
            df.loc[mask, 'sales_std_7'] = df.loc[mask, 'sales'].rolling(7, min_periods=1).std()
        
        # Merge external factors if provided
        if external_df is not None:
            external_df['date'] = pd.to_datetime(external_df['date'])
            df = df.merge(external_df, on='date', how='left')
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)
        
        return df
    
    def train_model(self, sku_id, sales_df, external_df=None):
        """
        Train forecasting model for a specific SKU.
        
        Args:
            sku_id: SKU identifier
            sales_df: Historical sales data
            external_df: External factors data
            
        Returns:
            Trained model and metrics
        """
        # Prepare data
        df = self.prepare_features(sales_df, external_df)
        sku_data = df[df['sku_id'] == sku_id].copy()
        
        if len(sku_data) < 30:
            raise ValueError(f"Insufficient data for {sku_id}. Need at least 30 records.")
        
        # Define features and target
        feature_cols = [
            'dayofweek', 'day', 'month', 'quarter', 'dayofyear', 'weekofyear',
            'is_weekend', 'is_holiday_season', 'is_summer',
            'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
            'sales_rolling_7', 'sales_rolling_30', 'sales_std_7'
        ]
        
        # Add external factors if available
        if external_df is not None:
            external_cols = ['is_holiday', 'promotion_active']
            feature_cols.extend([col for col in external_cols if col in sku_data.columns])
        
        X = sku_data[feature_cols]
        y = sku_data['sales']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        y_pred = model.predict(X_test_scaled)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        
        # Store model and scaler
        self.models[sku_id] = model
        self.scalers[sku_id] = scaler
        self.forecast_accuracy[sku_id] = {
            'train_score': train_score,
            'test_score': test_score,
            'mape': mape
        }
        
        # Log learning history
        self.learning_history.append({
            'sku_id': sku_id,
            'timestamp': pd.Timestamp.now(),
            'mape': mape,
            'test_score': test_score
        })
        
        return model, self.forecast_accuracy[sku_id]
    
    def predict_demand(self, sku_id, future_dates, external_factors=None):
        """
        Predict future demand for a SKU.
        
        Args:
            sku_id: SKU identifier
            future_dates: List of future dates to predict
            external_factors: DataFrame with external factors for future dates
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if sku_id not in self.models:
            raise ValueError(f"No trained model for {sku_id}. Train model first.")
        
        model = self.models[sku_id]
        scaler = self.scalers[sku_id]
        
        # Create feature DataFrame for future dates
        future_df = pd.DataFrame({'date': pd.to_datetime(future_dates)})
        future_df['dayofweek'] = future_df['date'].dt.dayofweek
        future_df['day'] = future_df['date'].dt.day
        future_df['month'] = future_df['date'].dt.month
        future_df['quarter'] = future_df['date'].dt.quarter
        future_df['dayofyear'] = future_df['date'].dt.dayofyear
        future_df['weekofyear'] = future_df['date'].dt.isocalendar().week
        future_df['is_weekend'] = future_df['dayofweek'].isin([5, 6]).astype(int)
        future_df['is_holiday_season'] = future_df['month'].isin([11, 12]).astype(int)
        future_df['is_summer'] = future_df['month'].isin([6, 7, 8]).astype(int)
        
        # Use placeholder values for lag features
        recent_avg = 100
        future_df['sales_lag_7'] = recent_avg
        future_df['sales_lag_14'] = recent_avg
        future_df['sales_lag_30'] = recent_avg
        future_df['sales_rolling_7'] = recent_avg
        future_df['sales_rolling_30'] = recent_avg
        future_df['sales_std_7'] = recent_avg * 0.2
        
        # Merge external factors
        if external_factors is not None:
            external_factors['date'] = pd.to_datetime(external_factors['date'])
            future_df = future_df.merge(external_factors, on='date', how='left')
        
        # Ensure all required features are present
        feature_cols = [col for col in future_df.columns if col != 'date']
        future_df = future_df.fillna(0)
        
        # Make predictions
        X_future = future_df[feature_cols]
        X_future_scaled = scaler.transform(X_future)
        predictions = model.predict(X_future_scaled)
        
        # Calculate confidence intervals
        mape = self.forecast_accuracy[sku_id]['mape']
        margin = predictions * (mape / 100) * 1.96
        
        result_df = pd.DataFrame({
            'date': future_dates,
            'sku_id': sku_id,
            'predicted_demand': np.maximum(0, predictions).astype(int),
            'lower_bound': np.maximum(0, predictions - margin).astype(int),
            'upper_bound': np.maximum(0, predictions + margin).astype(int),
            'confidence': 95
        })
        
        return result_df
    
    def calculate_dynamic_reorder(self, sku_id, forecast_df, current_stock, lead_time_days):
        """
        Calculate dynamic reorder point and quantity.
        
        Args:
            sku_id: SKU identifier
            forecast_df: Forecast predictions DataFrame
            current_stock: Current stock level
            lead_time_days: Lead time in days
            
        Returns:
            Dict with reorder recommendations
        """
        # Calculate expected demand during lead time
        lead_time_demand = forecast_df['predicted_demand'].iloc[:lead_time_days].sum()
        
        # Safety stock based on forecast uncertainty
        demand_std = forecast_df['upper_bound'].iloc[:lead_time_days].std()
        safety_stock = int(demand_std * 1.65)
        
        # Reorder point
        reorder_point = lead_time_demand + safety_stock
        
        # Economic order quantity (simplified)
        avg_daily_demand = forecast_df['predicted_demand'].mean()
        reorder_quantity = int(avg_daily_demand * lead_time_days * 2)
        
        # Determine if reorder is needed
        needs_reorder = current_stock <= reorder_point
        
        # Calculate days until stockout
        cumulative_demand = 0
        days_until_stockout = 0
        for i, demand in enumerate(forecast_df['predicted_demand']):
            cumulative_demand += demand
            if cumulative_demand >= current_stock:
                days_until_stockout = i + 1
                break
        
        if days_until_stockout == 0:
            days_until_stockout = len(forecast_df)
        
        return {
            'sku_id': sku_id,
            'current_stock': current_stock,
            'reorder_point': int(reorder_point),
            'reorder_quantity': reorder_quantity,
            'safety_stock': safety_stock,
            'lead_time_demand': int(lead_time_demand),
            'needs_reorder': needs_reorder,
            'days_until_stockout': days_until_stockout,
            'recommended_order_date': pd.Timestamp.now() if needs_reorder else None,
            'urgency': 'HIGH' if days_until_stockout <= lead_time_days else 'MEDIUM' if needs_reorder else 'LOW'
        }
    
    def get_model_performance(self):
        """Get performance metrics for all trained models."""
        return pd.DataFrame(self.forecast_accuracy).T
    
    def adapt_and_retrain(self, sku_id, new_data, external_df=None):
        """
        Adapt model based on new data (continuous learning).
        
        Args:
            sku_id: SKU identifier
            new_data: New sales data
            external_df: External factors
        """
        print(f"Retraining model for {sku_id} with new data...")
        return self.train_model(sku_id, new_data, external_df)