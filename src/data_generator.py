"""
Generate realistic synthetic sales data for demand forecasting demonstration.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SalesDataGenerator:
    """Generate synthetic sales data with realistic patterns."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
    def generate_sales_data(self, start_date='2022-01-01', periods=730, n_skus=10):
        """
        Generate synthetic sales data with realistic patterns.
        
        Args:
            start_date: Starting date for the time series
            periods: Number of days to generate
            n_skus: Number of SKUs to generate
            
        Returns:
            DataFrame with sales data
        """
        date_range = pd.date_range(start=start_date, periods=periods, freq='D')
        
        data = []
        sku_categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
        
        for sku_id in range(1, n_skus + 1):
            category = np.random.choice(sku_categories)
            base_demand = np.random.uniform(50, 200)
            
            for date in date_range:
                # Trend component
                trend = base_demand * (1 + 0.0005 * (date - date_range[0]).days)
                
                # Seasonal component (yearly)
                seasonal = 30 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                
                # Weekly pattern (weekends higher for some categories)
                weekly = 20 if date.dayofweek >= 5 and category in ['Clothing', 'Electronics'] else 0
                
                # Holiday boost
                if date.month == 12 or date.month == 11:
                    holiday_boost = 50
                elif date.month in [6, 7]:  # Summer
                    holiday_boost = 30
                else:
                    holiday_boost = 0
                    
                # Random noise
                noise = np.random.normal(0, 15)
                
                # Weather impact (simplified - affects certain categories)
                if category == 'Clothing':
                    weather_impact = 20 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                else:
                    weather_impact = 0
                
                # Calculate final demand
                demand = max(0, trend + seasonal + weekly + holiday_boost + weather_impact + noise)
                
                # Add stockout events (randomly)
                stockout = np.random.random() < 0.02  # 2% chance of stockout
                
                data.append({
                    'date': date,
                    'sku_id': f'SKU_{sku_id:03d}',
                    'category': category,
                    'sales': int(demand) if not stockout else 0,
                    'stockout': stockout,
                    'price': round(np.random.uniform(10, 100), 2),
                    'cost': round(np.random.uniform(5, 50), 2)
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_external_factors(self, sales_df):
        """
        Generate external factors data aligned with sales data.
        
        Args:
            sales_df: Sales DataFrame with date column
            
        Returns:
            DataFrame with external factors
        """
        dates = sales_df['date'].unique()
        
        external_data = []
        for date in dates:
            # Temperature (simplified seasonal pattern)
            temp = 60 + 30 * np.sin(2 * np.pi * pd.Timestamp(date).dayofyear / 365.25)
            temp += np.random.normal(0, 5)
            
            # Economic indicator (GDP growth proxy)
            economic_index = 100 + 10 * np.sin(2 * np.pi * pd.Timestamp(date).dayofyear / 365.25)
            economic_index += np.random.normal(0, 2)
            
            # Promotional activity
            promo = np.random.choice([0, 1], p=[0.8, 0.2])
            
            external_data.append({
                'date': date,
                'temperature': round(temp, 1),
                'economic_index': round(economic_index, 2),
                'promotion_active': promo,
                'competitor_price_index': round(np.random.uniform(90, 110), 2)
            })
        
        return pd.DataFrame(external_data)
    
    def generate_inventory_data(self, sales_df):
        """
        Generate current inventory levels for each SKU.
        
        Args:
            sales_df: Sales DataFrame
            
        Returns:
            DataFrame with inventory information
        """
        skus = sales_df['sku_id'].unique()
        
        inventory_data = []
        for sku in skus:
            sku_sales = sales_df[sales_df['sku_id'] == sku]
            avg_daily_sales = sku_sales['sales'].mean()
            
            inventory_data.append({
                'sku_id': sku,
                'category': sku_sales['category'].iloc[0],
                'current_stock': int(np.random.uniform(avg_daily_sales * 5, avg_daily_sales * 20)),
                'reorder_point': int(avg_daily_sales * 7),
                'reorder_quantity': int(avg_daily_sales * 14),
                'lead_time_days': np.random.randint(3, 15),
                'unit_cost': sku_sales['cost'].mean(),
                'unit_price': sku_sales['price'].mean()
            })
        
        return pd.DataFrame(inventory_data)