"""
Generate realistic retail sales data for demand forecasting demonstration.
Includes realistic retail patterns: promotions, holidays, store locations, etc.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RetailDataGenerator:
    """Generate realistic retail sales data with industry-specific patterns."""
    
    # Retail product categories with typical characteristics
    RETAIL_CATEGORIES = {
        'Electronics': {
            'base_price_range': (99.99, 999.99),
            'base_demand': (30, 80),
            'seasonality_strength': 0.4,
            'weekend_boost': 1.3,
            'holiday_boost': 2.5,
            'promo_effectiveness': 1.8
        },
        'Apparel': {
            'base_price_range': (19.99, 149.99),
            'base_demand': (50, 150),
            'seasonality_strength': 0.6,
            'weekend_boost': 1.5,
            'holiday_boost': 2.0,
            'promo_effectiveness': 2.2
        },
        'Home & Garden': {
            'base_price_range': (14.99, 199.99),
            'base_demand': (25, 90),
            'seasonality_strength': 0.5,
            'weekend_boost': 1.4,
            'holiday_boost': 1.8,
            'promo_effectiveness': 1.6
        },
        'Groceries': {
            'base_price_range': (2.99, 29.99),
            'base_demand': (100, 300),
            'seasonality_strength': 0.2,
            'weekend_boost': 1.1,
            'holiday_boost': 1.5,
            'promo_effectiveness': 2.5
        },
        'Beauty & Personal Care': {
            'base_price_range': (9.99, 79.99),
            'base_demand': (40, 120),
            'seasonality_strength': 0.3,
            'weekend_boost': 1.2,
            'holiday_boost': 1.7,
            'promo_effectiveness': 2.0
        },
        'Sports & Outdoors': {
            'base_price_range': (24.99, 299.99),
            'base_demand': (20, 70),
            'seasonality_strength': 0.7,
            'weekend_boost': 1.6,
            'holiday_boost': 1.9,
            'promo_effectiveness': 1.7
        },
        'Toys & Games': {
            'base_price_range': (9.99, 99.99),
            'base_demand': (35, 100),
            'seasonality_strength': 0.8,
            'weekend_boost': 1.4,
            'holiday_boost': 3.5,
            'promo_effectiveness': 1.9
        },
        'Books & Media': {
            'base_price_range': (12.99, 49.99),
            'base_demand': (30, 85),
            'seasonality_strength': 0.3,
            'weekend_boost': 1.2,
            'holiday_boost': 1.6,
            'promo_effectiveness': 1.5
        }
    }
    
    # U.S. Retail Holidays and Special Shopping Days
    RETAIL_HOLIDAYS = {
        'Black Friday': {'month': 11, 'week': 4, 'day': 5, 'boost': 4.0},
        'Cyber Monday': {'month': 11, 'week': 4, 'day': 1, 'boost': 3.5},
        'Christmas': {'month': 12, 'day_range': (20, 25), 'boost': 3.0},
        'Thanksgiving': {'month': 11, 'week': 4, 'boost': 2.5},
        'Memorial Day': {'month': 5, 'week': -1, 'boost': 2.0},
        'Labor Day': {'month': 9, 'week': 1, 'boost': 2.0},
        'Independence Day': {'month': 7, 'day': 4, 'boost': 2.2},
        'Back to School': {'month': 8, 'day_range': (15, 31), 'boost': 2.5},
        'Valentine\'s Day': {'month': 2, 'day': 14, 'boost': 1.8},
        'Mother\'s Day': {'month': 5, 'week': 2, 'day': 0, 'boost': 2.0},
        'Father\'s Day': {'month': 6, 'week': 3, 'day': 0, 'boost': 1.7},
        'Easter': {'month': 4, 'week': 2, 'boost': 1.6},
        'Prime Day': {'month': 7, 'day_range': (10, 15), 'boost': 3.0}
    }
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
    def generate_retail_sales_data(self, start_date='2022-01-01', periods=730, 
                                   n_skus=20, n_stores=5):
        """
        Generate realistic retail sales data.
        
        Args:
            start_date: Starting date for the time series
            periods: Number of days to generate
            n_skus: Number of SKUs to generate
            n_stores: Number of store locations
            
        Returns:
            DataFrame with retail sales data
        """
        date_range = pd.date_range(start=start_date, periods=periods, freq='D')
        
        # Generate store information
        stores = self._generate_stores(n_stores)
        
        # Generate product catalog
        products = self._generate_product_catalog(n_skus)
        
        data = []
        
        print(f"Generating {periods} days of sales data for {n_skus} SKUs across {n_stores} stores...")
        
        for store in stores:
            for product in products:
                # Get category characteristics
                category_info = self.RETAIL_CATEGORIES[product['category']]
                
                # Base demand varies by store
                base_demand = np.random.uniform(
                    category_info['base_demand'][0] * store['size_multiplier'],
                    category_info['base_demand'][1] * store['size_multiplier']
                )
                
                for date in date_range:
                    # Core demand components
                    trend = self._calculate_trend(date, date_range, base_demand)
                    seasonal = self._calculate_seasonal_effect(date, category_info)
                    weekly = self._calculate_weekly_pattern(date, category_info)
                    holiday = self._calculate_holiday_effect(date, category_info, product['category'])
                    
                    # Promotion effects
                    is_promo = self._is_promotion_day(date, product['sku_id'])
                    promo_effect = category_info['promo_effectiveness'] if is_promo else 1.0
                    
                    # Weather impact (simplified)
                    weather_effect = self._calculate_weather_impact(date, product['category'])
                    
                    # Competitor activity (random)
                    competitor_effect = np.random.uniform(0.9, 1.1)
                    
                    # Stock availability (occasional stockouts)
                    stockout = np.random.random() < 0.015  # 1.5% stockout rate
                    
                    # Random noise
                    noise = np.random.normal(0, base_demand * 0.15)
                    
                    # Calculate final demand
                    demand = (trend + seasonal + weekly + holiday) * promo_effect * \
                             weather_effect * competitor_effect + noise
                    
                    # Apply constraints
                    demand = max(0, demand)
                    units_sold = 0 if stockout else int(demand)
                    
                    # Calculate revenue
                    price = product['base_price'] * (0.7 if is_promo else 1.0)  # 30% discount on promo
                    revenue = units_sold * price
                    
                    data.append({
                        'date': date,
                        'store_id': store['store_id'],
                        'store_name': store['name'],
                        'store_location': store['location'],
                        'store_type': store['type'],
                        'sku_id': product['sku_id'],
                        'product_name': product['name'],
                        'category': product['category'],
                        'brand': product['brand'],
                        'units_sold': units_sold,
                        'unit_price': round(price, 2),
                        'revenue': round(revenue, 2),
                        'cost': product['cost'],
                        'margin': round(revenue - (units_sold * product['cost']), 2),
                        'promotion_active': is_promo,
                        'stockout': stockout,
                        'day_of_week': date.strftime('%A'),
                        'is_weekend': date.dayofweek >= 5,
                        'is_holiday': self._is_holiday(date)
                    })
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df):,} sales records")
        return df
    
    def _generate_stores(self, n_stores):
        """Generate store information."""
        store_types = ['Urban Flagship', 'Suburban', 'Mall', 'Outlet', 'Express']
        locations = ['New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 
                    'Phoenix, AZ', 'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA',
                    'Dallas, TX', 'Austin, TX']
        
        stores = []
        for i in range(n_stores):
            store_type = np.random.choice(store_types)
            size_mult = {
                'Urban Flagship': 1.5,
                'Suburban': 1.0,
                'Mall': 1.2,
                'Outlet': 0.8,
                'Express': 0.6
            }[store_type]
            
            stores.append({
                'store_id': f'STORE_{i+1:03d}',
                'name': f'Store {i+1}',
                'location': np.random.choice(locations),
                'type': store_type,
                'size_multiplier': size_mult
            })
        
        return stores
    
    def _generate_product_catalog(self, n_skus):
        """Generate product catalog with realistic retail items."""
        products = []
        brands = {
            'Electronics': ['TechPro', 'SmartChoice', 'DigitalEdge', 'VisionTech'],
            'Apparel': ['StyleCo', 'UrbanWear', 'ClassicFit', 'TrendLine'],
            'Home & Garden': ['HomeEssentials', 'GardenPro', 'ComfortLiving'],
            'Groceries': ['FreshChoice', 'NatureValue', 'DailyEats'],
            'Beauty & Personal Care': ['GlowUp', 'PureBeauty', 'SkinFirst'],
            'Sports & Outdoors': ['ActiveLife', 'TrailBlazer', 'FitPro'],
            'Toys & Games': ['PlayTime', 'FunZone', 'KidJoy'],
            'Books & Media': ['ReadMore', 'MediaHub', 'BookWise']
        }
        
        product_names = {
            'Electronics': ['Wireless Headphones', 'Smart Watch', 'Tablet', 'Bluetooth Speaker', 
                           'Phone Charger', 'USB Cable', 'Power Bank', 'Webcam'],
            'Apparel': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Sneakers', 'Sweater', 
                       'Shorts', 'Hoodie'],
            'Home & Garden': ['Throw Pillow', 'Bedding Set', 'Kitchen Utensil Set', 
                             'Garden Tools', 'Storage Bin', 'Wall Art'],
            'Groceries': ['Organic Pasta', 'Snack Mix', 'Coffee Beans', 'Cereal', 
                         'Olive Oil', 'Trail Mix', 'Energy Bars'],
            'Beauty & Personal Care': ['Face Cream', 'Shampoo', 'Body Lotion', 
                                       'Makeup Kit', 'Hair Dryer', 'Perfume'],
            'Sports & Outdoors': ['Yoga Mat', 'Water Bottle', 'Resistance Bands', 
                                 'Running Shoes', 'Backpack', 'Camping Gear'],
            'Toys & Games': ['Board Game', 'Action Figure', 'Puzzle', 'Building Blocks', 
                            'Doll', 'Remote Control Car'],
            'Books & Media': ['Bestseller Novel', 'Cookbook', 'Magazine', 'DVD', 
                             'Audiobook', 'Educational Book']
        }
        
        categories = list(self.RETAIL_CATEGORIES.keys())
        
        for i in range(n_skus):
            category = np.random.choice(categories)
            cat_info = self.RETAIL_CATEGORIES[category]
            
            base_price = np.random.uniform(cat_info['base_price_range'][0], 
                                          cat_info['base_price_range'][1])
            cost = base_price * np.random.uniform(0.4, 0.7)  # 30-60% margin
            
            products.append({
                'sku_id': f'SKU_{i+1:04d}',
                'name': f"{np.random.choice(product_names[category])} {i+1}",
                'category': category,
                'brand': np.random.choice(brands[category]),
                'base_price': round(base_price, 2),
                'cost': round(cost, 2)
            })
        
        return products
    
    def _calculate_trend(self, date, date_range, base_demand):
        """Calculate long-term trend component."""
        days_elapsed = (date - date_range[0]).days
        # Slight upward trend (0.05% per day)
        trend_factor = 1 + (0.0005 * days_elapsed)
        return base_demand * trend_factor
    
    def _calculate_seasonal_effect(self, date, category_info):
        """Calculate seasonal pattern based on time of year."""
        # Strong seasonality for certain categories
        day_of_year = date.timetuple().tm_yday
        seasonal_component = category_info['seasonality_strength'] * 50 * \
                           np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/2)
        return seasonal_component
    
    def _calculate_weekly_pattern(self, date, category_info):
        """Calculate day-of-week effects."""
        if date.dayofweek >= 5:  # Weekend (Friday=5, Saturday=6, Sunday=0)
            return 30 * (category_info['weekend_boost'] - 1)
        return 0
    
    def _calculate_holiday_effect(self, date, category_info, category):
        """Calculate effect of retail holidays."""
        holiday_boost = 0
        
        for holiday, info in self.RETAIL_HOLIDAYS.items():
            if self._is_near_holiday(date, info):
                # Some categories benefit more from certain holidays
                category_multiplier = 1.0
                if 'Christmas' in holiday and category in ['Toys & Games', 'Electronics']:
                    category_multiplier = 1.5
                elif 'Back to School' in holiday and category in ['Apparel', 'Electronics']:
                    category_multiplier = 1.3
                elif 'Valentine' in holiday and category in ['Beauty & Personal Care', 'Apparel']:
                    category_multiplier = 1.2
                
                holiday_boost += info['boost'] * 20 * category_multiplier
        
        return holiday_boost
    
    def _is_near_holiday(self, date, holiday_info):
        """Check if date is near a holiday (within 3 days)."""
        if 'day_range' in holiday_info:
            return (date.month == holiday_info['month'] and 
                   holiday_info['day_range'][0] <= date.day <= holiday_info['day_range'][1])
        elif 'day' in holiday_info:
            return (date.month == holiday_info['month'] and 
                   abs(date.day - holiday_info['day']) <= 3)
        return False
    
    def _is_holiday(self, date):
        """Check if date is a major retail holiday."""
        for holiday, info in self.RETAIL_HOLIDAYS.items():
            if self._is_near_holiday(date, info):
                return True
        return False
    
    def _is_promotion_day(self, date, sku_id):
        """Determine if product is on promotion (20% of days)."""
        # Use SKU and date for consistent but pseudo-random promotions
        hash_val = hash(f"{sku_id}_{date.strftime('%Y%m%d')}") % 100
        return hash_val < 20  # 20% promotion rate
    
    def _calculate_weather_impact(self, date, category):
        """Calculate weather impact on demand."""
        # Temperature effect (simplified sine wave)
        day_of_year = date.timetuple().tm_yday
        temp = 60 + 30 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        weather_multiplier = 1.0
        
        # Seasonal products
        if category == 'Sports & Outdoors':
            weather_multiplier = 1.0 + (temp - 60) / 100  # Better in warm weather
        elif category == 'Home & Garden':
            weather_multiplier = 1.0 + (temp - 50) / 80
        elif category == 'Apparel':
            # Boost in spring and fall (clothing transitions)
            if date.month in [3, 4, 9, 10]:
                weather_multiplier = 1.15
        
        return max(0.8, min(1.3, weather_multiplier))
    
    def generate_retail_inventory_snapshot(self, sales_df):
        """Generate current inventory status for retail operations."""
        
        # Group by store and SKU
        inventory_data = []
        
        for (store_id, sku_id), group in sales_df.groupby(['store_id', 'sku_id']):
            avg_daily_sales = group['units_sold'].mean()
            
            # Calculate stock levels
            weeks_of_stock = np.random.uniform(3, 8)
            current_stock = int(avg_daily_sales * 7 * weeks_of_stock)
            
            # Reorder points based on lead time and variability
            lead_time_days = np.random.randint(5, 14)
            safety_factor = 1.5
            reorder_point = int(avg_daily_sales * lead_time_days * safety_factor)
            
            inventory_data.append({
                'store_id': store_id,
                'store_name': group['store_name'].iloc[0],
                'sku_id': sku_id,
                'product_name': group['product_name'].iloc[0],
                'category': group['category'].iloc[0],
                'current_stock': current_stock,
                'reorder_point': reorder_point,
                'reorder_quantity': int(avg_daily_sales * 14),  # 2 weeks supply
                'lead_time_days': lead_time_days,
                'unit_cost': group['cost'].iloc[0],
                'unit_price': group['unit_price'].mean(),
                'avg_daily_sales': round(avg_daily_sales, 2),
                'stock_days': round(current_stock / avg_daily_sales if avg_daily_sales > 0 else 999, 1),
                'stock_status': self._get_stock_status(current_stock, reorder_point)
            })
        
        return pd.DataFrame(inventory_data)
    
    def _get_stock_status(self, current_stock, reorder_point):
        """Determine stock status."""
        if current_stock <= reorder_point * 0.5:
            return 'Critical'
        elif current_stock <= reorder_point:
            return 'Low'
        elif current_stock <= reorder_point * 2:
            return 'Normal'
        else:
            return 'Overstocked'
    
    def generate_customer_data(self, sales_df, n_customers=500):
        """Generate synthetic customer transaction data."""
        customers = []
        
        for i in range(n_customers):
            customers.append({
                'customer_id': f'CUST_{i+1:05d}',
                'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], 
                                                p=[0.5, 0.3, 0.15, 0.05]),
                'join_date': sales_df['date'].min() + timedelta(days=np.random.randint(0, 365)),
                'email_subscribed': np.random.choice([True, False], p=[0.6, 0.4]),
                'preferred_channel': np.random.choice(['In-Store', 'Online', 'Mobile App'], 
                                                     p=[0.5, 0.3, 0.2])
            })
        
        return pd.DataFrame(customers)
    
    def generate_summary_statistics(self, sales_df):
        """Generate summary statistics for retail data."""
        summary = {
            'total_revenue': sales_df['revenue'].sum(),
            'total_units_sold': sales_df['units_sold'].sum(),
            'avg_transaction_value': sales_df['revenue'].mean(),
            'total_stores': sales_df['store_id'].nunique(),
            'total_skus': sales_df['sku_id'].nunique(),
            'date_range': f"{sales_df['date'].min()} to {sales_df['date'].max()}",
            'days_of_data': sales_df['date'].nunique(),
            'avg_daily_revenue': sales_df.groupby('date')['revenue'].sum().mean(),
            'stockout_rate': (sales_df['stockout'].sum() / len(sales_df)) * 100,
            'promotion_rate': (sales_df['promotion_active'].sum() / len(sales_df)) * 100,
            'categories': sales_df['category'].unique().tolist(),
            'top_category_by_revenue': sales_df.groupby('category')['revenue'].sum().idxmax(),
            'top_category_by_units': sales_df.groupby('category')['units_sold'].sum().idxmax()
        }
        
        return summary