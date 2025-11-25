#!/usr/bin/env python3
"""
Automated forecast scheduler - runs daily forecasts automatically.
"""
import sys
import os
from datetime import datetime
sys.path.insert(0, 'src')

from automated_forecasting import AutomatedForecastScheduler
from forecasting_agent import DemandForecastingAgent
from retail_data_generator import RetailDataGenerator
import json

def load_latest_data():
    """
    Load latest sales data.
    
    In production, replace this with database query:
    sales_df = pd.read_sql('SELECT * FROM sales WHERE date >= ...', connection)
    """
    from datetime import datetime, timedelta
    
    # For now, use generator (replace with actual data loading in production)
    gen = RetailDataGenerator(seed=42)
    
    # Generate data from 2022 to today
    start_date = '2022-01-01'
    days_to_generate = (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days
    
    sales_df = gen.generate_retail_sales_data(
        start_date=start_date,
        periods=days_to_generate,
        n_skus=20,
        n_stores=5
    )
    
    # Aggregate for forecasting
    sales_agg = sales_df.groupby(['date', 'sku_id', 'category']).agg({
        'units_sold': 'sum',
        'unit_price': 'mean',
        'cost': 'mean'
    }).reset_index()
    sales_agg.rename(columns={'units_sold': 'sales', 'unit_price': 'price'}, inplace=True)
    
    # External factors
    external_df = sales_df[['date', 'is_holiday', 'promotion_active']].drop_duplicates()
    external_df = external_df.groupby('date').agg({
        'is_holiday': 'max',
        'promotion_active': 'mean'
    }).reset_index()
    
    # Inventory
    inventory_df = gen.generate_retail_inventory_snapshot(sales_df)
    inventory_df = inventory_df.groupby('sku_id').agg({
        'current_stock': 'sum',
        'reorder_point': 'mean',
        'reorder_quantity': 'mean',
        'lead_time_days': 'mean',
        'unit_cost': 'first',
        'unit_price': 'first',
        'category': 'first'
    }).reset_index()
    
    return sales_agg, external_df, inventory_df

def main():
    print("\n" + "="*70)
    print(" AUTOMATED FORECAST SCHEDULER")
    print("="*70 + "\n")
    
    # Load configuration
    config_path = 'config/notification_config.json'
    
    if not os.path.exists(config_path):
        print("  Configuration file not found. Creating default config...")
        os.makedirs('config', exist_ok=True)
        
        default_config = {
            "enabled": False,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipient_emails": ["inventory@yourcompany.com"],
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "schedule_time": "09:00"
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✓ Created default config: {config_path}")
        print("⚠️  Email notifications disabled by default")
        print("   Edit config/notification_config.json to enable\n")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f" Email Notifications: {'ENABLED ✓' if config.get('enabled') else 'DISABLED'}")
    print(f" Schedule Time: {config.get('schedule_time', '09:00')} daily")
    print()
    
    # Initialize components
    print(" Initializing forecasting agent...")
    agent = DemandForecastingAgent()
    
    print(" Initializing automated scheduler...")
    scheduler = AutomatedForecastScheduler(agent, load_latest_data, config)
    
    print("✓ Initialization complete\n")
    
    # Start scheduler
    print(" Starting automated forecast scheduler...")
    print("\nScheduler will:")
    print("  1. Run immediately (initial forecast)")
    print(f"  2. Run daily at {config.get('schedule_time', '09:00')}")
    print("  3. Save results to data/forecasts/")
    if config.get('enabled'):
        print(f"  4. Send email alerts to {len(config.get('recipient_emails', []))} recipients")
    print("\n⚠️  Press Ctrl+C to stop\n")
    
    scheduler.start_scheduler(schedule_time=config.get('schedule_time', '09:00'))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Scheduler stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)