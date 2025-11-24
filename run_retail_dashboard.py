#!/usr/bin/env python3
"""
Run retail demand forecasting dashboard.
Quick launcher with retail-specific data generation.
"""

import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retail_data_generator import RetailDataGenerator
from forecasting_agent import DemandForecastingAgent
from dashboard import ForecastingDashboard
import warnings
warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*70)
    print("🏪 RETAIL DEMAND FORECASTING SYSTEM")
    print("="*70 + "\n")
    
    # Step 1: Generate retail data
    print("📊 Step 1/4: Generating retail sales data...")
    
    # Calculate days from 2022-01-01 to today
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    days_to_generate = (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days
    
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Total days: {days_to_generate} ({days_to_generate/365:.1f} years)")
    print(f"   20 SKUs, 5 stores - this takes 3-4 minutes)\n")
    
    generator = RetailDataGenerator(seed=42)
    sales_df = generator.generate_retail_sales_data(
        start_date=start_date,
        periods=days_to_generate,  # Up to today
        n_skus=20,
        n_stores=5
    )
    
    print(f"\n✓ Generated {len(sales_df):,} sales records")
    print(f"✓ Data coverage: {sales_df['date'].min()} to {sales_df['date'].max()}")
    
    # Step 2: Generate supporting data
    print("\n📦 Step 2/4: Generating inventory and external data...")
    inventory_df = generator.generate_retail_inventory_snapshot(sales_df)
    
    # Aggregate sales by SKU and date
    sales_agg = sales_df.groupby(['date', 'sku_id', 'category']).agg({
        'units_sold': 'sum',
        'unit_price': 'mean',
        'cost': 'mean',
        'stockout': 'max'
    }).reset_index()
    sales_agg.rename(columns={'units_sold': 'sales', 'unit_price': 'price'}, inplace=True)
    
    # Create external factors
    external_df = sales_df[['date', 'is_holiday', 'promotion_active']].drop_duplicates()
    external_df = external_df.groupby('date').agg({
        'is_holiday': 'max',
        'promotion_active': 'mean'
    }).reset_index()
    
    print(f"✓ Generated inventory for {len(inventory_df)} SKU-Store combinations")
    
    # Step 3: Initialize agent
    print("\n🤖 Step 3/4: Initializing forecasting agent...")
    agent = DemandForecastingAgent()
    print("✓ Agent initialized")
    
    # Step 4: Train models
    print("\n🎓 Step 4/4: Training forecasting models...")
    skus = sales_agg['sku_id'].unique()[:5]  # Train first 5 for quick startup
    
    for i, sku in enumerate(skus, 1):
        print(f"   [{i}/{len(skus)}] Training {sku}...", end=" ")
        try:
            model, metrics = agent.train_model(sku, sales_agg, external_df)
            print(f"MAPE: {metrics['mape']:.1f}%")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✓ Model training complete")
    
    # Prepare inventory for dashboard
    inv_summary = inventory_df.groupby('sku_id').agg({
        'current_stock': 'sum',
        'reorder_point': 'mean',
        'reorder_quantity': 'mean',
        'lead_time_days': 'mean',
        'unit_cost': 'first',
        'unit_price': 'first',
        'category': 'first'
    }).reset_index()
    
    # Launch dashboard
    print("\n" + "="*70)
    print("🚀 LAUNCHING DASHBOARD")
    print("="*70)
    print("\n📈 Dashboard will open at: http://127.0.0.1:8050")
    print("\n✨ Features:")
    print("   • Real-time demand forecasting")
    print("   • Dynamic reorder recommendations")
    print("   • Historical sales analysis")
    print("   • Model performance metrics")
    print("   • Interactive visualizations")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    dashboard = ForecastingDashboard(agent, sales_agg, external_df, inv_summary)
    dashboard.run(host='127.0.0.1', port=8050, debug=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)