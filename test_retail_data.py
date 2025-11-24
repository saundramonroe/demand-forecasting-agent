#!/usr/bin/env python3
"""
Test script for retail data generation.
Place this file in the ROOT directory of your project.
"""

import sys
import os
from datetime import datetime, timedelta

# Ensure src directory is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retail_data_generator import RetailDataGenerator
import pandas as pd
import numpy as np

def main():
    print("\n" + "="*70)
    print("🧪 RETAIL DATA GENERATOR TEST")
    print("="*70 + "\n")
    
    # Initialize generator
    print("Step 1/4: Initializing generator...")
    generator = RetailDataGenerator(seed=42)
    print("✓ Generator initialized\n")
    
    # Generate sales data
    print("Step 2/4: Generating retail sales data...")
    
    # Calculate realistic date range (last 1 year to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    days_to_generate = 365
    
    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"   (1 year, 10 SKUs, 3 stores - this takes ~30 seconds)\n")
    
    sales_df = generator.generate_retail_sales_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        periods=days_to_generate,
        n_skus=10,    # 10 products
        n_stores=3    # 3 stores
    )
    
    print(f"\n✓ Generated {len(sales_df):,} sales records\n")
    
    # Generate inventory
    print("Step 3/4: Generating inventory snapshot...")
    inventory_df = generator.generate_retail_inventory_snapshot(sales_df)
    print(f"✓ Generated inventory for {len(inventory_df)} SKU-Store combinations\n")
    
    # Generate summary
    print("Step 4/4: Calculating summary statistics...")
    summary = generator.generate_summary_statistics(sales_df)
    
    # Display results
    print("\n" + "="*70)
    print("📊 RETAIL SALES SUMMARY")
    print("="*70 + "\n")
    
    print("💰 REVENUE METRICS:")
    print(f"   Total Revenue: ${summary['total_revenue']:,.2f}")
    print(f"   Avg Daily Revenue: ${summary['avg_daily_revenue']:,.2f}")
    print(f"   Avg Transaction Value: ${summary['avg_transaction_value']:.2f}")
    
    print("\n📦 OPERATIONS:")
    print(f"   Total Units Sold: {summary['total_units_sold']:,}")
    print(f"   Stockout Rate: {summary['stockout_rate']:.2f}%")
    print(f"   Promotion Rate: {summary['promotion_rate']:.2f}%")
    
    print("\n🏪 COVERAGE:")
    print(f"   Total Stores: {summary['total_stores']}")
    print(f"   Total SKUs: {summary['total_skus']}")
    print(f"   Categories: {len(summary['categories'])}")
    print(f"   Date Range: {summary['date_range']}")
    
    print("\n🏆 TOP PERFORMERS:")
    print(f"   Best Category (Revenue): {summary['top_category_by_revenue']}")
    print(f"   Best Category (Units): {summary['top_category_by_units']}")
    
    print("\n" + "="*70 + "\n")
    
    # Display sample data
    print("📋 SAMPLE SALES RECORDS (First 5):\n")
    print(sales_df.head().to_string())
    
    print("\n\n📦 SAMPLE INVENTORY RECORDS (First 5):\n")
    print(inventory_df.head().to_string())
    
    # Category breakdown
    print("\n\n📊 REVENUE BY CATEGORY:\n")
    category_revenue = sales_df.groupby('category')['revenue'].sum().sort_values(ascending=False)
    for category, revenue in category_revenue.items():
        print(f"   {category:<25} ${revenue:>15,.2f}")
    
    # Store breakdown
    print("\n\n🏪 REVENUE BY STORE:\n")
    store_revenue = sales_df.groupby('store_name')['revenue'].sum().sort_values(ascending=False)
    for store, revenue in store_revenue.items():
        print(f"   {store:<25} ${revenue:>15,.2f}")
    
    # Save data
    print("\n" + "="*70)
    print("💾 SAVING DATA FILES")
    print("="*70 + "\n")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save files
    sales_df.to_csv('data/test_retail_sales.csv', index=False)
    print("✓ Saved: data/test_retail_sales.csv")
    
    inventory_df.to_csv('data/test_retail_inventory.csv', index=False)
    print("✓ Saved: data/test_retail_inventory.csv")
    
    # Save summary as JSON
    import json
    
    # Convert non-serializable types properly
    summary_clean = {}
    for k, v in summary.items():
        if isinstance(v, (list, pd.Timestamp)):
            summary_clean[k] = str(v)
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            summary_clean[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            summary_clean[k] = float(v)
        else:
            summary_clean[k] = v
    
    with open('data/test_summary.json', 'w') as f:
        json.dump(summary_clean, f, indent=2)
    print("✓ Saved: data/test_summary.json")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE - All systems working!")
    print("="*70 + "\n")
    
    print("🎯 NEXT STEPS:")
    print("   1. Check the generated CSV files in data/ directory")
    print("   2. Run the full notebook: jupyter notebook notebooks/retail_forecasting_demo.ipynb")
    print("   3. Launch the dashboard: python run_retail_dashboard.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   1. Make sure you're in the project root directory")
        print("   2. Activate conda environment: conda activate demand-forecast")
        print("   3. Check that src/retail_data_generator.py exists")
        import traceback
        traceback.print_exc()
        sys.exit(1)