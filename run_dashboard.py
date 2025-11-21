#!/usr/bin/env python3
"""
Run demand forecasting dashboard.
Quick launcher script for the original (non-retail) data generator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import SalesDataGenerator
from forecasting_agent import DemandForecastingAgent
from dashboard import ForecastingDashboard
import warnings
warnings.filterwarnings('ignore')

def main():
    print("\n" + "="*70)
    print(" AI DEMAND FORECASTING & DYNAMIC REPLENISHMENT AGENT")
    print("="*70 + "\n")
    
    # Step 1: Generate data
    print(" Step 1/4: Generating sales data...")
    print("   (2 years, 10 SKUs - this takes ~1 minute)\n")
    
    generator = SalesDataGenerator(seed=42)
    sales_df = generator.generate_sales_data(
        start_date='2022-01-01',
        periods=730,  # 2 years
        n_skus=10
    )
    
    print(f"\n✓ Generated {len(sales_df):,} sales records")
    
    # Step 2: Generate supporting data
    print("\n Step 2/4: Generating external factors and inventory...")
    external_df = generator.generate_external_factors(sales_df)
    inventory_df = generator.generate_inventory_data(sales_df)
    
    print(f" Generated external factors for {len(external_df)} days")
    print(f" Generated inventory for {len(inventory_df)} SKUs")
    
    # Step 3: Initialize agent
    print("\n Step 3/4: Initializing forecasting agent...")
    agent = DemandForecastingAgent()
    print(" Agent initialized")
    
    # Step 4: Train models
    print("\n Step 4/4: Training forecasting models...")
    skus = sales_df['sku_id'].unique()[:5]  # Train first 5 for quick startup
    
    for i, sku in enumerate(skus, 1):
        print(f"   [{i}/{len(skus)}] Training {sku}...", end=" ")
        try:
            model, metrics = agent.train_model(sku, sales_df, external_df)
            print(f"MAPE: {metrics['mape']:.1f}%")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✓ Model training complete")
    
    # Launch dashboard
    print("\n" + "="*70)
    print(" LAUNCHING DASHBOARD")
    print("="*70)
    print("\n Dashboard will open at: http://127.0.0.1:8050")
    print("\n Features:")
    print("   • Real-time demand forecasting")
    print("   • Dynamic reorder recommendations")
    print("   • Historical sales analysis")
    print("   • Model performance metrics")
    print("   • Interactive visualizations")
    print("\n  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    dashboard = ForecastingDashboard(agent, sales_df, external_df, inventory_df)
    dashboard.run(host='127.0.0.1', port=8050, debug=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Dashboard stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)