#!/usr/bin/env python3
"""
Advanced dashboard with supplier tracking and customer segmentation.
Includes all new features integrated into the dashboard.
"""

import sys
import os
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from retail_data_generator import RetailDataGenerator
from forecasting_agent import DemandForecastingAgent
from dashboard import ForecastingDashboard
from supplier_tracking import SupplierPerformanceTracker
from customer_segmentation import CustomerSegmentationAnalyzer
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

def generate_sample_suppliers(n_suppliers=10):
    """Generate sample supplier data."""
    suppliers = []
    categories = ['Electronics', 'Apparel', 'Home & Garden', 'Groceries', 
                 'Beauty & Personal Care', 'Sports & Outdoors', 'Toys & Games', 'Books & Media']
    
    for i in range(n_suppliers):
        suppliers.append({
            'supplier_id': f'SUP_{i+1:03d}',
            'name': f'Supplier {i+1}',
            'category': np.random.choice(categories),
            'contact_email': f'supplier{i+1}@example.com'
        })
    
    return suppliers

def generate_sample_customers(sales_df, n_customers=500):
    """Generate sample customer transaction data."""
    # Assign customer IDs to sales
    sales_with_customers = sales_df.copy()
    sales_with_customers['customer_id'] = np.random.randint(1, n_customers+1, size=len(sales_df))
    sales_with_customers['customer_id'] = sales_with_customers['customer_id'].apply(lambda x: f'CUST_{x:05d}')
    
    return sales_with_customers

def main():
    print("\n" + "="*70)
    print(" ADVANCED RETAIL DEMAND FORECASTING SYSTEM")
    print("   with Supplier Tracking & Customer Segmentation")
    print("="*70 + "\n")
    
    # Step 1: Generate retail data
    print(" Step 1/6: Generating retail sales data...")
    
    start_date = '2022-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    days_to_generate = (datetime.now() - datetime.strptime(start_date, '%Y-%m-%d')).days
    
    print(f"   Date range: {start_date} to {end_date}")
    print(f"   Total days: {days_to_generate} ({days_to_generate/365:.1f} years)")
    print(f"   20 SKUs, 5 stores - this takes 3-4 minutes)\n")
    
    generator = RetailDataGenerator(seed=42)
    sales_df = generator.generate_retail_sales_data(
        start_date=start_date,
        periods=days_to_generate,
        n_skus=20,
        n_stores=5
    )
    
    print(f"\n✓ Generated {len(sales_df):,} sales records")
    
    # Step 2: Add customer data
    print("\n Step 2/6: Adding customer transaction data...")
    sales_with_customers = generate_sample_customers(sales_df, n_customers=500)
    print(f"✓ Added {sales_with_customers['customer_id'].nunique()} unique customers")
    
    # Step 3: Initialize customer segmentation
    print("\n Step 3/6: Performing customer segmentation...")
    segmentation_analyzer = CustomerSegmentationAnalyzer()
    
    # RFM Analysis
    rfm_df = segmentation_analyzer.perform_rfm_analysis(
        sales_with_customers,
        customer_id_col='customer_id',
        date_col='date',
        revenue_col='revenue'
    )
    print(f"✓ RFM analysis complete")
    print(f"   Customer segments identified:")
    for segment, count in rfm_df['segment'].value_counts().items():
        print(f"     - {segment}: {count} customers")
    
    # Save segmentation results
    rfm_df.to_csv('data/customer_segments.csv', index=False)
    print(f"✓ Saved: data/customer_segments.csv")
    
    # Step 4: Initialize supplier tracking
    print("\n Step 4/6: Initializing supplier performance tracking...")
    supplier_tracker = SupplierPerformanceTracker()
    
    # Add suppliers
    suppliers = generate_sample_suppliers(n_suppliers=10)
    for supplier in suppliers:
        supplier_tracker.add_supplier(
            supplier['supplier_id'],
            supplier['name'],
            supplier['category'],
            {'email': supplier['contact_email']}
        )
    print(f"✓ Added {len(suppliers)} suppliers to tracking system")
    
    # Generate sample purchase orders and deliveries
    print("   Generating sample supplier performance data...")
    for i in range(50):
        supplier = np.random.choice(suppliers)
        po_date = datetime.now() - timedelta(days=np.random.randint(1, 90))
        expected_delivery = po_date + timedelta(days=np.random.randint(5, 15))
        actual_delivery = expected_delivery + timedelta(days=np.random.randint(-2, 5))
        
        po_id = f'PO_{i+1:05d}'
        
        supplier_tracker.record_purchase_order(
            po_id, supplier['supplier_id'], f'SKU_{np.random.randint(1, 20):04d}',
            np.random.randint(100, 1000), np.random.uniform(10, 50),
            po_date, expected_delivery
        )
        
        supplier_tracker.record_delivery(
            po_id, actual_delivery,
            np.random.randint(95, 105) / 100 * np.random.randint(100, 1000),
            quality_rating=np.random.randint(3, 6)
        )
    
    # Generate supplier report
    supplier_performance = supplier_tracker.get_all_supplier_performance()
    supplier_performance.to_csv('data/supplier_performance.csv', index=False)
    print(f"✓ Supplier performance report saved: data/supplier_performance.csv")
    
    # Step 5: Prepare data for forecasting
    print("\n Step 5/6: Preparing data and initializing forecasting agent...")
    inventory_df = generator.generate_retail_inventory_snapshot(sales_df)
    
    sales_agg = sales_df.groupby(['date', 'sku_id', 'category']).agg({
        'units_sold': 'sum',
        'unit_price': 'mean',
        'cost': 'mean',
        'stockout': 'max'
    }).reset_index()
    sales_agg.rename(columns={'units_sold': 'sales', 'unit_price': 'price'}, inplace=True)
    
    external_df = sales_df[['date', 'is_holiday', 'promotion_active']].drop_duplicates()
    external_df = external_df.groupby('date').agg({
        'is_holiday': 'max',
        'promotion_active': 'mean'
    }).reset_index()
    
    # Prepare inventory summary for dashboard
    inv_summary = inventory_df.groupby('sku_id').agg({
        'current_stock': 'sum',
        'reorder_point': 'mean',
        'reorder_quantity': 'mean',
        'lead_time_days': 'mean',
        'unit_cost': 'first',
        'unit_price': 'first',
        'category': 'first'
    }).reset_index()
    
    agent = DemandForecastingAgent()
    print(" Agent initialized")
    
    # Step 6: Train models
    print("\n🎓 Step 6/6: Training forecasting models...")
    skus = sales_agg['sku_id'].unique()[:5]
    
    for i, sku in enumerate(skus, 1):
        print(f"   [{i}/{len(skus)}] Training {sku}...", end=" ")
        try:
            model, metrics = agent.train_model(sku, sales_agg, external_df)
            print(f"MAPE: {metrics['mape']:.1f}%")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✓ Model training complete")
    
    # Pre-calculate urgent reorders for all SKUs
    print("\n  Calculating urgent reorders for all trained SKUs...")
    print(f"   Total SKUs in inventory: {len(inv_summary)}")
    print(f"   Trained models: {len(agent.models)}")
    
    urgent_skus = []
    
    for sku in inv_summary['sku_id'].unique():
        if sku in agent.models:
            try:
                future = pd.date_range(start=datetime.now(), periods=30, freq='D')
                fcast = agent.predict_demand(sku, future, external_df)
                inv = inv_summary[inv_summary['sku_id'] == sku].iloc[0]
                
                reorder_check = agent.calculate_dynamic_reorder(
                    sku, fcast, int(inv['current_stock']), int(inv['lead_time_days'])
                )
                
                print(f"   {sku}: {reorder_check['urgency']} - {reorder_check['days_until_stockout']} days to stockout")
                
                if reorder_check['urgency'] == 'HIGH':
                    urgent_skus.append(sku)
                    print(f"       MARKED AS URGENT")
            except Exception as e:
                print(f"   {sku}: Error - {e}")
        else:
            print(f"   {sku}: Not trained yet (skipping)")
    
    print(f"\n✓ Urgent reorder calculation complete")
    print(f"✓ Found {len(urgent_skus)} SKUs with HIGH urgency: {urgent_skus}")
    
    # Save urgent list for dashboard
    urgent_reorder_count = len(urgent_skus)
    print(f"✓ Urgent count set to: {urgent_reorder_count}")
    
    # Save analytics results
    print("\n Saving analytics results...")
    print("✓ data/customer_segments.csv")
    print("✓ data/supplier_performance.csv")
    
    # Debug: Check data
    print(f"\n DEBUG: Checking data for dashboard...")
    print(f"   Customer segments: {len(rfm_df) if rfm_df is not None else 0} records")
    print(f"   Supplier performance: {len(supplier_performance) if supplier_performance is not None else 0} records")
    print(f"   Segments preview:")
    if rfm_df is not None and len(rfm_df) > 0:
        print(rfm_df['segment'].value_counts().head())
    
    # Launch dashboard
    print("\n" + "="*70)
    print(" LAUNCHING ADVANCED DASHBOARD")
    print("="*70)
    print("\n Dashboard URL: http://127.0.0.1:8050")
    print("\n Advanced Features Enabled:")
    print("   • Real-time demand forecasting")
    print("   • Dynamic reorder recommendations")
    print(f"   • Customer segmentation ({len(rfm_df) if rfm_df is not None else 0} customers)")
    print(f"   • Supplier performance tracking ({len(supplier_performance) if supplier_performance is not None else 0} suppliers)")
    print("   • Risk assessment dashboard")
    print("   • AI-powered insights")
    print("\n Analytics Available:")
    print(f"   • {sales_with_customers['customer_id'].nunique()} customers segmented")
    print(f"   • {len(suppliers)} suppliers tracked")
    print(f"   • {len(supplier_tracker.deliveries)} deliveries analyzed")
    print("\n  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Pass all data to dashboard including analytics
    print(" Passing data to dashboard...")
    print(f"   - Sales data: {len(sales_agg):,} records")
    print(f"   - Customer segments: {len(rfm_df) if rfm_df is not None else 0} customers")
    print(f"   - Supplier metrics: {len(supplier_performance) if supplier_performance is not None else 0} suppliers")
    print(f"   - Urgent reorders: {urgent_reorder_count}\n")
    
    dashboard = ForecastingDashboard(
        agent, 
        sales_agg, 
        external_df, 
        inv_summary,
        customer_segments=rfm_df,
        supplier_performance=supplier_performance
    )
    print("RFM DF CHECK:")
    print(type(rfm_df))
    print(rfm_df.head())
    print("Rows:", len(rfm_df))
    print("Columns:", rfm_df.columns.tolist())
    # Store urgent count in dashboard
    dashboard.urgent_reorder_count = urgent_reorder_count
    
    dashboard.run(host='127.0.0.1', port=8050, debug=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)