"""
Supplier performance tracking and analytics.
Monitors delivery times, quality, costs, and reliability.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SupplierPerformanceTracker:
    """Track and analyze supplier performance metrics."""
    
    def __init__(self):
        self.suppliers = {}
        self.purchase_orders = []
        self.deliveries = []
        self.quality_issues = []
        
    def add_supplier(self, supplier_id, name, category, contact_info=None):
        """
        Add a supplier to the tracking system.
        
        Args:
            supplier_id: Unique supplier identifier
            name: Supplier name
            category: Product category they supply
            contact_info: Contact details dict
        """
        self.suppliers[supplier_id] = {
            'supplier_id': supplier_id,
            'name': name,
            'category': category,
            'contact_info': contact_info or {},
            'onboarding_date': datetime.now(),
            'status': 'Active'
        }
    
    def record_purchase_order(self, po_id, supplier_id, sku_id, quantity, 
                             unit_cost, order_date, expected_delivery_date):
        """Record a new purchase order."""
        self.purchase_orders.append({
            'po_id': po_id,
            'supplier_id': supplier_id,
            'sku_id': sku_id,
            'quantity': quantity,
            'unit_cost': unit_cost,
            'total_cost': quantity * unit_cost,
            'order_date': order_date,
            'expected_delivery_date': expected_delivery_date,
            'actual_delivery_date': None,
            'status': 'Pending'
        })
    
    def record_delivery(self, po_id, actual_delivery_date, quantity_received, 
                       quality_rating=5, notes=None):
        """Record delivery receipt."""
        # Find and update PO
        for po in self.purchase_orders:
            if po['po_id'] == po_id:
                po['actual_delivery_date'] = actual_delivery_date
                po['status'] = 'Delivered'
                
                # Calculate lead time accuracy
                expected = pd.to_datetime(po['expected_delivery_date'])
                actual = pd.to_datetime(actual_delivery_date)
                days_late = (actual - expected).days
                
                self.deliveries.append({
                    'po_id': po_id,
                    'supplier_id': po['supplier_id'],
                    'sku_id': po['sku_id'],
                    'delivery_date': actual_delivery_date,
                    'quantity_ordered': po['quantity'],
                    'quantity_received': quantity_received,
                    'fill_rate': (quantity_received / po['quantity']) * 100,
                    'days_late': days_late,
                    'on_time': days_late <= 0,
                    'quality_rating': quality_rating,
                    'notes': notes
                })
                
                # Record quality issues if any
                if quality_rating < 4:
                    self.quality_issues.append({
                        'po_id': po_id,
                        'supplier_id': po['supplier_id'],
                        'date': actual_delivery_date,
                        'quality_rating': quality_rating,
                        'notes': notes
                    })
                
                break
    
    def calculate_supplier_metrics(self, supplier_id, days_lookback=90):
        """
        Calculate comprehensive supplier performance metrics.
        
        Args:
            supplier_id: Supplier to analyze
            days_lookback: Number of days to look back
            
        Returns:
            Dict with performance metrics
        """
        # Get deliveries for this supplier
        deliveries_df = pd.DataFrame(self.deliveries)
        
        if len(deliveries_df) == 0:
            return self._empty_metrics()
        
        supplier_deliveries = deliveries_df[
            (deliveries_df['supplier_id'] == supplier_id) &
            (pd.to_datetime(deliveries_df['delivery_date']) >= 
             datetime.now() - timedelta(days=days_lookback))
        ]
        
        if len(supplier_deliveries) == 0:
            return self._empty_metrics()
        
        # Calculate metrics
        metrics = {
            'supplier_id': supplier_id,
            'supplier_name': self.suppliers[supplier_id]['name'],
            'total_orders': len(supplier_deliveries),
            'on_time_delivery_rate': (supplier_deliveries['on_time'].sum() / len(supplier_deliveries)) * 100,
            'avg_days_late': supplier_deliveries['days_late'].mean(),
            'avg_fill_rate': supplier_deliveries['fill_rate'].mean(),
            'avg_quality_rating': supplier_deliveries['quality_rating'].mean(),
            'total_quantity_ordered': supplier_deliveries['quantity_ordered'].sum(),
            'total_quantity_received': supplier_deliveries['quantity_received'].sum(),
            'quality_issues_count': len([q for q in self.quality_issues 
                                        if q['supplier_id'] == supplier_id]),
            'performance_score': self._calculate_performance_score(supplier_deliveries),
            'reliability_tier': None
        }
        
        # Assign reliability tier
        if metrics['performance_score'] >= 90:
            metrics['reliability_tier'] = 'A - Excellent'
        elif metrics['performance_score'] >= 75:
            metrics['reliability_tier'] = 'B - Good'
        elif metrics['performance_score'] >= 60:
            metrics['reliability_tier'] = 'C - Acceptable'
        else:
            metrics['reliability_tier'] = 'D - Needs Improvement'
        
        return metrics
    
    def _calculate_performance_score(self, deliveries):
        """Calculate overall supplier performance score (0-100)."""
        if len(deliveries) == 0:
            return 0
        
        # Weighted scoring
        on_time_score = (deliveries['on_time'].sum() / len(deliveries)) * 40  # 40% weight
        fill_rate_score = deliveries['fill_rate'].mean() * 0.30  # 30% weight
        quality_score = (deliveries['quality_rating'].mean() / 5) * 30  # 30% weight
        
        total_score = on_time_score + fill_rate_score + quality_score
        
        return round(total_score, 1)
    
    def _empty_metrics(self):
        """Return empty metrics structure."""
        return {
            'supplier_id': None,
            'total_orders': 0,
            'on_time_delivery_rate': 0,
            'avg_days_late': 0,
            'avg_fill_rate': 0,
            'avg_quality_rating': 0,
            'performance_score': 0,
            'reliability_tier': 'No Data'
        }
    
    def get_all_supplier_performance(self, days_lookback=90):
        """Get performance metrics for all suppliers."""
        metrics_list = []
        
        for supplier_id in self.suppliers.keys():
            metrics = self.calculate_supplier_metrics(supplier_id, days_lookback)
            metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list).sort_values('performance_score', ascending=False)
    
    def generate_supplier_report(self, supplier_id=None, output_path='data/supplier_report.csv'):
        """Generate comprehensive supplier performance report."""
        if supplier_id:
            # Single supplier report
            metrics = self.calculate_supplier_metrics(supplier_id)
            report_df = pd.DataFrame([metrics])
        else:
            # All suppliers report
            report_df = self.get_all_supplier_performance()
        
        # Save report
        report_df.to_csv(output_path, index=False)
        print(f"✓ Supplier report saved: {output_path}")
        
        return report_df
    
    def identify_at_risk_suppliers(self, min_performance_score=60):
        """Identify suppliers that need attention."""
        all_metrics = self.get_all_supplier_performance()
        
        at_risk = all_metrics[all_metrics['performance_score'] < min_performance_score]
        
        return at_risk
    
    def recommend_supplier_actions(self):
        """Generate action recommendations for supplier management."""
        all_metrics = self.get_all_supplier_performance()
        
        recommendations = []
        
        for _, supplier in all_metrics.iterrows():
            actions = []
            
            if supplier['on_time_delivery_rate'] < 80:
                actions.append(f"⚠️ On-time delivery below 80% - Schedule performance review")
            
            if supplier['avg_fill_rate'] < 95:
                actions.append(f"⚠️ Fill rate below 95% - Discuss capacity issues")
            
            if supplier['avg_quality_rating'] < 4:
                actions.append(f"⚠️ Quality concerns - Implement improvement plan")
            
            if supplier['performance_score'] < 60:
                actions.append(f" Overall performance poor - Consider alternative suppliers")
            
            if supplier['performance_score'] >= 90:
                actions.append(f" Excellent performance - Consider expanding relationship")
            
            if actions:
                recommendations.append({
                    'supplier_id': supplier['supplier_id'],
                    'supplier_name': supplier['supplier_name'],
                    'performance_score': supplier['performance_score'],
                    'actions': actions
                })
        
        return recommendations