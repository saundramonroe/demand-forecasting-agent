"""
Automated daily forecasting scheduler with email notifications.
Runs forecasts automatically and sends alerts for urgent reorders.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import json

class AutomatedForecastScheduler:
    """Automated daily forecast generation and alert system."""
    
    def __init__(self, agent, data_loader, notification_config=None):
        """
        Initialize automated scheduler.
        
        Args:
            agent: DemandForecastingAgent instance
            data_loader: Function to load latest sales data
            notification_config: Email configuration dict
        """
        self.agent = agent
        self.data_loader = data_loader
        self.notification_config = notification_config or {}
        self.forecast_history = []
        
    def run_daily_forecast(self):
        """Execute daily forecast for all SKUs."""
        print(f"\n{'='*70}")
        print(f" AUTOMATED DAILY FORECAST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        try:
            # Load latest data
            print(" Loading latest sales data...")
            sales_df, external_df, inventory_df = self.data_loader()
            print(f"✓ Loaded {len(sales_df):,} sales records")
            
            # Generate forecasts for all SKUs
            all_forecasts = []
            urgent_reorders = []
            
            skus = sales_df['sku_id'].unique()
            print(f"\n Generating forecasts for {len(skus)} SKUs...")
            
            for i, sku in enumerate(skus, 1):
                try:
                    # Train/update model
                    if sku not in self.agent.models:
                        self.agent.train_model(sku, sales_df, external_df)
                    else:
                        # Retrain with latest data
                        self.agent.adapt_and_retrain(sku, sales_df, external_df)
                    
                    # Generate 30-day forecast
                    future_dates = pd.date_range(
                        start=datetime.now(),
                        periods=30,
                        freq='D'
                    )
                    forecast_df = self.agent.predict_demand(sku, future_dates, external_df)
                    
                    # Get inventory info
                    inv_info = inventory_df[inventory_df['sku_id'] == sku]
                    if len(inv_info) > 0:
                        inv_info = inv_info.iloc[0]
                        
                        # Calculate reorder recommendation
                        reorder_info = self.agent.calculate_dynamic_reorder(
                            sku,
                            forecast_df,
                            int(inv_info['current_stock']),
                            int(inv_info['lead_time_days'])
                        )
                        
                        # Store forecast
                        forecast_record = {
                            'timestamp': datetime.now(),
                            'sku_id': sku,
                            'category': inv_info['category'],
                            'forecast_avg': forecast_df['predicted_demand'].mean(),
                            'forecast_total': forecast_df['predicted_demand'].sum(),
                            'current_stock': reorder_info['current_stock'],
                            'reorder_needed': reorder_info['needs_reorder'],
                            'urgency': reorder_info['urgency'],
                            'days_until_stockout': reorder_info['days_until_stockout'],
                            'recommended_quantity': reorder_info['reorder_quantity'],
                            'model_accuracy': self.agent.forecast_accuracy[sku]['test_score']
                        }
                        
                        all_forecasts.append(forecast_record)
                        
                        # Track urgent reorders
                        if reorder_info['urgency'] == 'HIGH':
                            urgent_reorders.append(forecast_record)
                        
                        print(f"  [{i}/{len(skus)}] {sku}: ✓ Forecast complete | Urgency: {reorder_info['urgency']}")
                    
                except Exception as e:
                    print(f"  [{i}/{len(skus)}] {sku}: ✗ Error - {e}")
            
            # Save forecast results
            self._save_forecast_results(all_forecasts)
            
            # Send notifications if needed
            if urgent_reorders:
                self._send_urgent_alerts(urgent_reorders)
            
            # Log to history
            self.forecast_history.append({
                'timestamp': datetime.now(),
                'total_skus': len(skus),
                'forecasts_generated': len(all_forecasts),
                'urgent_reorders': len(urgent_reorders)
            })
            
            print(f"\n{'='*70}")
            print(f" DAILY FORECAST COMPLETE")
            print(f"{'='*70}")
            print(f"   Total SKUs: {len(skus)}")
            print(f"   Successful Forecasts: {len(all_forecasts)}")
            print(f"   Urgent Reorders: {len(urgent_reorders)}")
            print(f"   Next run: Tomorrow at {self.notification_config.get('schedule_time', '09:00')}")
            print(f"{'='*70}\n")
            
            return all_forecasts
            
        except Exception as e:
            print(f" Error in daily forecast: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _save_forecast_results(self, forecasts):
        """Save forecast results to files."""
        os.makedirs('data/forecasts', exist_ok=True)
        
        # Save as CSV
        forecast_df = pd.DataFrame(forecasts)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'data/forecasts/daily_forecast_{timestamp}.csv'
        forecast_df.to_csv(filepath, index=False)
        print(f"\n✓ Saved forecast results: {filepath}")
        
        # Also save latest
        forecast_df.to_csv('data/forecasts/latest_forecast.csv', index=False)
        
        return filepath
    
    def _send_urgent_alerts(self, urgent_reorders):
        """Send email alerts for urgent reorders."""
        if not self.notification_config.get('enabled', False):
            print("\n  Email notifications disabled. Enable in config to send alerts.")
            return
        
        try:
            # Prepare email
            sender = self.notification_config['sender_email']
            recipients = self.notification_config['recipient_emails']
            password = self.notification_config['sender_password']
            
            subject = f" URGENT: {len(urgent_reorders)} SKUs Require Immediate Reordering"
            
            # Create HTML email body
            body = self._create_alert_email(urgent_reorders)
            
            # Send email
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html'))
            
            # Connect to SMTP server
            smtp_server = self.notification_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.notification_config.get('smtp_port', 587)
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            
            print(f"\n✓ Sent urgent reorder alerts to {len(recipients)} recipients")
            
        except Exception as e:
            print(f"\n⚠️  Failed to send email: {e}")
    
    def _create_alert_email(self, urgent_reorders):
        """Create HTML email for urgent alerts."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; }}
                .alert {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; }}
                .urgent {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th {{ background-color: #667eea; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1> Urgent Reorder Alert</h1>
                <p>Automated Demand Forecasting System - {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            
            <div class="urgent" style="margin: 20px;">
                <h2>⚠️ Immediate Action Required</h2>
                <p><strong>{len(urgent_reorders)} SKUs require urgent reordering to prevent stockouts.</strong></p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>SKU</th>
                        <th>Category</th>
                        <th>Current Stock</th>
                        <th>Days to Stockout</th>
                        <th>Recommended Order Qty</th>
                        <th>Urgency</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for item in urgent_reorders:
            html += f"""
                    <tr>
                        <td><strong>{item['sku_id']}</strong></td>
                        <td>{item['category']}</td>
                        <td>{item['current_stock']:,} units</td>
                        <td><strong>{item['days_until_stockout']} days</strong></td>
                        <td>{item['recommended_quantity']:,} units</td>
                        <td><span style="color: #dc3545; font-weight: bold;">HIGH</span></td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <div style="margin: 20px; padding: 15px; background-color: #e7f3ff; border-radius: 5px;">
                <h3> Summary</h3>
                <ul>
                    <li>Total urgent reorders: {}</li>
                    <li>Recommended action: Review and place orders immediately</li>
                    <li>Access dashboard: <a href="http://localhost:8050">View Dashboard</a></li>
                </ul>
            </div>
            
            <div style="margin: 20px; color: #6c757d; font-size: 12px;">
                <p>This is an automated alert from the AI Demand Forecasting System.</p>
                <p>For questions, contact your inventory management team.</p>
            </div>
        </body>
        </html>
        """.format(len(urgent_reorders))
        
        return html
    
    def start_scheduler(self, schedule_time="09:00"):
        """
        Start the automated scheduler.
        
        Args:
            schedule_time: Time to run daily (24-hour format, e.g., "09:00")
        """
        print(f"\n{'='*70}")
        print(f" STARTING AUTOMATED FORECAST SCHEDULER")
        print(f"{'='*70}")
        print(f"\nSchedule: Daily at {schedule_time}")
        print(f" Alerts: {'Enabled' if self.notification_config.get('enabled') else 'Disabled'}")
        print(f"\n Press Ctrl+C to stop the scheduler")
        print(f"{'='*70}\n")
        
        # Schedule daily forecast
        schedule.every().day.at(schedule_time).do(self.run_daily_forecast)
        
        # Run once immediately
        print("Running initial forecast...")
        self.run_daily_forecast()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_on_demand(self):
        """Run forecast immediately (on-demand)."""
        return self.run_daily_forecast()
    
    def get_forecast_history(self):
        """Get history of automated forecasts."""
        return pd.DataFrame(self.forecast_history)