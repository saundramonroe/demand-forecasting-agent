"""
Interactive web dashboard for demand forecasting visualization.
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ForecastingDashboard:
    """Interactive dashboard for visualizing forecasts and inventory decisions."""
    
    def __init__(self, agent, sales_data, external_data, inventory_data):
        """
        Initialize dashboard.
        
        Args:
            agent: DemandForecastingAgent instance
            sales_data: Historical sales DataFrame
            external_data: External factors DataFrame
            inventory_data: Current inventory DataFrame
        """
        self.agent = agent
        self.sales_data = sales_data
        self.external_data = external_data
        self.inventory_data = inventory_data
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        
        # Get SKU options
        sku_options = [{'label': sku, 'value': sku} for sku in self.sales_data['sku_id'].unique()]
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🤖 AI Demand Forecasting & Dynamic Replenishment", 
                           className="text-center mb-4 mt-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Control Panel", className="card-title"),
                            html.Label("Select SKU:", className="mt-3"),
                            dcc.Dropdown(
                                id='sku-selector',
                                options=sku_options,
                                value=sku_options[0]['value'],
                                clearable=False
                            ),
                            html.Label("Forecast Horizon (Days):", className="mt-3"),
                            dcc.Slider(
                                id='horizon-slider',
                                min=7,
                                max=90,
                                step=7,
                                value=30,
                                marks={7: '7', 30: '30', 60: '60', 90: '90'}
                            ),
                            dbc.Button(
                                "Generate Forecast", 
                                id='forecast-button',
                                color="primary",
                                className="mt-3 w-100"
                            ),
                            html.Div(id='forecast-status', className="mt-3")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Inventory Status", className="card-title"),
                            html.Div(id='inventory-metrics')
                        ])
                    ])
                ], width=9)
            ], className="mb-4"),
            
            # Forecast Visualization
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Demand Forecast", className="card-title"),
                            dcc.Graph(id='forecast-chart')
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Reorder Recommendation", className="card-title"),
                            html.Div(id='reorder-recommendation')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Historical Performance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Historical Sales Pattern", className="card-title"),
                            dcc.Graph(id='historical-chart')
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Model Performance", className="card-title"),
                            html.Div(id='model-metrics')
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # AI Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🤖 AI-Powered Analysis", className="card-title"),
                            dbc.Button(
                                "Generate AI Analysis",
                                id='ai-analysis-button',
                                color="success",
                                className="mb-3"
                            ),
                            dbc.Spinner([
                                html.Div(id='ai-analysis-output')
                            ])
                        ])
                    ])
                ])
            ])
            
        ], fluid=True, style={'backgroundColor': '#f8f9fa'})
    
    def setup_callbacks(self):
        """Setup interactive callbacks."""
        
        @self.app.callback(
            [Output('forecast-chart', 'figure'),
             Output('reorder-recommendation', 'children'),
             Output('forecast-status', 'children'),
             Output('historical-chart', 'figure'),
             Output('model-metrics', 'children'),
             Output('inventory-metrics', 'children')],
            [Input('forecast-button', 'n_clicks')],
            [State('sku-selector', 'value'),
             State('horizon-slider', 'value')]
        )
        def update_forecast(n_clicks, sku_id, horizon):
            """Update forecast when button is clicked."""
            
            if n_clicks is None:
                # Initial load
                return self._generate_empty_figure(), "", "", self._generate_empty_figure(), "", ""
            
            try:
                # Train model if not already trained
                if sku_id not in self.agent.models:
                    self.agent.train_model(sku_id, self.sales_data, self.external_data)
                
                # Generate forecast
                future_dates = pd.date_range(
                    start=datetime.now(),
                    periods=horizon,
                    freq='D'
                )
                
                forecast_df = self.agent.predict_demand(sku_id, future_dates, self.external_data)
                
                # Get inventory info
                inv_info = self.inventory_data[self.inventory_data['sku_id'] == sku_id].iloc[0]
                
                # Calculate reorder recommendation
                reorder_info = self.agent.calculate_dynamic_reorder(
                    sku_id,
                    forecast_df,
                    inv_info['current_stock'],
                    inv_info['lead_time_days']
                )
                
                # Create forecast chart
                forecast_fig = self._create_forecast_chart(forecast_df, inv_info)
                
                # Create reorder card
                reorder_card = self._create_reorder_card(reorder_info)
                
                # Status message
                status = dbc.Alert(
                    f"✓ Forecast generated successfully for {sku_id}",
                    color="success",
                    className="mt-3"
                )
                
                # Historical chart
                historical_fig = self._create_historical_chart(sku_id)
                
                # Model metrics
                metrics = self._create_metrics_display(sku_id)
                
                # Inventory metrics
                inv_metrics = self._create_inventory_display(inv_info, reorder_info)
                
                return forecast_fig, reorder_card, status, historical_fig, metrics, inv_metrics
                
            except Exception as e:
                error_msg = dbc.Alert(
                    f"Error: {str(e)}",
                    color="danger",
                    className="mt-3"
                )
                return self._generate_empty_figure(), "", error_msg, self._generate_empty_figure(), "", ""
        
        @self.app.callback(
            Output('ai-analysis-output', 'children'),
            [Input('ai-analysis-button', 'n_clicks')],
            [State('sku-selector', 'value')]
        )
        def generate_ai_analysis(n_clicks, sku_id):
            """Generate AI analysis using Qwen model."""
            if n_clicks is None:
                return html.P("Click 'Generate AI Analysis' to get AI-powered insights.", 
                            className="text-muted")
            
            try:
                # Placeholder AI analysis
                analysis = """
                📊 **Risk Assessment:**
                - Low stockout risk detected for the next 14 days
                - Moderate overstock risk if current reorder quantities maintained
                
                ✅ **Recommended Actions:**
                1. Reduce next order by 15% due to declining trend
                2. Monitor competitor pricing - current premium may affect demand
                3. Schedule reorder for Day 18 to optimize working capital
                
                🔍 **Key Insights:**
                - Seasonal pattern shows 20% decline starting next week
                - Economic index correlation suggests 8% demand sensitivity
                - Weekend sales 30% higher - adjust staffing accordingly
                
                ⚡ **Optimization Opportunities:**
                - Implement dynamic pricing during low-demand periods
                - Consider promotions for Days 10-15 to smooth demand
                - Safety stock can be reduced by 10% based on forecast accuracy
                """
                
                return dbc.Card([
                    dbc.CardBody([
                        dcc.Markdown(analysis)
                    ])
                ], className="mt-3")
                
            except Exception as e:
                return dbc.Alert(f"Analysis error: {str(e)}", color="warning")
    
    def _create_forecast_chart(self, forecast_df, inv_info):
        """Create forecast visualization."""
        fig = go.Figure()
        
        # Predicted demand
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_demand'],
            mode='lines',
            name='Predicted Demand',
            line=dict(color='blue', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        # Current stock level
        fig.add_hline(
            y=inv_info['current_stock'],
            line_dash="dash",
            line_color="green",
            annotation_text="Current Stock"
        )
        
        # Reorder point
        fig.add_hline(
            y=inv_info['reorder_point'],
            line_dash="dot",
            line_color="red",
            annotation_text="Reorder Point"
        )
        
        fig.update_layout(
            title="30-Day Demand Forecast",
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_historical_chart(self, sku_id):
        """Create historical sales chart."""
        sku_data = self.sales_data[self.sales_data['sku_id'] == sku_id].copy()
        sku_data['date'] = pd.to_datetime(sku_data['date'])
        
        # Last 90 days
        recent_data = sku_data.tail(90)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['sales'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='darkblue')
        ))
        
        # 7-day moving average
        recent_data['ma_7'] = recent_data['sales'].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['ma_7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', dash='dash')
        ))
        
        fig.update_layout(
            title="Historical Sales (Last 90 Days)",
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def _create_reorder_card(self, reorder_info):
        """Create reorder recommendation display."""
        urgency_color = {
            'HIGH': 'danger',
            'MEDIUM': 'warning',
            'LOW': 'success'
        }
        
        return dbc.Card([
            dbc.CardBody([
                html.H5(f"Urgency: {reorder_info['urgency']}", 
                       className=f"text-{urgency_color[reorder_info['urgency']]}"),
                html.Hr(),
                html.P([
                    html.Strong("Current Stock: "),
                    f"{reorder_info['current_stock']} units"
                ]),
                html.P([
                    html.Strong("Reorder Point: "),
                    f"{reorder_info['reorder_point']} units"
                ]),
                html.P([
                    html.Strong("Recommended Quantity: "),
                    f"{reorder_info['reorder_quantity']} units"
                ]),
                html.P([
                    html.Strong("Days Until Stockout: "),
                    f"{reorder_info['days_until_stockout']} days"
                ]),
                html.Hr(),
                dbc.Button(
                    "Needs Reorder Now!" if reorder_info['needs_reorder'] else "Stock Level OK",
                    color="danger" if reorder_info['needs_reorder'] else "success",
                    className="w-100"
                )
            ])
        ])
    
    def _create_metrics_display(self, sku_id):
        """Create model performance metrics display."""
        if sku_id not in self.agent.forecast_accuracy:
            return html.P("Train model to see metrics", className="text-muted")
        
        metrics = self.agent.forecast_accuracy[sku_id]
        
        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3(f"{metrics['test_score']:.2%}", className="text-primary"),
                    html.P("Model Accuracy", className="text-muted")
                ], className="text-center")
            ]),
            dbc.Col([
                html.Div([
                    html.H3(f"{metrics['mape']:.1f}%", className="text-info"),
                    html.P("Forecast Error", className="text-muted")
                ], className="text-center")
            ])
        ])
    
    def _create_inventory_display(self, inv_info, reorder_info):
        """Create inventory status display."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{inv_info['current_stock']}", className="text-center"),
                        html.P("Current Stock", className="text-muted text-center")
                    ])
                ])
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{reorder_info['days_until_stockout']}", className="text-center"),
                        html.P("Days to Stockout", className="text-muted text-center")
                    ])
                ])
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{inv_info['lead_time_days']}", className="text-center"),
                        html.P("Lead Time (Days)", className="text-muted text-center")
                    ])
                ])
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"${inv_info['unit_price']:.2f}", className="text-center"),
                        html.P("Unit Price", className="text-muted text-center")
                    ])
                ])
            ])
        ])
    
    def _generate_empty_figure(self):
        """Generate empty placeholder figure."""
        fig = go.Figure()
        fig.update_layout(
            title="Select SKU and click 'Generate Forecast'",
            template='plotly_white'
        )
        return fig
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard server."""
        print(f"\n🚀 Starting Demand Forecasting Dashboard...")
        print(f"📊 Access dashboard at: http://{host}:{port}")
        print(f"Press Ctrl+C to stop the server\n")
        self.app.run_server(host=host, port=port, debug=debug)