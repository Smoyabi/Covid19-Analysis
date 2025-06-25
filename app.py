import dash
from dash import dcc, html, Input, Output, callback, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import base64
import os
import flask
from flask import send_from_directory, abort
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask server first
server = flask.Flask(__name__)

# Load and prepare data
def load_and_prepare_data():
    """Load and clean the COVID-19 dataset"""
    try:
        df = pd.read_csv("Covid_Analysis_Data.csv", parse_dates=['date'])
        
        # Clean data
        df = df.dropna(subset=['date', 'location', 'total_cases', 'total_deaths', 'population'])
        df = df.sort_values(['location', 'date'])
        
        # Calculate additional metrics
        df['case_fatality_rate'] = (df['total_deaths'] / df['total_cases'] * 100).round(2)
        df['cases_per_million'] = (df['total_cases'] / df['population'] * 1_000_000).round(2)
        df['deaths_per_million'] = (df['total_deaths'] / df['population'] * 1_000_000).round(2)
        
        # Calculate daily new cases and deaths
        df['new_cases'] = df.groupby('location')['total_cases'].diff().fillna(0)
        df['new_deaths'] = df.groupby('location')['total_deaths'].diff().fillna(0)
        
        # Calculate 7-day rolling averages
        df['new_cases_7day'] = df.groupby('location')['new_cases'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        df['new_deaths_7day'] = df.groupby('location')['new_deaths'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_and_prepare_data()

# Initialize Dash app with Flask server
app = dash.Dash(__name__, 
                server=server,
                external_stylesheets=[
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
                    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
                ])

app.title = "COVID-19 Professional Dashboard"

# Static file serving routes - Fixed for production
@server.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files"""
    try:
        # Check if file exists in static directory
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        if os.path.exists(os.path.join(static_dir, filename)):
            return send_from_directory(static_dir, filename)
        else:
            print(f"Static file not found: {filename}")
            abort(404)
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
        abort(404)

@server.route("/static/images/<path:filename>")
def serve_images(filename):
    """Serve image files from static/images directory"""
    try:
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
        if os.path.exists(os.path.join(images_dir, filename)):
            return send_from_directory(images_dir, filename)
        else:
            print(f"Image file not found: {filename}")
            abort(404)
    except Exception as e:
        print(f"Error serving image file {filename}: {e}")
        abort(404)

# Define color scheme
colors = {
    'primary': '#1f4e79',
    'secondary': '#2c5aa0', 
    'accent': '#ff6b35',
    'success': '#4caf50',
    'warning': '#ff9800',
    'danger': '#f44336',
    'dark': '#333333',
    'light': '#f8f9fa',
    'white': '#ffffff',
    'gray': '#6c757d'
}

# Custom CSS styles
custom_styles = {
    'dashboard-container': {
        'fontFamily': 'Inter, sans-serif',
        'backgroundColor': colors['light'],
        'minHeight': '100vh',
        'padding': '0',
        'margin': '0'
    },
    'header': {
        'backgroundColor': colors['primary'],
        'color': colors['white'],
        'padding': '20px 0',
        'marginBottom': '30px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
    },
    'header-content': {
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '0 20px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center'
    },
    'title': {
        'fontSize': '2.5rem',
        'fontWeight': '700',
        'margin': '0',
        'color': colors['white']
    },
    'subtitle': {
        'fontSize': '1.1rem',
        'fontWeight': '300',
        'margin': '5px 0 0 0',
        'opacity': '0.9'
    },
    'kpi-container': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
        'gap': '20px',
        'maxWidth': '1200px',
        'margin': '0 auto 40px auto',
        'padding': '0 20px'
    },
    'kpi-card': {
        'backgroundColor': colors['white'],
        'borderRadius': '12px',
        'padding': '24px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.05)',
        'border': '1px solid #e9ecef',
        'transition': 'transform 0.2s ease, box-shadow 0.2s ease'
    },
    'kpi-icon': {
        'fontSize': '2.5rem',
        'marginBottom': '12px',
        'display': 'block'
    },
    'kpi-value': {
        'fontSize': '2.2rem',
        'fontWeight': '700',
        'margin': '8px 0 4px 0',
        'lineHeight': '1.2'
    },
    'kpi-label': {
        'fontSize': '0.95rem',
        'color': colors['gray'],
        'fontWeight': '500',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px'
    },
    'controls-container': {
        'backgroundColor': colors['white'],
        'borderRadius': '12px',
        'padding': '24px',
        'marginBottom': '30px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.05)',
        'maxWidth': '1200px',
        'margin': '0 auto 30px auto',
        'marginLeft': '20px',
        'marginRight': '20px'
    },
    'charts-container': {
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '0 20px'
    },
    'chart-card': {
        'backgroundColor': colors['white'],
        'borderRadius': '12px',
        'padding': '24px',
        'marginBottom': '30px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.05)',
        'border': '1px solid #e9ecef'
    },
    'download-section': {
        'backgroundColor': colors['white'],
        'borderRadius': '12px',
        'padding': '24px',
        'marginBottom': '30px',
        'boxShadow': '0 4px 12px rgba(0,0,0,0.05)',
        'textAlign': 'center',
        'border': f'2px solid {colors["primary"]}'
    },
    'download-button': {
        'backgroundColor': colors['primary'],
        'color': colors['white'],
        'border': 'none',
        'padding': '12px 24px',
        'borderRadius': '8px',
        'fontSize': '1.1rem',
        'fontWeight': '600',
        'cursor': 'pointer',
        'textDecoration': 'none',
        'display': 'inline-block',
        'transition': 'all 0.2s ease'
    },
    'static-chart': {
        'width': '100%',
        'height': 'auto',
        'borderRadius': '8px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'maxWidth': '100%'
    },
    'placeholder-chart': {
        'width': '100%',
        'height': '400px',
        'backgroundColor': colors['light'],
        'borderRadius': '8px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'border': f'2px dashed {colors["gray"]}',
        'color': colors['gray'],
        'fontSize': '1.2rem',
        'fontWeight': '500'
    }
}

def create_kpi_card(icon, value, label, color):
    """Create a KPI card component"""
    return html.Div([
        html.I(className=f"fas fa-{icon}", style={**custom_styles['kpi-icon'], 'color': color}),
        html.Div(value, style={**custom_styles['kpi-value'], 'color': colors['dark']}),
        html.Div(label, style=custom_styles['kpi-label'])
    ], style=custom_styles['kpi-card'])

def format_number(num):
    """Format large numbers with K, M, B suffixes"""
    if pd.isna(num):
        return "N/A"
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:,.0f}"

def check_file_exists(file_path):
    """Check if file exists and return appropriate path"""
    if os.path.exists(file_path):
        return True
    else:
        print(f"File not found: {file_path}")
        return False

def create_placeholder_div(message):
    """Create a placeholder div for missing images"""
    return html.Div(
        message,
        style=custom_styles['placeholder-chart']
    )

# App Layout
app.layout = html.Div([
    # Header Section
    html.Div([
        html.Div([
            html.Div([
                html.H1("COVID-19 Analysis Dashboard", style=custom_styles['title']),
                html.P("Comprehensive Data Analysis & Insights", style=custom_styles['subtitle'])
            ]),
            html.Div([
                html.P(f"Last Updated: {datetime.now().strftime('%B %d, %Y')}", 
                      style={'margin': '0', 'fontSize': '0.9rem', 'opacity': '0.8'})
            ])
        ], style=custom_styles['header-content'])
    ], style=custom_styles['header']),
    
    # KPI Cards Section
    html.Div(id='kpi-cards'),
    
    # Controls Section
    html.Div([
        html.H3("Dashboard Controls", style={'marginBottom': '20px', 'color': colors['primary']}),
        html.Div([
            html.Div([
                html.Label("Select Country:", style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': c, 'value': c} for c in sorted(df['location'].unique())] if not df.empty else [],
                    value='Kenya' if 'Kenya' in df['location'].unique() else (df['location'].iloc[0] if not df.empty else None),
                    style={'marginBottom': '20px'}
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Select Comparison Countries:", style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='comparison-dropdown',
                    options=[{'label': c, 'value': c} for c in sorted(df['location'].unique())] if not df.empty else [],
                    value=['United States', 'India', 'Brazil'] if not df.empty else [],
                    multi=True,
                    style={'marginBottom': '20px'}
                ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        
        html.Div([
            html.Label("Select Date Range:", style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=df['date'].min() if not df.empty else datetime.now() - timedelta(days=365),
                end_date=df['date'].max() if not df.empty else datetime.now(),
                display_format='YYYY-MM-DD',
                style={'marginBottom': '20px'}
            ),
        ], style={'width': '100%', 'marginTop': '20px'}),
        
    ], style=custom_styles['controls-container']),
    
    # PDF Report Download Section
    html.Div([
        html.H3("📊 Professional Report Available", style={'color': colors['primary'], 'marginBottom': '15px'}),
        html.P("Download the comprehensive COVID-19 analysis report with detailed insights and visualizations.", 
               style={'marginBottom': '20px', 'fontSize': '1.1rem'}),
        html.A([
            html.I(className="fas fa-download", style={'marginRight': '8px'}),
            "📥 Download Professional Report (PDF)"
        ], 
        href="/static/Professional_Covid_Report.pdf",
        target="_blank",
        style={
            **custom_styles['download-button'],
            'display': 'block',
            'margin': '20px auto',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'width': 'fit-content'
        })
    ], style={**custom_styles['download-section'], **{'maxWidth': '1200px', 'margin': '0 auto 30px auto', 'marginLeft': '20px', 'marginRight': '20px'}}),
    
    # Static Charts Section
    html.Div([
        html.H2("Report Visualizations", style={'textAlign': 'center', 'color': colors['primary'], 'marginBottom': '30px'}),
        
        # Global Cases and Deaths Chart
        html.Div([
            html.H3("Global COVID-19 Cases and Deaths Overview", style={'color': colors['secondary'], 'marginBottom': '15px'}),
            html.Div(id='global-chart-container'),
            html.P("This visualization shows the global progression of COVID-19 cases and deaths, highlighting key trends and patterns observed throughout the pandemic.", 
                   style={'marginTop': '15px', 'fontStyle': 'italic', 'color': colors['gray']})
        ], style=custom_styles['chart-card']),
        
        # Kenya Trend Chart
        html.Div([
            html.H3("COVID-19 Trends in Kenya", style={'color': colors['secondary'], 'marginBottom': '15px'}),
            html.Div(id='kenya-chart-container'),
            html.P("Detailed analysis of Kenya's COVID-19 trajectory, showcasing the country's unique pandemic experience and response effectiveness.", 
                   style={'marginTop': '15px', 'fontStyle': 'italic', 'color': colors['gray']})
        ], style=custom_styles['chart-card']),
        
        # Correlation Heatmap
        html.Div([
            html.H3("Statistical Correlation Analysis", style={'color': colors['secondary'], 'marginBottom': '15px'}),
            html.Div(id='correlation-chart-container'),
            html.P("Advanced correlation matrix revealing relationships between various COVID-19 metrics and demographic factors.", 
                   style={'marginTop': '15px', 'fontStyle': 'italic', 'color': colors['gray']})
        ], style=custom_styles['chart-card']),
        
    ], style=custom_styles['charts-container']),
    
    # Interactive Charts Section
    html.Div([
        html.H2("Interactive Analysis", style={'textAlign': 'center', 'color': colors['primary'], 'marginBottom': '30px'}),
        
        # Country-specific trends
        html.Div([
            dcc.Graph(id='country-trends')
        ], style=custom_styles['chart-card']),
        
        # Multi-country comparison
        html.Div([
            dcc.Graph(id='multi-country-comparison')
        ], style=custom_styles['chart-card']),
        
        # Global scatter plot
        html.Div([
            dcc.Graph(id='global-scatter')
        ], style=custom_styles['chart-card']),
        
        # Top countries bar chart
        html.Div([
            dcc.Graph(id='top-countries-bar')
        ], style=custom_styles['chart-card']),
        
        # Data table
        html.Div([
            html.H3("Data Table", style={'color': colors['secondary'], 'marginBottom': '15px'}),
            dash_table.DataTable(
                id='data-table',
                columns=[
                    {'name': 'Country', 'id': 'location'},
                    {'name': 'Date', 'id': 'date', 'type': 'datetime'},
                    {'name': 'Total Cases', 'id': 'total_cases', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                    {'name': 'Total Deaths', 'id': 'total_deaths', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                    {'name': 'Cases per Million', 'id': 'cases_per_million', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
                    {'name': 'Case Fatality Rate (%)', 'id': 'case_fatality_rate', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                ],
                sort_action='native',
                filter_action='native',
                page_action='native',
                page_size=10,
                style_cell={'textAlign': 'left', 'fontFamily': 'Inter, sans-serif'},
                style_header={'backgroundColor': colors['primary'], 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': colors['light']
                    }
                ]
            )
        ], style=custom_styles['chart-card']),
        
    ], style=custom_styles['charts-container']),
    
    # Footer
    html.Div([
        html.P([
            "COVID-19 Professional Dashboard • ",
            html.A("Data Sources", href="#", style={'color': colors['primary']}),
            " • ",
            html.A("Methodology", href="#", style={'color': colors['primary']}),
            " • ",
            f"© {datetime.now().year} Sammi Oyabi"
        ], style={'textAlign': 'center', 'color': colors['gray'], 'margin': '40px 0 20px 0'})
    ])
    
], style=custom_styles['dashboard-container'])

# Callback to load static images with better error handling
@app.callback(
    [Output('global-chart-container', 'children'),
     Output('kenya-chart-container', 'children'),
     Output('correlation-chart-container', 'children')],
    [Input('country-dropdown', 'value')]  # Dummy input to trigger callback
)
def load_static_images(selected_country):
    """Load static chart images with improved error handling"""
    
    # Image configurations
    image_configs = [
        {
            'filename': 'global_cases_deaths.png',
            'alt': 'Global Cases and Deaths Chart',
            'placeholder': 'Global COVID-19 Cases and Deaths Chart Not Available'
        },
        {
            'filename': 'kenya_trend.png',
            'alt': 'Kenya Trend Chart',
            'placeholder': 'Kenya COVID-19 Trend Chart Not Available'
        },
        {
            'filename': 'correlation_heatmap.png',
            'alt': 'Correlation Heatmap',
            'placeholder': 'Statistical Correlation Heatmap Not Available'
        }
    ]
    
    containers = []
    
    for config in image_configs:
        # Check if image file exists
        image_path = os.path.join('static', 'images', config['filename'])
        
        if check_file_exists(image_path):
            # Image exists, create img element with URL
            container = html.Img(
                src=f"/static/images/{config['filename']}",
                alt=config['alt'],
                style=custom_styles['static-chart']
            )
        else:
            # Image doesn't exist, create placeholder
            container = create_placeholder_div(config['placeholder'])
        
        containers.append(container)
    
    return containers[0], containers[1], containers[2]

# Callback for KPI cards
@app.callback(
    Output('kpi-cards', 'children'),
    [Input('country-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_kpi_cards(selected_country, start_date, end_date):
    """Update KPI cards based on selected country and date range"""
    if df.empty or not selected_country:
        return []
    
    # Filter data
    filtered_df = df[
        (df['location'] == selected_country) &
        (df['date'] >= start_date) &
        (df['date'] <= end_date)
    ]
    
    if filtered_df.empty:
        return []
    
    latest_data = filtered_df.iloc[-1]
    
    total_cases = latest_data['total_cases']
    total_deaths = latest_data['total_deaths']
    case_fatality_rate = latest_data['case_fatality_rate']
    cases_per_million = latest_data['cases_per_million']
    
    return html.Div([
        create_kpi_card("virus", format_number(total_cases), "Total Cases", colors['primary']),
        create_kpi_card("skull-crossbones", format_number(total_deaths), "Total Deaths", colors['danger']),
        create_kpi_card("percentage", f"{case_fatality_rate:.1f}%", "Case Fatality Rate", colors['warning']),
        create_kpi_card("users", format_number(cases_per_million), "Cases per Million", colors['secondary'])
    ], style=custom_styles['kpi-container'])

# Callback for interactive charts
@app.callback(
    [Output('country-trends', 'figure'),
     Output('multi-country-comparison', 'figure'),
     Output('global-scatter', 'figure'),
     Output('top-countries-bar', 'figure'),
     Output('data-table', 'data')],
    [Input('country-dropdown', 'value'),
     Input('comparison-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_interactive_charts(selected_country, comparison_countries, start_date, end_date):
    """Update all interactive charts"""
    if df.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return empty_fig, empty_fig, empty_fig, empty_fig, []
    
    # Filter data by date range
    date_filtered_df = df[
        (df['date'] >= start_date) &
        (df['date'] <= end_date)
    ]
    
    # Country trends chart
    country_df = date_filtered_df[date_filtered_df['location'] == selected_country]
    
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Cases Over Time', 'Total Deaths Over Time', 
                       'Daily New Cases (7-day avg)', 'Daily New Deaths (7-day avg)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if not country_df.empty:
        fig_trends.add_trace(
            go.Scatter(x=country_df['date'], y=country_df['total_cases'], 
                      name='Total Cases', line=dict(color=colors['primary'])),
            row=1, col=1
        )
        fig_trends.add_trace(
            go.Scatter(x=country_df['date'], y=country_df['total_deaths'], 
                      name='Total Deaths', line=dict(color=colors['danger'])),
            row=1, col=2
        )
        fig_trends.add_trace(
            go.Scatter(x=country_df['date'], y=country_df['new_cases_7day'], 
                      name='New Cases (7-day avg)', line=dict(color=colors['secondary'])),
            row=2, col=1
        )
        fig_trends.add_trace(
            go.Scatter(x=country_df['date'], y=country_df['new_deaths_7day'], 
                      name='New Deaths (7-day avg)', line=dict(color=colors['accent'])),
            row=2, col=2
        )
    
    fig_trends.update_layout(
        title=f'COVID-19 Trends - {selected_country}',
        height=600,
        showlegend=False,
        font=dict(family="Inter, sans-serif")
    )
    
    # Multi-country comparison
    comparison_df = date_filtered_df[date_filtered_df['location'].isin(comparison_countries or [])]
    
    fig_comparison = px.line(
        comparison_df,
        x='date',
        y='total_cases',
        color='location',
        title='Multi-Country Comparison - Total Cases',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_comparison.update_layout(font=dict(family="Inter, sans-serif"))
    
    # Global scatter plot
    latest_date = date_filtered_df['date'].max()
    latest_global_df = date_filtered_df[date_filtered_df['date'] == latest_date]
    
    fig_scatter = px.scatter(
        latest_global_df,
        x='population',
        y='total_deaths',
        color='location',
        size='total_cases',
        hover_name='location',
        log_x=True,
        title=f'Total Deaths vs Population - {latest_date.strftime("%Y-%m-%d")}',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_scatter.update_layout(font=dict(family="Inter, sans-serif"))
    
    # Top countries bar chart
    top_countries = latest_global_df.nlargest(15, 'total_cases')
    
    fig_bar = px.bar(
        top_countries,
        x='location',
        y='total_cases',
        title='Top 15 Countries by Total Cases',
        color='total_cases',
        color_continuous_scale='Viridis'
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        font=dict(family="Inter, sans-serif")
    )
    
    # Data table
    table_data = latest_global_df[['location', 'date', 'total_cases', 'total_deaths', 
                                  'cases_per_million', 'case_fatality_rate']].to_dict('records')
    
    return fig_trends, fig_comparison, fig_scatter, fig_bar, table_data

if __name__ == '__main__':
    # Get port from environment variable or default to 8050
    port = int(os.environ.get("PORT", 8050))
    
    # Set debug based on environment
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    
    print(f"Starting COVID-19 Dashboard on port {port}")
    print(f"Debug mode: {debug_mode}")
    
    # Check if critical files exist
    required_files = [
        "Covid_Analysis_Data.csv",
        "static/Professional_Covid_Report.pdf",
        "static/images/global_cases_deaths.png",
        "static/images/kenya_trend.png", 
        "static/images/correlation_heatmap.png"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)