import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"C:\Users\Harinidivakar\Downloads\housing_cleaned_fixed.csv")

# Train model
X = df[['bedrooms', 'area']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dcc.Store(id="dark-mode-store", data={"dark": False}),  # Store for theme persistence

    dbc.Row([
        dbc.Col(html.H1("🏡 Housing Price Prediction", className="text-center fw-bold"), width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Select Location:"),
            dcc.Dropdown(
                id='location-dropdown',
                options=[{'label': loc, 'value': loc} for loc in sorted(df['location'].unique())],
                value=df['location'].unique()[0],
                searchable=True,
                className="animated-dropdown"
            ),
        ], md=6),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Select Bedrooms:"),
            dcc.Slider(
                id='bedrooms-slider',
                min=int(df['bedrooms'].min()), max=int(df['bedrooms'].max()), step=1, value=2,
                marks={i: str(i) for i in range(int(df['bedrooms'].min()), int(df['bedrooms'].max())+1)},
                className="animated-slider"
            ),
        ], md=6),
        dbc.Col([
            html.Label("Enter Area (sq ft):"),
            dcc.Input(id='area-input', type='number', value=1000, step=10, className="form-control animated-input"),
        ], md=6),
    ], className="mb-3"),

    dbc.Button("Predict Price", id='predict-button', color="primary", className="animated-button mt-2 mb-3"),
    html.Div(id='loading-output', className="text-warning mt-2"),
    html.H2(id='prediction-output', className="text-success mt-3 fw-bold"),

    dbc.Row([
        dbc.Col(dcc.Graph(id='price-distribution', animate=True), md=6),
        dbc.Col(dcc.Graph(id='area-price-scatter', animate=True), md=6),
    ]),

    # Dark Mode Toggle Button
    dbc.Button("Toggle Dark Mode", id="theme-button", color="dark", className="mt-3 animated-button"),
], fluid=True, id="main-container", className="p-4")

# Dark Mode Callback
@app.callback(
    Output("main-container", "className"),
    Output("dark-mode-store", "data"),
    Input("theme-button", "n_clicks"),
    State("dark-mode-store", "data"),
    prevent_initial_call=True
)
def toggle_theme(n_clicks, data):
    dark = not data["dark"]
    new_class = "p-4 bg-dark text-light transition-all" if dark else "p-4 bg-light text-dark transition-all"
    return new_class, {"dark": dark}

# Prediction Callback with Loading Effect
@app.callback(
    Output('loading-output', 'children'),
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('bedrooms-slider', 'value'),
    State('area-input', 'value'),
    prevent_initial_call=True
)
def predict_price(n_clicks, bedrooms, area):
    if area is None or bedrooms is None:
        return "", "❌ Please enter valid inputs."

    # Simulate loading effect
    time.sleep(1)

    predicted_price = model.predict(pd.DataFrame([[bedrooms, area]], columns=['bedrooms', 'area']))[0]
    return "", f"🏠 Estimated House Price: ${predicted_price:,.2f}"

# Graph Updates with Animations
@app.callback(
    Output('price-distribution', 'figure'),
    Input('location-dropdown', 'value')
)
def update_price_distribution(selected_location):
    filtered_df = df[df['location'] == selected_location]
    fig = px.histogram(filtered_df, x='price', nbins=20, title=f"Price Distribution ")
    fig.update_layout(transition_duration=500)
    return fig

@app.callback(
    Output('area-price-scatter', 'figure'),
    Input('location-dropdown', 'value')
)
def update_area_price_scatter(selected_location):
    filtered_df = df[df['location'] == selected_location]
    fig = px.scatter(filtered_df, x='area', y='price', title=f"Area vs Price ", trendline="ols")
    fig.update_layout(transition_duration=500)
    return fig

# Run app on external network
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)
