# Plotly Dash Season Playoff Dashboard
# 
# Project Description:
# 
# The below dashboard allows the user to input NHL team stats and have returned season and playoff outcome predictions.

# Import packages
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_auth

# import numpy as np
import pandas as pd
# from pickle import dump
from pickle import load
# from sklearn.preprocessing import StandardScaler
from textwrap import dedent

# import plotly.offline as pyo
import plotly.graph_objs as go
# import plotly.figure_factory as ff
# from plotly import tools


######################################################################################################################
# NEW BLOCK - Pre app setup
######################################################################################################################

# Set login credentials
USERNAME_PASSWORD_PAIRS = [['data','analyst']]

# establish app
app = dash.Dash(
    __name__,
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    )

# Set login credentials
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server

question1 = "Save percentage?, e.g., 89.50"
question2 = "Average saves per game?, e.g., 24.94"
question3 = "Shooting percentage?, e.g., 9.52"
question4 = "Average shots per game that don't lead to goals?, e.g., 28.53"
question5 = "Penalty kill percentage?, e.g., 84.40"


######################################################################################################################
# NEW BLOCK - App layout
######################################################################################################################

# Set app layout
app.layout = html.Div([
    
    # Header
    html.Div([
        html.H2(
            'Season & Playoff Predictions Dashboard',
            # Font/background styling
            style = {
                'padding':10,
                'margin':0,
                'font-family':'Arial, Helvetica, sans-serif',
                'background':'#00008B',
                'color':'#FFFFFF',
                'textAlign':'center'
            }
        )
    ]),
    
    # Historical data for chosen field
    html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    html.Div([
                        html.H3("Season & Playoff Predictions Form",
                                # Font styling
                                style = {
                                    'padding-bottom': 69.5,
                                    'padding-top': 10
                                }),
                        
                        # Question 1
                        dbc.Row([
                            dbc.Input(
                                id = "input1",
                                placeholder = question1,
                                type = "number"
                            )
                            # Row styling
                        ], style = {
                            'padding-bottom': 10,
                            'padding-left': 10,
                            'padding-right': 10
                        }),
                        
                        # Question 2
                        dbc.Row([
                            dbc.Input(
                                id = "input2",
                                placeholder = question2,
                                type = "number"
                            )
                        # Row styling
                        ], style = {
                            'padding-bottom': 10,
                            'padding-left': 10,
                            'padding-right': 10
                        }),
                        
                        # Question 3
                        dbc.Row([
                            dbc.Input(
                                id = "input3",
                                placeholder = question3,
                                type = "number"
                            )
                        # Row styling
                        ], style = {
                            'padding-bottom': 10,
                            'padding-left': 10,
                            'padding-right': 10
                        }),
                        
                        # Question 4
                        dbc.Row([
                            dbc.Input(
                                id = "input4",
                                placeholder = question4,
                                type = "number"
                            )
                        # Row styling
                        ], style = {
                                'padding-bottom': 10,
                                'padding-left': 10,
                                'padding-right': 10
                            }),
                        
                        # Question 5
                        dbc.Row([
                            dbc.Input(
                                id = "input5",
                                placeholder = question5,
                                type = "number"
                            )
                        # Row styling
                        ], style = {
                                'padding-bottom': 10,
                                'padding-left': 10,
                                'padding-right': 10
                            }),
                        
                        # Submit button
                        dbc.Row([
                            html.Button(
                                'Submit',
                                id = 'submit-val'
                            )
                        # Row styling
                        ], style = {
                            'padding-top': 20,
                            'padding-left': 10,
                            'padding-right': 10,
                            'padding-bottom': 73
                        })
                        
                    ]), 
                    body = True,
                    color = "dark",
                    outline = True
                    
                ),
                width = {
                    "size": 5,
                    "order": 1
                }),
            
            # Season Adjusted Wins Deviation prediction
            dbc.Col([
                dbc.Row([
                    dbc.Card(
                        html.H3(
                            id = 'games',
                            # Font styling
                            style = {
                                'padding': 10,
                                'margin': 0,
                                'font-family': 'Arial, Helvetica, sans-serif',
                                'color': 'black',
                                'textAlign': 'center'
                            }),
                        body = True,
                        color = "dark",
                        outline = True
                    )
                # Row styling
                ], style = {'padding-bottom': 10,
                            'padding-right': 20
                           }),
                
                # Gauge
                dbc.Row([
                    dbc.Card(
                        dcc.Graph(
                            id = 'gauge'
                        ),
                        body = True, 
                        color = "dark", 
                        outline = True
                    )
                # Row styling
                ], style = {
                    'padding-right': 20
                })
            ],width = {
                "size": 7,
                "order": 2
            })
        ])
    # Div styling
    ], style = {
        'padding-left': 20,
        'padding-right': 20,
        'padding-top': 40,
        'padding-bottom': 40
    }),
    
    # Instructions
    html.Div([
        html.H1(
            'Instructions',
            # Font/background styling
            style = {
                'padding':10,
                'margin':0,
                'font-family':'Arial, Helvetica, sans-serif',
                'background':'#00008B',
                'color':'#FFFFFF',
                'textAlign':'center'
            }),
        
        html.Div(
            dcc.Markdown(
                dedent('''
                Enter team stats into the Season & Playoff Predictions Form and click submit to view season and 
                playoff predictions. Below are the stats' formulas.
                
                * **Save Percentage** - Total Saves / Total Shots Against
                
                * **Average Saves Per Game** - Total Saves / Total Games Played
                
                * **Shooting Percentage** - Total Goals / Total Shots For
                
                * **Average Shots Per Game That Don't Lead to Goals** - (Total Shots / Total Games) * (1 - (Shooting Percentage / 100))
                
                * **Penalty Kill Percentage** - Total Successful Penalty Kills / Total Penalty Kills
                
                * **Season Adjusted Wins Deviation** - (Total Team Wins + (Total Team Ties / 2)) - ((Total League Wins + (Total League Ties / 2)) / Total Number of Teams)
                ''')
            ),
            # div styling
            style = {
                'padding':40,
                'font-family':'Arial, Helvetica, sans-serif',
                'line-height':30,
                'textAlign':'left',
                'fontSize':20
            })
    ]),
    
    # Final ending block
    html.Div([
        html.H1('',
                # Font/background styling
                style = {
                    'padding':30,
                    'margin':0,
                    'font-family':'Arial, Helvetica, sans-serif',
                    'background':'#00008B',
                    'color':'#FFFFFF',
                    'textAlign':'center'
                })
    ])
# Dive styling
], style = {
    'margin':0
})


######################################################################################################################
# NEW BLOCK - Callback functions
######################################################################################################################

# Return results function
@app.callback(Output('games', 'children'),
              Output('gauge', 'figure'),
              Input('submit-val', 'n_clicks'),
              State('input1', 'value'),
              State('input2', 'value'),
              State('input3', 'value'),
              State('input4', 'value'),
              State('input5', 'value'))

def compute(n_clicks, input1, input2, input3, input4, input5):
    
    # Preset inputs
    if input1 == None:
        input1 = 89.50
    if input2 == None:
        input2 = 24.94
    if input3 == None:
        input3 = 9.52
    if input4 == None:
        input4 = 28.53
    if input5 == None:
        input5 = 84.40
        
    # Create data frame form input values
    X_dict = [{
        'penalty_kill_percentage': input5,
        'shooting_pctg': input3,
        'save_pctg': input1,
        'failed_shots_per_game': input4,
        'saves_per_game': input2
    }]
    
    X_df = pd.DataFrame(X_dict)
    
    # Linear regression
    #############################################################################################
    # Load regression model
    model_r = load(open('Models/NHL_Season_Wins_Linear_Regression_Model.pkl', 'rb'))
    # load the model
    scaler_lin = load(open('Models/Scaler_Lin.pkl', 'rb'))
    
    # Scale X values (converts X values to numpy 2D array)
    X = scaler_lin.transform(X_df)
    # Predict aboveMeanAdjWins with input values
    y_pred_r = model_r.predict(X)
    
    # Create data frame with predicted aboveMeanAdjWins values
    y_pred_r = pd.DataFrame(y_pred_r, columns = ['seasonAdjWinsDev'])
    
    # Logistic regression
    #############################################################################################
    # load logistic regression model
    model_l = load(open('Models/NHL_Playoffs_Logistic_Regression_Model.pkl', 'rb'))
    # load the model
    scaler_logi = load(open('Models/Scaler_Logi.pkl', 'rb'))

    # Scale X values (converts X values to numpy 2D array)
    X = X_df.drop(
        ['penalty_kill_percentage'],
        axis = 1,
        errors = 'ignore'
    )
    
    X = scaler_logi.transform(X)
    
    # Creates a data frame from the prediction probabilities
    proba = model_l.predict_proba(X)
    proba = pd.DataFrame(proba, columns = ['0', 'predicted_proba'])
    proba['predicted_proba_making_playoffs'] = (1 - proba['predicted_proba']) * 100
    proba = proba[['predicted_proba_making_playoffs']]

    # Concatenate both data frames
    results = pd.concat([y_pred_r, proba], axis = 1)
    
    # predAboveAvgAdjWins
    prediction = results.iloc[0,1]

    # Set prediction threshold colour
    if prediction < 50:
        colour = 'red'
    else:
        colour = 'green'

    # Create gauge
    fig = go.Figure(data = go.Indicator(mode = "gauge+number",
                                        value = prediction,
                                        domain = {'x': [0, 1], 'y': [0, 1] },
                                        title = {'text': "Predicted Precentage to Make Playoffs"},
                                        gauge = {
                                            'axis': {
                                                'range': [0, 100], 
                                                'tickwidth': 1, 
                                                'tickcolor': "black"
                                            },
                                            'bar': {'color': colour},
                                            'threshold': {
                                                'line': {'color': "black", 'width': 5},
                                                'thickness': 1,
                                                'value': 50
                                            }
                                        }),
                    layout = go.Layout(height = 379)
                   )
    
    # predAboveMeanAdjWins prediction via string
    pred_above_mean_adj_wins = "Predicted Season Adjusted Wins Deviation: " + str(round(results.iloc[0,0],2))

    return pred_above_mean_adj_wins, fig


if __name__ == '__main__':
    app.run_server()
