import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import jax.numpy as jnp
from jax import vmap

def plot_predictions(u, a, x, t, *models, port=8050, **kwargs):
    """Displays the predictions and errors of a list of models on a given sample.
    Pressing the button generates the animation for a new sample

    Args:
        u (n_samples, t_dim, x_dim): ground truth solutions
        a (n_samples, x_dim): input to models
        x (x_dim): spatial points
        t (t_dim): time points
    """
    app = dash.Dash(external_stylesheets=[dbc.themes.MATERIA])

    app.layout = dbc.Container(
        [
            html.H1("Predictions and Errors"),
            dbc.Button("New sample",color="primary", id="button"),
            dbc.Tabs(
                [
                    dbc.Tab(label="Predictions", tab_id="predictions"),
                    dbc.Tab(label="Errors", tab_id="errors"),
                    dbc.Tab(label="Both", tab_id="both"),
                ],
                id="tabs",
                active_tab="predictions",
            ),
            dbc.Spinner(
                [
                    dcc.Store(id="store"),
                    html.Div(id="tab-content", className="p-4"),
                ],
                delay_show=100,
            ),
        ]
    )

    @app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "active_tab"), Input("store", "data")],
    )
    def render_tab_content(active_tab, data):
        if active_tab is not None and data:
            if active_tab == "predictions":
                return dcc.Graph(figure=data["Predictions"])
            elif active_tab == "errors":
                return dcc.Graph(figure=data["Errors"])
            elif active_tab == "both":
                return dcc.Graph(figure=data["Both"])
        return "No tab selected"

    @app.callback(
        Output("store", "data"),
        [Input("button", "n_clicks")]
    )
    def generate_graphs(n):
        if not n:
            n = 0
        # Select a new function based on the button click
        a_sample = a[n]
        u_sample = u[n].ravel()

        model_names = kwargs.get("model_names", [f"Model {i}" for i in range(len(models))])

        df = pd.DataFrame({
            "Ground Truth": u_sample,
            "x": jnp.tile(round(x, 2), len(t)),
            "t": jnp.repeat(round(t, 2), len(x))
        })

        prediction_columns = ["Ground Truth"] + model_names
        error_columns = [f"Error {model_name}" for model_name in model_names]

        for i, model in enumerate(models):
            #prediction = vmap(vmap(model, (None, 0, None)), (None, None, 0))(a_sample, x, t).ravel()
            #prediction = vmap(model, (None, None, 0))(a_sample, x, t).ravel()
            prediction = model.predict_whole_grid(a_sample, x, t).ravel()
            error = jnp.abs(prediction - u_sample)
            df[model_names[i]] = prediction
            df[f"Error {model_names[i]}"] = error
            
        y_min = df[prediction_columns].min().min()
        y_max = df[prediction_columns].max().max()

        animations = {
            "Predictions": px.line(df, x='x', y=prediction_columns, animation_frame='t', range_y=[y_min, y_max],
                                   labels={'t': 'Time (t)', 'u': 'u(x, t)'}),
            "Errors": px.line(df, x='x', y=error_columns, animation_frame='t', range_y=[0, df[error_columns].max().max()],
                              labels={'t': 'Time (t)', 'u': 'u(x, t)'}),
            "Both": px.line(df, x='x', y=prediction_columns + error_columns, animation_frame='t', range_y=[y_min, y_max],
                            labels={'t': 'Time (t)', 'u': 'u(x, t)'}),
        }

        for key, fig in animations.items():
            fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 20
            for data in fig.data:
                if key == "Both":
                    if data.name.startswith("Error"):
                        data.line.dash = 'dash'
                    elif data.name == 'Ground Truth':
                        data.line.dash = 'dot'
                        data.line.color = 'black'
                elif key == "Predictions":
                    if data.name == 'Ground Truth':
                        data.line.dash = 'dot'
                        data.line.color = 'black'

            for frame in fig.frames:
                for data in frame.data:
                    if key == "Both":
                        if data.name.startswith("Error"):
                            data.line.dash = 'dash'
                        elif data.name == 'Ground Truth':
                            data.line.dash = 'dot'
                            data.line.color = 'black'
                    elif key == "Predictions":
                        if data.name == 'Ground Truth':
                            data.line.dash = 'dot'
                            data.line.color = 'black'

        return animations


    app.run(port = port)

def plot_self_adaptive_weights(*trainers, port = 8050, **kwargs):
    """Displays the self-adaptive weights of a list of trainers."""
    
    app = dash.Dash(external_stylesheets=[dbc.themes.MATERIA])
    
    trainer_names = kwargs.get("trainer_names", [f"Trainer {i}" for i in range(len(trainers))])

    app.layout = dbc.Container(
        [
            html.H1("Self-Adaptive Weights"),
            dbc.Tabs(
                [dbc.Tab(label=f"{trainer_names[i]}", tab_id=f"{i}") for i in range(len(trainers))],
                id="tabs",
            ),
            dbc.Spinner(
                [html.Div(id="tab-content", className="p-4"),],
                delay_show=100,
            ),
        ]
    )

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "active_tab"),
    )
    def render_tab_content(active_tab):
        if active_tab is not None:
            return dcc.Graph(figure=px.imshow(trainers[int(active_tab)].λ_history, 
                                              animation_frame=0,
                                              labels={"animation_frame": "Epoch"}))
            
        return "No tab selected"

    app.run(port=port)
    
def plot_loss(trainer, port=8050):
        """Plots the loss history of the training.
        """
        app = dash.Dash(external_stylesheets=[dbc.themes.MATERIA])
        # Define layout
        config = {
            'toImageButtonOptions': {
                'format': 'svg', 
                'filename': 'loss_plot',
        }}
                
        app.layout = dbc.Container([
            dbc.Row(dcc.Graph(id="loss-plot", config=config)),
            dbc.Row([
                    dbc.Col(dbc.Switch(
                        id="toggle-markers",
                        label="Show Markers",
                        value=True,  # Default to show markers
                    ),width=3),
                    dbc.Col(dbc.Switch(
                        id="toggle-batch-loss",
                        label="Show Batch Loss",
                        value=True,  # Default to show batch loss
                    ),width=3)], 
                    justify='start')
                    ],
                )
        # Callback to update the plot based on toggle state
        @app.callback(
            Output("loss-plot", "figure"),
            [Input("toggle-markers", "value"),
            Input("toggle-batch-loss", "value")]
        )
        def update_plot(show_markers, show_batch):
            train_color = '#bc5090'
            val_color = '#ffa600'
            
            fig = go.Figure()

            # Add batch loss traces if show_batch is enabled
            if show_batch:
                fig.add_trace(go.Scatter(
                    x=jnp.linspace(0, trainer.epochs_trained - 1, len(trainer.train_loss_history_batch)),
                    y=trainer.train_loss_history_batch,
                    mode='lines',
                    line=dict(color=train_color, width=1),
                    opacity=0.3,
                    name='Batch',
                    legendgroup='train'
                ))
                fig.add_trace(go.Scatter(
                    x=jnp.linspace(0, trainer.epochs_trained - 1, len(trainer.val_loss_history_batch)),
                    y=trainer.val_loss_history_batch,
                    mode='lines',
                    line=dict(color=val_color, width=1),
                    opacity=0.3,
                    name='Batch',
                    legendgroup='validation'
                ))

            # Add epoch loss traces with optional markers
            fig.add_trace(go.Scatter(
                x=jnp.linspace(0, trainer.epochs_trained - 1, len(trainer.val_loss_history)),
                y=trainer.val_loss_history,
                line=dict(color=val_color, width=2),
                name='Epoch',
                mode='lines+markers' if show_markers else 'lines',
                legendgroup='validation',
                legendgrouptitle=dict(text='Validation'),
            ))

            fig.add_trace(go.Scatter(
                x=jnp.arange(len(trainer.train_loss_history)),
                y=trainer.train_loss_history,
                line=dict(color=train_color, width=2),
                name='Epoch',
                mode='lines+markers' if show_markers else 'lines',
                legendgroup='train',
                legendgrouptitle=dict(text='Train'),
            ))
            

            fig.update_layout(
                title='Loss History',
                legend=dict(x=1, y=1),
                template='seaborn',
                xaxis=dict(title ="Epoch", range=[0, trainer.epochs_trained - 1]),
                yaxis=dict(title ="Loss", type='log', tickformat = '.1e'),
            )

            return fig

        app.run(port=port)