import pandas as pd
from plotly_resampler import FigureResampler
import plotly.graph_objects as go


def plot(df: pd.DataFrame, title: str, x_title: str, y_title: str):
    latex_font = dict(family='tex')
    fig = FigureResampler(go.Figure())
    for col in df.columns:
        fig.add_trace(go.Scattergl(name=col, showlegend=True), hf_x=df.index, hf_y=df[col])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.show_dash(mode='external')
    return df
