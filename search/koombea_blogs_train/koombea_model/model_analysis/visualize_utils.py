from typing import Any, Dict
import plotly.graph_objects as go
import pandas as pd


def plot_accuracies_information_df(accuracies_df: pd.DataFrame, title: str):
    # gather data and configurations for plot
    x = accuracies_df.index.values
    y = accuracies_df["accuracy_results"].values
    text = ["{}%".format(y_value * 100) for y_value in y]
    # get plotly figure
    fig = go.Figure(
        data=[
            go.Bar(x=x, y=y, text=text, textposition="auto", name="accuracy_for_tops")
        ]
    )
    fig.update_layout(xaxis_tickangle=-45, title_text=title)
    fig.update_xaxes(title_text="tops result")
    fig.update_yaxes(title_text="accuracy percentage")
    return fig


def plot_blogs_embedding_information(data_plot_config: Dict[str, Any], **plotly_kwargs):
    blogs_idx = data_plot_config["blogs_idx_selected"]
    fig = go.Figure(
        data=go.Scatter(
            x=data_plot_config["tsne"][blogs_idx, 0],
            y=data_plot_config["tsne"][blogs_idx, 1],
            mode="markers",
            marker=dict(
                size=plotly_kwargs.get("marker_size", 16),
                color=data_plot_config["clusters"][blogs_idx],
                colorscale="Viridis",  # one of plotly colorscales
                showscale=True,
            ),
            text=[" ".join(words) for words in data_plot_config["words"]],
        )
    )
    if "marker_size" in plotly_kwargs:
        del plotly_kwargs["marker_size"]
    fig.update_layout(**plotly_kwargs)
    return fig
