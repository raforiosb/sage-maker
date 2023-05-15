from typing import List, Tuple
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from nltk.probability import FreqDist
import numpy as np
import wordcloud as wd
import matplotlib.pyplot as plt
import wandb


def add_scatter_graph(fig, x, y, row, col, updated_xaxis=None, updated_yaxis=None):
    fig.add_trace(go.Scatter(x=x, y=y), row=row, col=col)
    updated_xaxis and fig.update_xaxes(**updated_xaxis, row=row, col=col)
    updated_yaxis and fig.update_yaxes(**updated_yaxis, row=row, col=col)


def add_bar_scatter_graph(
    fig, x, words, row, col, updated_xaxis=None, updated_yaxis=None
):
    y = np.arange(len(words))[::-1]
    fig.add_trace(go.Bar(x=x, y=y, orientation="h"), row=row, col=col)
    updated_yaxis and updated_yaxis.update(
        dict(tickmode="array", tickvals=y, ticktext=words)
    )
    add_scatter_graph(
        fig,
        x=x,
        y=y,
        row=row,
        col=col,
        updated_xaxis=updated_xaxis,
        updated_yaxis=updated_yaxis,
    )


def plot_tfidf_mean_words_info(
    vocab: List[str],
    mean_weights: np.ndarray,
    n_top: int,
    title: str,
    sup_titles: Tuple[str, str],
    **kwargs,
):
    fig = make_subplots(rows=2, cols=1, subplot_titles=sup_titles)
    vocab, mean_weights = zip(
        *sorted(zip(vocab, mean_weights), key=lambda x: x[1], reverse=True)
    )
    vocab_list = [vocab[:n_top], vocab[:-n_top:-1][::-1]]
    weights_list = [mean_weights[:n_top], mean_weights[:-n_top:-1][::-1]]
    for index, vocab_, weights_ in zip(
        range(len(sup_titles)), vocab_list, weights_list
    ):
        add_bar_scatter_graph(
            fig,
            x=weights_,
            words=vocab_,
            row=index + 1,
            col=1,
            updated_yaxis=dict(title=dict(text="word's samples")),
            updated_xaxis=dict(title=dict(text="tfidf weights")),
        )
    fig.update_layout(title_text=title, **kwargs)
    return fig


def plot_frequency_words_info(
    words_fdist: FreqDist, n_top: int, title: str, sup_titles: Tuple[str, str], **kwargs
):
    fig = make_subplots(rows=2, cols=1, subplot_titles=sup_titles)
    samples_list = [
        [item for item, _ in words_fdist.most_common(n=n_top)],
        [item for item, _ in words_fdist.most_common()[:-n_top:-1]],
    ]
    freqs_list = [
        [words_fdist[sample] for sample in samples] for samples in samples_list
    ]
    ylabel = "Counts"
    for index, samples, freqs in zip(
        range(len(samples_list)), samples_list, freqs_list
    ):
        add_scatter_graph(
            fig,
            x=np.arange(len(samples)),
            y=freqs,
            row=index + 1,
            col=1,
            updated_xaxis=dict(
                tickmode="array",
                tickvals=np.arange(len(samples)),
                ticktext=[str(s) for s in samples],
                title=dict(text="word's samples"),
            ),
            updated_yaxis=dict(title=dict(text=ylabel)),
        )
    fig.update_layout(title_text=title, **kwargs)
    return fig


def make_subplot_config(nmf):
    even = nmf.n_components % 2 == 0
    n_rows = nmf.n_components // 2 if even else nmf.n_components // 2 + 1
    specs = []
    sup_titles = []
    for n_row in range(n_rows):
        spec = []
        sup_titles.append(f"topic {n_row * 2 + 1}")
        spec.append({})
        if even or n_row < (n_rows - 1):
            sup_titles.append(f"topic {n_row * 2 + 2}")
            spec.append({})
        else:
            spec[-1] = {"colspan": 2}
            spec.append(None)
        specs.append(spec)
    return dict(specs=specs, n_rows=n_rows, sup_titles=sup_titles, even=even)


def get_words_importance_by_topics(nmf, vocab, n_top=8):
    matrix_V = nmf.components_
    get_top = lambda row: [
        (vocab[i], row[i]) for i in np.argsort(row)[: -n_top - 1 : -1]
    ]
    return [get_top(row_V) for row_V in matrix_V]


def plot_tfidf_topic_words_info(nmf, vocab, title, n_top, **kwargs):
    config = make_subplot_config(nmf)
    fig = make_subplots(
        rows=config["n_rows"],
        cols=2,
        specs=config["specs"],
        subplot_titles=config["sup_titles"],
    )
    top_words_by_topic = get_words_importance_by_topics(nmf, vocab, n_top)
    for n_row in range(config["n_rows"]):
        # first plot
        words, stats = zip(*top_words_by_topic[n_row * 2])
        add_bar_scatter_graph(
            fig,
            x=stats,
            words=words,
            row=n_row + 1,
            col=1,
            updated_yaxis=dict(title=dict(text="word's samples")),
            updated_xaxis=dict(title=dict(text="tfidf weights")),
        )

        if config["even"] or n_row < (config["n_rows"] - 1):
            # second plot if needed
            words, stats = zip(*top_words_by_topic[n_row * 2 + 1])
            add_bar_scatter_graph(
                fig,
                x=stats,
                words=words,
                row=n_row + 1,
                col=2,
                updated_yaxis=dict(title=dict(text="word's samples")),
                updated_xaxis=dict(title=dict(text="tfidf weights")),
            )
    fig.update_layout(title_text=title, **kwargs)
    return fig


def generate_wordcloud(ax, word2freq, max_words, title):
    wordcloud = wd.WordCloud(
        width=1200,
        height=500,
        background_color="white",
        max_words=max_words,
        max_font_size=500,
        normalize_plurals=False,
    )
    wordcloud.generate_from_frequencies(word2freq)
    ax.set_title(title)
    ax.imshow(wordcloud)
    ax.axis("off")


def plot_wordcloud_tfidif_info(nmf, vocab, n_top, title, **kwargs):
    config = make_subplot_config(nmf)
    top_words_by_topic = get_words_importance_by_topics(nmf, vocab, n_top)
    fig, axes_matrix = plt.subplots(
        nrows=config["n_rows"],
        ncols=2,
        tight_layout=True,
        figsize=(35, 35),
        squeeze=False,
        **kwargs,
    )
    for n_row, ax in zip(range(config["n_rows"]), axes_matrix):
        # first plot
        word2freq = dict(top_words_by_topic[n_row * 2])
        generate_wordcloud(
            ax[0],
            word2freq,
            max_words=n_top,
            title=f"topic {config['sup_titles'][n_row * 2]}",
        )
        if config["even"] or n_row < (config["n_rows"] - 1):
            word2freq = dict(top_words_by_topic[n_row * 2 + 1])
            generate_wordcloud(
                ax[1],
                word2freq,
                max_words=n_top,
                title=f"topic {config['sup_titles'][n_row * 2 + 1]}",
            )
    fig.savefig("/tmp/plot.png")
    return wandb.Image("/tmp/plot.png", caption=title)
