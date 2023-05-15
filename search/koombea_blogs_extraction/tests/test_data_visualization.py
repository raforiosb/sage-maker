from pytest import MonkeyPatch
from koombea_blogs.data_analysis import data_visualizing, visualize_utils
from wandb.wandb_run import Run
from collections import namedtuple


def test_make_frequency_analysis(wandb_run: Run, monkeypatch: MonkeyPatch):
    def mock_get_wandb_run() -> Run:
        return wandb_run

    monkeypatch.setattr(
        data_visualizing, "get_wandb_run_visualizing_analysis", mock_get_wandb_run
    )
    visualizing_process = data_visualizing.DataVisualizing(
        english_data=[["hello", "hello", "world"], ["hello", "hello", "world"]],
        spanish_data=[["hola", "hola", "mundo"], ["hola", "hola", "mundo"]],
        num_topic_es=2,
        num_topic_en=2,
    )
    assert visualizing_process.wandb_run.id == "test_id", "wandb id is not the mock id"
    visualizing_process.make_frequency_analysis(
        title="Frequency distribution for english data",
        sup_titles=[
            "Most repeated english words in blogs",
            "Less repeated english words in blogs",
        ],
        lang="en",
        n_top=3,
    )
    visualizing_process.make_frequency_analysis(
        title="Frequency distribution for spanish data",
        sup_titles=[
            "Most repeated spanish words in blogs",
            "Less repeated spanish words in blogs",
        ],
        lang="es",
        n_top=3,
    )


def test_make_tfidf_mean_analysis(wandb_run: Run, monkeypatch: MonkeyPatch):
    def mock_get_wandb_run() -> None:
        return wandb_run

    monkeypatch.setattr(
        data_visualizing, "get_wandb_run_visualizing_analysis", mock_get_wandb_run
    )
    visualizing_process = data_visualizing.DataVisualizing(
        english_data=[["hello", "hello", "world"], ["hello", "hello", "world"]],
        spanish_data=[["hola", "hola", "mundo"], ["hola", "hola", "mundo"]],
        num_topic_en=2,
        num_topic_es=2,
    )
    assert hasattr(
        visualizing_process, "tfidf_en"
    ), "visualizing process did not create tfidf object for english"
    assert hasattr(
        visualizing_process, "tfidf_es"
    ), "visualizing process did not create tfidf object for spanish"
    assert hasattr(
        visualizing_process, "matrix_en"
    ), "visualizing process did not create numpy matrix en for english"
    assert hasattr(
        visualizing_process, "matrix_es"
    ), "visualizing process did not create numpy matrix es for spanish"

    visualizing_process.make_tfidf_mean_analysis(
        title="English tfidf mean weights across blogs",
        sup_titles=[
            "Highest tfidf weights english words in blogs",
            "Lowest tfidf weights english words in blogs",
        ],
        lang="en",
        n_top=3,
    )
    visualizing_process.make_tfidf_mean_analysis(
        title="Spanish tfidf mean weights across blogs",
        sup_titles=[
            "Highest tfidf weights spanish words in blogs",
            "Lowest tfidf weights spanish words in blogs",
        ],
        lang="es",
        n_top=3,
    )


def test_make_tfidf_by_topic_analysis(wandb_run: Run, monkeypatch: MonkeyPatch):
    def mock_get_wandb_run() -> None:
        return wandb_run

    monkeypatch.setattr(
        data_visualizing, "get_wandb_run_visualizing_analysis", mock_get_wandb_run
    )
    visualizing_process = data_visualizing.DataVisualizing(
        english_data=[["hello", "hello", "world"], ["hello", "hello", "world"]],
        spanish_data=[["hola", "hola", "mundo"], ["hola", "hola", "mundo"]],
        num_topic_en=2,
        num_topic_es=2,
    )
    assert hasattr(
        visualizing_process, "nmf_en"
    ), "visualizing process did not create nmf object for english"
    assert hasattr(
        visualizing_process, "nmf_es"
    ), "visualizing process did not create nmf object for spanish"
    try:
        visualizing_process.make_tfidf_by_topics_analysis(
            title="English tfidf topic weights analysis",
            lang="en",
            n_top=3,
        )
        visualizing_process.make_tfidf_by_topics_analysis(
            title="Spanish tfidf topic weights analysis",
            lang="es",
            n_top=3,
        )
        assert True
    except Exception as error:
        assert False, "Error making analysis {}".format(error)


def test_make_subplot_config():
    NMF = namedtuple("NMF", ["n_components"])
    nmf = NMF(n_components=4)
    config = visualize_utils.make_subplot_config(nmf)
    assert config == {
        "specs": [[{}, {}], [{}, {}]],
        "n_rows": 2,
        "sup_titles": ["topic 1", "topic 2", "topic 3", "topic 4"],
        "even": True,
    }, "subplot configuration bad config for even number of rows"
    nmf = NMF(n_components=3)
    config = visualize_utils.make_subplot_config(nmf)
    assert config == {
        "specs": [[{}, {}], [{"colspan": 2}, None]],
        "n_rows": 2,
        "sup_titles": ["topic 1", "topic 2", "topic 3"],
        "even": False,
    }, "subplot configuration bad config for odd number of rows"


def test_make_tfidf_wordcloud_analysis(wandb_run: Run, monkeypatch: MonkeyPatch):
    def mock_get_wandb_run() -> None:
        return wandb_run

    monkeypatch.setattr(
        data_visualizing, "get_wandb_run_visualizing_analysis", mock_get_wandb_run
    )
    visualizing_process = data_visualizing.DataVisualizing(
        english_data=[
            ["hello", "hello", "world"] * 100,
            ["hello", "hello", "world"] * 100,
        ],
        spanish_data=[["hola", "hola", "mundo"] * 100, ["hola", "hola", "mundo"] * 100],
        num_topic_en=1,
        num_topic_es=1,
    )
    try:
        visualizing_process.make_tfidf_wordcloud_by_topics_analysis(
            title="Wordcloud for english topic analysis", lang="en", n_top=3
        )
        visualizing_process.make_tfidf_wordcloud_by_topics_analysis(
            title="Wordcloud for spanish topic analysis", lang="es", n_top=3
        )
        assert True
    except Exception as error:
        assert False, "There was an error making wordcloud analysis {}".format(error)
