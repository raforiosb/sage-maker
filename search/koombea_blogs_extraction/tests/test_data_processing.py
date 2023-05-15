import pandas as pd
from koombea_blogs.config.settings import settings
from koombea_blogs.data_module.data_processing import DataProcessing, DataInformation


def test_data_processing_process_data():
    blogs = pd.read_csv(f"test_data/{settings.DB_NAME}_blogs.csv")
    extraction_process = DataProcessing(blogs)
    extraction_process.preprocess_data(test=True)
    data = extraction_process.data
    spanish_data = extraction_process.spanish_data
    english_data = extraction_process.english_data

    assert len(data.columns) == len(
        DataInformation.final_columns
    ), "processed data does not have the same lenght of columns"
    assert (
        len(spanish_data) + len(english_data) == data.shape[0]
    ), "sum of lenght of english data and spanish data must be equal to total blogs"
    assert all(
        data.columns == DataInformation.final_columns
    ), "process data does not have the same columns names"
    assert (
        blogs.shape[0] == data.shape[0]
    ), "process data does not have the same number of blogs"

    if spanish_data:
        assert isinstance(spanish_data, list)
        assert isinstance(spanish_data[0], tuple)
        assert isinstance(spanish_data[0][0], int)
        assert isinstance(spanish_data[0][1], list)
        assert isinstance(spanish_data[0][1][0], str)
    if english_data:
        assert isinstance(english_data, list)
        assert isinstance(english_data[0], tuple)
        assert isinstance(english_data[0][0], int)
        assert isinstance(english_data[0][1], list)
        assert isinstance(english_data[0][1][0], str)
