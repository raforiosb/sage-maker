from koombea_blogs.data_module.data_extraction import DataExtraction, BlogsInformation
from sqlalchemy.engine import Connection
from koombea_blogs.config.settings import settings
import pandas as pd


def test_data_extraction_extract(conn: Connection):
    extraction_process = DataExtraction(conn)
    try:
        extraction_process.extract(test=True)
        blogs = extraction_process.blogs
        assert True
    except Exception as error:
        assert False, f"There was an error {error}"
    # read
    test_blogs = pd.read_csv(f"test_data/{settings.DB_NAME}_blogs.csv")
    # test columns
    assert len(blogs.columns) == len(
        test_blogs.columns
    ), "extraction columns length blogs are not equal to test blogs df"
    assert all(
        blogs.columns == test_blogs.columns
    ), "columns blogst are not equal to test blogs df"
    assert all(
        blogs.columns == BlogsInformation.final_blogs_columns
    ), "columns blogst are not equal to test blogs df"
    assert all(blogs == test_blogs), "the df infos are not the same"
