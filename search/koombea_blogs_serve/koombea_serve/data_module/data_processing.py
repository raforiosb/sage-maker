import pandas as pd
from tqdm.std import tqdm
from ..config.logger import logger
from ..data_module.data_cleaning_utils import (
    clean_html,
    pandas_process_data_wrap,
    get_patterns_replacement,
)


class DataInformation:
    columns_needed = [
        "ID",
        "post_content",
        "post_excerpt",
        "post_name",
        "post_title",
        "industry_name",
        "lang",
    ]
    final_columns = columns_needed + ["data", "tokenized_data"]


class DataProcessing:
    def __init__(self, blogs: pd.DataFrame) -> None:
        self.data = blogs[DataInformation.columns_needed].copy()
        cols = []
        for col in self.data.columns:
            if self.data[col].isna().sum() != 0:
                logger.debug(f"column: {col} have na values")
                cols.append(col)
        self.cols_na = cols

    def preprocess_data(self, test: bool = False):
        if self.cols_na:
            logger.info("Cleaning na values")
            for col in self.cols_na:
                self.data[col].fillna("", inplace=True)
        logger.info("Cleaning html values")
        tqdm.pandas(desc="Cleaning html")
        self.data["post_content"] = self.data["post_content"].progress_apply(clean_html)
        logger.info("Join all data in one column")
        self.data["data"] = (
            self.data["post_content"]
            + " "
            + self.data["post_excerpt"]
            + " "
            + self.data["post_name"].apply(
                lambda post_name_val: " ".join(post_name_val.split("-"))
            )
            + " "
            + self.data["post_title"]
            + " "
            + self.data["industry_name"]
        )
        logger.info("tokenize data")
        patterns_replacement = get_patterns_replacement()
        lemmatize, remove_stops, normalize = True, True, True
        tqdm.pandas(desc="Tokenizing data")
        self.data["tokenized_data"] = self.data[["data", "lang"]].progress_apply(
            pandas_process_data_wrap,
            axis=1,
            args=(lemmatize, remove_stops, patterns_replacement, normalize),
        )
        # build metadata information for data dataset
        self.english_data = [
            (row.ID, row.tokenized_data)
            for row in self.data[self.data["lang"] == "en"].itertuples()
        ]
        self.spanish_data = [
            (row.ID, row.tokenized_data)
            for row in self.data[self.data["lang"] == "es"].itertuples()
        ]
