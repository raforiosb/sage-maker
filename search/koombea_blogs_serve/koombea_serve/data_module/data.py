from ..data_module.data_cleaning_utils import get_patterns_replacement, process_data
from ..ml_model.model import Model

patterns_replacement = get_patterns_replacement()


class Data:
    def __init__(self, query: str, lang="en") -> None:
        self.lang = lang
        self.query = query

    def get_response(self):
        self.query = process_data(
            self.query,
            self.lang,
            lemmatize=True,
            remove_stops=True,
            patterns_replacement=patterns_replacement,
            normalize=True,
        )
        return Model.get_results(self.query, self.lang)
