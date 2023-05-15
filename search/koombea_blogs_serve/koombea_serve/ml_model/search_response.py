import numpy as np

import numpy as np


class SearchResponse:
    def __init__(self, indices, rank, blogs_df, lang: str):
        """Search response constructor
        Args:
            indices (np.array): array with indices in order
            rank (np.array): array with similarity ranking
            blogs_df (Dataframe): pandas dataframe with data
        """
        self.results = list(zip(indices, rank))
        self.lang = lang
        # Handling dataframe
        self.blogs_df = blogs_df.copy()
        self.blogs_df["index"] = self.blogs_df["id"].copy()
        self.blogs_df.set_index("index", drop=True, inplace=True)

        # Transformations
        self.industries = self.blogs_df.industry_term.unique().tolist()
        self.content_types = self.blogs_df.content_type_term.unique().tolist()

        # Features
        self.features = [
            "id",
            "slug",
            "link",
            "title",
            "post_modified",
            "post_date",
            "author",
            "industry",
            "content_type",
            "image_alt",
            "image",
        ]

    def handling_params(self, content_type, term):
        """handlig params, it converts the incoming
        param to the one in our dataset
        Args:
            content_type (list or str): content_type list value
            term (list or str): term list value
        Returns:
            content_type (list): content_type value on df
            ter (list): term value on df
        """
        if isinstance(content_type, str):
            content_type = [content_type]

        if isinstance(term, str):
            term = [term]

        if content_type[0] != "":
            content_type = [
                content_type_
                for content_type_ in content_type
                if content_type_ in self.content_types
            ]
        else:
            content_type = None

        if term[0] != "":
            term = [term_ for term_ in term if term_ in self.industries]
        else:
            term = None

        return content_type, term

    def handling_per_page(self, per_page):
        """handling per page to special cases
        Args:
            per_page (int): per_page response
        Returns:
            per_page (int): handled per_page response
        """
        # Handling per_page
        if per_page >= 20:
            if per_page >= len(self.results):
                per_page = len(self.results)
            else:
                per_page = 20
        elif per_page <= 0:
            per_page = 1

        return per_page

    def handling_results(self, content_type, term, lang):
        """handling results list with ranked results
        Args:
            content_type (str): content_type str to response
            term (str): term str to response
            lang (str): languange str
        """
        # cast results to the content_type or term
        index = None

        if content_type is not None and term is not None:
            query = (
                (self.blogs_df["industry_term"].isin(term))
                & (self.blogs_df["content_type_term"].isin(content_type))
                & (self.blogs_df["lang"] == lang)
            )
            index = self.blogs_df[query].index.tolist()
        elif content_type is not None and term is None:
            query = (self.blogs_df["content_type_term"].isin(content_type)) & (
                self.blogs_df["lang"] == lang
            )
            index = self.blogs_df[query].index.tolist()
        elif content_type is None and term is not None:
            query = (self.blogs_df["industry_term"].isin(term)) & (
                self.blogs_df["lang"] == lang
            )
            index = self.blogs_df[query].index.tolist()
        elif content_type is None and term is None:
            query = self.blogs_df["lang"] == lang
            index = self.blogs_df[query].index.tolist()

        if index is not None:
            minimum_base = self.results[0][1]
            index = [self.id_to_index[ids] for ids in index]
            self.results = [
                result_
                for result_ in self.results
                if (result_[0] in index and result_[1] < (minimum_base + 0.20))
            ]

    def handling_page(self, page, max_num_pages):
        """handling page
        Args:
            page (int): page number current_apage
            max_num_pages (int): max number of pages on response
        Returns:
            page (int): accept value of page
        """
        # Handling page
        if page - 1 < 0:
            page = 1
        elif page > max_num_pages:
            page = max_num_pages

        return page

    def get_response(self, per_page=1, page=1, content_type=[""], term=[""]):
        """[summary]
        Args:
            per_page (int, optional): per_page value of pagination. Defaults to 1.
            page (int, optional): current_page value of patination. Defaults to 1.
            content_type (list, optional): content_type list of blogs. Defaults to None.
            term (list, optional): term list blogs. Defaults to None.
            lang (str, optional): language unicode. Defatuls to en (English filter)
        Returns:
            responde (dict): dictionary response with the response format:
                    {
                        "paging": {
                            "total_count": 462,
                            "total_pages": 231,
                            "current_page": 2,
                            "per_page": 2
                        },
                        "posts": [
                            {
                                ...
                            },
                            {
                                ...
                            }
                        ]
                    }
        """
        lang = self.lang
        content_type, term = self.handling_params(content_type, term)

        self.index_to_id = dict(enumerate(self.blogs_df.index))
        self.id_to_index = dict((ids, index) for index, ids in self.index_to_id.items())
        # Handling results and ranking
        self.handling_results(content_type, term, lang)
        per_page = self.handling_per_page(per_page)
        response = {}
        len_results = len(self.results)
        if len_results % per_page == 0:
            max_num_pages = len_results // per_page
        else:
            max_num_pages = len_results // per_page + 1

        page = self.handling_page(page, max_num_pages)

        paginate_results_ = self.paginate_results(per_page, num_page=page - 1)
        response["paging"] = {}
        response["paging"]["total_count"] = len_results
        response["paging"]["total_pages"] = max_num_pages
        response["paging"]["current_page"] = page
        response["paging"]["per_page"] = per_page

        paginate_results_ = [self.index_to_id[index] for index in paginate_results_]
        response["posts"] = [
            dict(val[self.features])
            for _, val in self.blogs_df.loc[paginate_results_].iterrows()
        ]
        return response

    def paginate_results(self, per_page, num_page):
        """paginate results generator
        Args:
            per_page (int): indicate the per_page value
            num_page (int): indicate the current_page
        Yields:
            value (int): index response from the df
        """
        pagination_values = self.results[
            num_page * per_page : (num_page + 1) * per_page
        ]
        for value in pagination_values:
            yield int(value[0])
