from typing import List, Tuple
import bs4
import re
import many_stop_words
import spacy

stopwords = many_stop_words.get_stop_words("en").union(
    many_stop_words.get_stop_words("es")
)
nlp_es = spacy.load("es_core_news_sm")  # load spanish tokenizer
nlp_en = spacy.load("en_core_web_sm")  # load english tokenizer


def clean_html(html_text: str) -> str:
    """Parse HTMl text and return clean text

    Args:
        html_text (str): string with html tags

    Returns:
        str: string without html tags
    """
    # Insert a break line at the end of all tags
    html_text = ">\n".join(html_text.split(">"))
    soup = bs4.BeautifulSoup(html_text, "html.parser")
    return soup.get_text()


def get_patterns_replacement() -> List[Tuple[re.Pattern, str]]:
    """Basic normalization patterns

    Returns:
        List[Tuple[re.Pattern, str]]: list of tuples that map
            a regex pattern and the replacement
    """
    # basic normalization
    _patterns_ = [
        r"\'",
        r"\"",
        r"\.",
        r"<br \/>",
        r",",
        r"\(",
        r"\)",
        r"\!",
        r"\?",
        r"\;",
        r"\:",
        r"\s+",
    ]

    _replacements_ = [
        " '  ",
        " ",
        " . ",
        " ",
        " , ",
        " ( ",
        " ) ",
        " ! ",
        " ? ",
        " ",
        " ",
        " ",
    ]
    # Match non alphanumeric characters
    NON_ALPHANUMERIC_REGEX = r"[^a-zA-Z0-9À-ÿ\u00f1\u00d1\s]"
    # Match numerical characters
    NUMERICAL_REGEX = r"[0-9]+"
    # Match any link or url from text
    LINKS_REGEX = r"https?:\/\/.*[\r\n]"
    # Match hashtags
    HASHTAGS_REGEX = r"\#[^\s]*"
    # Match twitter accounts
    TWITTER_ACCOUNTS_REGEX = r"\@[^\s]*"
    # Match Author:
    AUTHOR_REGEX = r"author"
    # Match email
    EMAIL_REGEX = r"\S*@\S+"
    # Group regex
    patterns = [
        LINKS_REGEX,
        HASHTAGS_REGEX,
        TWITTER_ACCOUNTS_REGEX,
        AUTHOR_REGEX,
        EMAIL_REGEX,
        NON_ALPHANUMERIC_REGEX,
        NUMERICAL_REGEX,
    ]
    replacements = [" "] * (len(patterns))
    patterns += _patterns_
    replacements += _replacements_
    patterns_replacement = list(
        (re.compile(p), r) for p, r in zip(patterns, replacements)
    )
    return patterns_replacement


def remove_unnecesary_whitespaces(text: str) -> str:
    return " ".join(text.strip().split())


def apply_basic_normalization(
    text: str, patterns_replacement: List[Tuple[re.Pattern, str]]
) -> str:
    text = text.lower()
    for pattern_re, replace_str in patterns_replacement:
        text = pattern_re.sub(replace_str, text)
    return text


def tokenize(
    text: str, lang: str, lemmatize: bool = True, remove_stops: bool = True
) -> List[str]:
    """Receive a basic normalize text to tokenize, we can apply lemmatize or remove stops

    Args:
        text (str): Basic normalize text
        lang (str): language, can be either `en` or `es`
        lemmatize (bool, optional): Wether to lemmatize or not. Defaults to True.
        remove_stops (bool, optional): Wether to remove stops or not. Defaults to True.

    Returns:
        List[str]: list of tokens
    """
    # choose the right tokenizer
    nlp = nlp_en if lang == "en" else nlp_es
    # tokenize
    tokens = nlp(text)
    # check for number and punctuation
    tokens = [token for token in tokens if not (token.is_punct or token.like_num)]
    if remove_stops:
        tokens = [
            token for token in tokens if not (token.is_stop or token.text in stopwords)
        ]
    return (
        [token.lemma_.strip() for token in tokens]
        if lemmatize
        else [token.text.strip() for token in tokens]
    )


def process_data(
    text: str,
    lang: str,
    lemmatize: bool,
    remove_stops: bool,
    patterns_replacement: List[Tuple[re.Pattern, str]],
    normalize=True,
) -> List[str]:
    """Process all the raw data

    Args:
        text (str): raw text data (cleaned up of html tags)
        lang (str): languange, can be either 'en' or 'es'
        lemmatize (bool): Wether to lemmatize or not.
        remove_stops (bool):  Wether to remove stops or not.
        patterns_replacement (List[Tuple[re.Pattern, str]]): mapping the regular
            pattern to replacement string.
        normalize (bool, optional): ether to apply simple text normalize or not. Defaults to True.

    Returns:
        List[str]: list of tokens
    """
    text = (
        remove_unnecesary_whitespaces(
            apply_basic_normalization(text, patterns_replacement)
        )
        if normalize
        else text
    )
    return tokenize(text, lang, lemmatize, remove_stops)


def pandas_process_data_wrap(
    raw_data,
    lemmatize: bool,
    remove_stops: bool,
    patterns_replacement: List[Tuple[re.Pattern, str]],
    normalize: bool,
) -> List[str]:
    """pandas wrap function to tokenize data using apply function

    Args:
        raw_data ([type]): row_data from pandas df, have the data and the lang attributes
        lemmatize (bool): Wether to lemmatize or not.
        remove_stops (bool): Wether to remove stops or not.
        patterns_replacement (List[Tuple[re.Pattern, str]]): mapping the regular
            pattern to replacement string.
        normalize (bool): wether to normalize or not

    Returns:
        List[str]: List of tokens
    """
    return process_data(
        raw_data.data,
        raw_data.lang,
        lemmatize=lemmatize,
        remove_stops=remove_stops,
        patterns_replacement=patterns_replacement,
        normalize=normalize,
    )
