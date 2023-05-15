import re, calendar, datetime, time
import spacy
import json
from tqdm import tqdm
from app.utils import logger
from stop_words import get_stop_words
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

stop_words = list(get_stop_words('en')) + list(get_stop_words('es')) + ['app']
nltk_words = list(stopwords.words('english')) + list(stopwords.words('spanish')) + ['app']
stop_words.extend(nltk_words)

NLP_EN = spacy.load("en_core_web_sm")
NLP_ES = spacy.load("es_core_news_sm")

new_stops_words = ["y", "a", "en"]

for word in new_stops_words:
    lexeme = NLP_ES.vocab[word]
    lexeme.is_stop = True

def clean_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text()

def remove_unneccesary_whitespace(text):
    return ' '.join(text.split())

def basic_normalize(text, patterns_dict):
    text = text.lower()
    for pattern_re, replaced_str in patterns_dict:
        text = pattern_re.sub(replaced_str, text)
    return text

def get_regex_expression():
    # Basic normalization
    _patterns_ = [r'\'',
             r'\"',
             r'\.',
             r'<br \/>',
             r',',
             r'\(',
             r'\)',
             r'\!',
             r'\?',
             r'\;',
             r'\:',
             r'\s+']

    _replacements_ = [' \'  ',
                     '',
                     ' . ',
                     ' ',
                     ' , ',
                     ' ( ',
                     ' ) ',
                     ' ! ',
                     ' ? ',
                     ' ',
                     ' ',
                     ' ']

    # Match non alphanumeric characters
    NON_ALPHANUMERIC_REGEX = r'[^a-zA-Z0-9À-ÿ\u00f1\u00d1\s]'
    # Match any link or url from text
    LINKS_REGEX = r'https?:\/\/.*[\r\n]'
    # Match hashtags
    HASHTAGS_REGEX = r'\#[^\s]*'
    # Match twitter accounts
    TWITTER_ACCOUNTS_REGEX = r'\@[^\s]*'
    # Match Author:
    AUTHOR_REGEX = r'author'
    # Match email
    EMAIL_REGEX = r"\S*@\S+"
    # Group regex
    _patterns = [LINKS_REGEX,
                 HASHTAGS_REGEX,
                 TWITTER_ACCOUNTS_REGEX,
                 AUTHOR_REGEX,
                 EMAIL_REGEX,
                 NON_ALPHANUMERIC_REGEX]
    _replacements = [' ']*(len(_patterns))
    _patterns += _patterns_
    _replacements += _replacements_
    _patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))
    return _patterns_dict

# Get patterns_dict
patterns_dict = get_regex_expression()

def preprocess_data(text, lang, removing_stops=False, lemmatize=False):
    # Clean text
    text = remove_unneccesary_whitespace(basic_normalize(text, patterns_dict))
    # Choose the right tokenizer
    NLP = NLP_EN if lang == 'en' else NLP_ES
    # Tokenize the text of the blogs
    tokens = NLP(text)
    # Remove all punctuation marks
    tokens = [token for token in tokens if not token.is_punct]
    # Remove numbers or amount representation
    tokens = [token for token in tokens if not token.like_num]
    if removing_stops:
        # Remove stopswords
        tokens = [token for token in tokens if not token.is_stop]
        tokens = [token for token in tokens if token.text not in stop_words]
    if lemmatize:
        # Lemmatize words
        tokens = [token.lemma_.strip() for token in tokens]
    else:
        # Convert to str and lowerize
        tokens = [token.text.strip() for token in tokens]
    return tokens

def report_to_list(report, start_date, end_date):
    column_header = report.get('columnHeader', {})
    dimension_headers = column_header.get('dimensions', [])
    metric_headers = column_header.get('metricHeader', {}).get('metricHeaderEntries', [])
    report_list = []
    for row in report.get('data', {}).get('rows', []):
        data_temp = {}
        dimensions = row.get('dimensions', [])
        date_range_values = row.get('metrics', [])
        
        for header, dimension in zip(dimension_headers, dimensions):
            data_temp[header] = dimension
        for i, values in enumerate(date_range_values):
            for metric, value in zip(metric_headers, values.get('values')):
                if ',' in value or '.' in value:
                    data_temp[metric.get('name')] = float(value)
                else:
                    data_temp[metric.get('name')] = int(value)
        data_temp['startDate'] = start_date
        data_temp['endDate'] = end_date
        report_list.append(data_temp)
    return report_list

def get_week_data(analytics, VIEW_ID, 
               metrics, dimensions):
    """Get week data to make a tranding week list of blogs
    Args:
        analytics: An authorized Analytics Reporting API V4 service object.
    """
    today = datetime.datetime.now()
    dates_week = [today + datetime.timedelta(days=i) 
                  for i in range(0 - today.weekday(), 7 - today.weekday())]
    data_list = []
    initial_bar = tqdm(dates_week, 
                   total = len(dates_week), 
                   desc="Getting reports within the actual week")
    for date_week in initial_bar:
        start_date = "{:%Y-%m-%d}".format(date_week)
        end_date = start_date
        while True:
            # time.sleep(10)
            response = get_data(analytics, start_date, end_date, VIEW_ID, 
               metrics, dimensions)
            if type(response) != str:
                data_list += response
                break;
            else:
                index_day -= 1
                end_date = "{:%Y-%m-%d}".format(datetime.datetime(year, month, index_day))
        initial_bar.set_postfix(day=date_week.day, lenght=len(data_list))
    return data_list

def get_month_data(analytics, year, month, VIEW_ID, 
               metrics, dimensions):
    """ Get month data for a given year and month number
    
    Args:
        analytics: An authorized Analytics Reporting API V4 service object.
        year (int): year int number
        month (int): month int number
    """
    # analytics = initialize_analyticsreporting_api()
    last_day_month = calendar.monthrange(year, month)[1] # Get last day of the month
    data_list = []
    index_day = 1
    while index_day < last_day_month:
        start_date = "{:%Y-%m-%d}".format(datetime.datetime(year, month, index_day))
        # index_day += 3
        # if (index_day > last_day_month):
            # index_day = last_day_month
        # end_date = "{:%Y-%m-%d}".format(datetime.datetime(year, month, index_day))   
        end_date = start_date
        while True:
            response = get_data(analytics, start_date, end_date, VIEW_ID, 
               metrics, dimensions)
            if type(response) != str:
                data_list += response
                break;
            else:
                index_day -= 1
                end_date = "{:%Y-%m-%d}".format(datetime.datetime(year, month, index_day))
            time.sleep(100)
        index_day += 1
    return data_list

def get_data(analytics, start_date, end_date, VIEW_ID, 
               metrics, dimensions):
    """ Get data given the start_date and end_date
    
    Args:
        analytics: An authorized Analytics Reporting API V4 service object.
        start_date (str): str start_date in the format %Y-%m-%d
        end_date (str): str end_date in the format %Y-%m-%d
    """
    report_temp = get_report(analytics, [{'startDate': start_date,
                                    'endDate': end_date}], VIEW_ID, 
                               metrics, dimensions)
    report = report_temp.get('reports', [])[0]
    report_data = report.get('data', {})
    #if report_data.get('samplesReadCounts', []) or report_data.get('samplingSpaceSizes', []):
    #    return 'Sampled Data'
    #if report_data.get('rowCount') > 900000:
    #    return 'Exceeded Row Count'
    next_page_token = report.get('nextPageToken')
    if next_page_token:
        print("entro")
        raise("STOP")
        # Iterating through pages
        pass
    report = report_to_list(report, start_date, end_date)
    return report

def get_report(analytics, date, VIEW_ID, 
               metrics, dimensions, page_token = None):
    """Queries the Analytics Reporting API V4.

    Args:
        analytics: An authorized Analytics Reporting API V4 service object.
        date: date ranges body argument request
        page_token: page_token just in case
    Returns:
        The Analytics Reporting API V4 response.
    """
    if page_token is None:
        reports = analytics.reports().batchGet(
                    body={
                        'reportRequests': [
                        {
                            'viewId': VIEW_ID,
                            'dateRanges': date,
                            'metrics': metrics,
                            'dimensions': dimensions,
                            #'samplingLevel': 'LARGE',
                            #'pageSize' : 100000
                       }]
                    }
                ).execute()
    else:
        reports = analytics.reports().batchGet(
            body={
                        'reportRequests': [
                        {
                            'viewId': VIEW_ID,
                            'dateRanges': date,
                            'metrics': metrics,
                            'dimensions': dimensions,
                            'samplingLevel': 'LARGE',
                            'pageToken': page_token,
                            'pageSize' : 100000
                       }]
                    }
        ).execute()
    return reports