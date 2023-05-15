import json, os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import datetime
from app.utils import logger
from app.ml_model import ScoringModel
from app.dataset.data_utils import (clean_html, preprocess_data, 
                                    get_report, get_month_data, get_week_data)
from app.sql_app import crud, models
from langdetect import detect, DetectorFactory
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

DetectorFactory.seed = 0
tqdm.pandas(desc="Preprocessing Data")

config_path = "/opt/ml/input/config"
        
class Data(object):
    """Class Data To use for data preprocessing, and request in case 
    there are no cache yet"""
    
    def __init__(self, slug, content='', lang='en', special_data=False, clean_first=False, 
                  **kwargs):
        """Constructor for data object"""
        
        if clean_first:
            content = clean_html(content)
            
        slug = slug.replace('-', ' ')
        
        if special_data:
            input_data = (" " + slug) * 4 + " " + content
            print(input_data)
        else:
            input_data = slug + " " + content
            
        
        self.data = tokenize(input_data, 
                                lang=lang,
                                **kwargs)
        
        if special_data:
            print(self.data)
            
        self.lang = lang
        
    def get_recommendations(self, top_k):
        return ScoringModel.predict_new(self.data, self.lang, top_k)
    
    def get_recommendations_special_data(self, top_k, key_value, slug, slug_blocked=None):
        return ScoringModel.predict_special_data(self.data, top_k, key_value, slug, slug_blocked=slug_blocked)
    
    @classmethod
    def get_recommendations_popular_data(cls, top_k, type_analysis):
        return ScoringModel.predict_popular_blogs(top_k, type_analysis)


class Dataset(object):
    """Class Dataset to use for data preprocessing and clean, also to generate the new 
    database"""

#     def __init__(self, data, index_slug, redis_db):
#         """Constructor for dataset object"""
#         self.data = data
#         self.index_slug = index_slug
#         self.redis_db = redis_db
        
    def __init__(self, data, redis_db, additional_data):
        self.data = data
        self.data["post_date"] = pd.to_datetime(self.data.post_date_str, infer_datetime_format=True)
        self.redis_db = redis_db 
        self.additional_data = dict(additional_data)
    
#     @classmethod
#     def create_dataset(cls, redis_db, db):
#         logger.info("Using new version without pandas")
#         # Retrieve all the data 
#         blogs = crud.get_publish_blogs(db)
#         data_zipped = [(blog.id, blog.slug, blog.content) for blog in blogs]
#         # Sorted By id descence order
#         data_zipped = sorted(data_zipped, key = lambda x: x[0])
#         # Unzipped
#         data_unzipped = list(zip(*data_zipped))
#         # Select contents and slugs
#         contents = data_unzipped[2]
#         slugs = data_unzipped[1]
#         # Clean HTML of contents
#         contents = list(map(clean_html, tqdm(contents, desc="HTML parsing")))
#         train_data = list(map(lambda x, y: x.replace("-", " ") + " " + y, slugs, contents))
#         languages_data = list(map(detect_language, tqdm(train_data, desc="Getting Languages")))
#         # Zipp
#         all_data = zip(train_data, languages_data, slugs)
#         data_en = [_data for _data in all_data if _data[1] == 'en']
#         all_data = zip(train_data, languages_data, slugs)
#         data_es = [_data for _data in all_data if _data[1] == 'es']
#         logger.info(f"Num data with 'en' : {len(data_en)}")
#         logger.info(f"Num data with 'es' : {len(data_es)}")
#         # Preprocessing data
#         clean_data_en = list(map(lambda x: tokenize(x[0], 'en', True, True), tqdm(data_en, desc="Tokenize")))
#         clean_data_es = list(map(lambda x: tokenize(x[0], 'es', True, True), tqdm(data_es, desc="Tokenize")))
#         # Get Dictionary
#         index_slug_en = [_data[2] for index, _data in enumerate(data_en)]
#         index_slug_es = [_data[2] for index, _data in enumerate(data_es)]
#         # print(index_slug_es)
#         # Cache Dictionary
#         redis_db.set("index_slug_en", json.dumps(index_slug_en))
#         redis_db.set("index_slug_es", json.dumps(index_slug_es))
#         # Create en and es dataset
#         dataset_en = cls(clean_data_en, index_slug_en, redis_db)
#         dataset_es = cls(clean_data_es, index_slug_es, redis_db)
#         return dataset_en, dataset_es
    
    def predict(self, lang):
        # Get Vectors
        self.get_vectors(lang)
        self.get_index_model()
        self.cache_recommendations()
        if lang == "en":
            self.cache_recoommendations_for_additional_data()
        
    def get_vectors(self, lang):
        vectors = np.zeros(shape = (len(self.data), 300), dtype=np.float32)
        for idx, _data in tqdm(enumerate(self.data.preprocess_data), total = len(self.data), desc="Generating Vectors"):
            vectors[idx] = ScoringModel.get_vectors_cache(_data, lang)
        self.vectors = vectors
        
    def get_index_model(self):
        self.index_model = ScoringModel.get_index_model_cache(self.vectors)
        
    def cache_recommendations(self):
        with self.redis_db.pipeline() as pipe:
            for idx, slug in tqdm(enumerate(self.data.slug), total=len(self.data),
                                 desc="Cache Recommendations"):
                vector = self.vectors[idx].reshape(1, -1)
                recommendations_vector = ScoringModel.predict_cache(vector, self.index_model)
                recommendations_slug = [self.data.slug.values[idx_] for idx_ in recommendations_vector]
                pipe.set(slug, json.dumps(recommendations_slug))
            pipe.execute()
    
    def cache_recoommendations_for_additional_data(self):
        with self.redis_db.pipeline() as pipe:
            for name, additional_inference in tqdm(self.additional_data.items(), total=len(self.additional_data),
                                            desc="Cache Recommendation for additional data"):
                for idx, data in additional_inference.iterrows():
                    slug = data.slug
                    blocked_slugs = self.redis_db.get('blocked_slug_for_'+slug)
                    if blocked_slugs:
                        blocked_slugs = json.loads(blocked_slugs)
                    input_data = (slug.replace("-", " ") + " ") * 4 + data.title # + " " + data.post_excerpt
                    input_data = tokenize(input_data, "en", True, True)
                    input_vector = ScoringModel.get_vectors_cache(input_data, lang="en")
                    results = ScoringModel.predict_cache_additional_data(self.data, input_vector, 
                                                                         self.index_model, name, slug)
                    # pipe.set(slug, json.dumps([(val[3], val[1]) for val in results]))
                    pipe.set(slug, json.dumps([val[3] for val in results]))
            pipe.execute()
                
            
    @classmethod
    def create_dataset_(cls, redis_db, db):
        # Get DataFrames tables
        blogs = transform_post_query_to_dataframe(crud.get_publish_blogs(db), 
                                                  ["id", "slug", "content", "post_date", "author_id"])
        services = transform_post_query_to_dataframe(crud.get_services_posts(db), ["title", "slug", "post_excerpt"])
        industries = transform_post_query_to_dataframe(crud.get_industries_posts(db), ["title", "slug", "post_excerpt"])
        logger.info("Num blogs {}".format(blogs.shape[0]))
        # Get date str
        blogs["post_date_str"] = blogs["post_date"].progress_apply(convert_date)
        blogs["post_date"] = blogs["post_date"].progress_apply(lambda x: str(x))
        blogs.reset_index(drop=True, inplace=True)
        # Get term taxonomy
        terms_taxonomy_id = crud.get_industry_id_taxonomy(db)
        tqdm.pandas(desc="Get term slug")
        blogs["term_slug"] = blogs["id"].progress_apply(lambda x: get_term_industry_for_blog(x, terms_taxonomy_id, db))
        # Get author blog slug name
        tqdm.pandas(desc="Get author blog slug name")
        blogs["user_nicename"] = blogs["author_id"].progress_apply(lambda x: get_blog_author_name_slug(x, db))
        logger.info("Complete blogs retrieving dataset!")
        # Clean html data
        tqdm.pandas(desc="Clean Html data")
        blogs["content"] = blogs["content"].progress_apply(lambda x: clean_html(x))
        # Getting Languages
        tqdm.pandas(desc="Getting Languages")
        blogs["data"] = blogs["slug"].apply(lambda x: " ".join(x.split("-"))) + " " + blogs["content"]
        blogs["lang"] = blogs["data"].progress_apply(detect_language)
        # Logs languages
        logger.info("Num Blogs with language 'es': " + str(blogs[blogs["lang"] == 'es'].shape[0]))
        logger.info("Num Blogs with language 'en': " + str(blogs[blogs["lang"] == 'en'].shape[0]))
        # Preprocessing Data
        preprocess_data = []
        for index, blog in tqdm(blogs.iterrows(), total=blogs.shape[0], desc="Preprocess data"):
            data = blog.data
            lang = blog.lang
            preprocess_data.append(tokenize(data, lang, True, True))
        blogs["preprocess_data"] = preprocess_data
        # Create
        data_en = blogs[blogs["lang"] == "en"].copy()
        data_es = blogs[blogs["lang"] == "es"].copy()
        data_en.reset_index(drop=True, inplace=True)
        data_es.reset_index(drop=True, inplace=True)
        # Cache
        redis_db.set("data_en", json.dumps(data_en.to_dict()))
        redis_db.set("data_es", json.dumps(data_es.to_dict()))
        
        additional_data = [("services",services), ("industries",industries)]
        return cls(data_en, redis_db, additional_data), cls(data_es, redis_db, additional_data)
        
class AnalyticDataset(object):
    """Class for google analytics dataset in order to get popular blogs"""
    
    SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
    google_key_file = os.path.join(config_path, 
                                       'analytics-key.json')
    VIEW_ID = '183468851'
    metrics = [{'expression': 'ga:pageviews'},
           {'expression': 'ga:uniquePageviews'},
           {'expression': 'ga:timeOnPage'},
           {'expression': 'ga:avgTimeOnPage'},
           #{'expression': 'ga:exits'},
           {'expression': 'ga:exitRate'},
           {'expression': 'ga:sessions'},
           {'expression': 'ga:visits'},
           {'expression': 'ga:bounces'},
           {'expression': 'ga:bounceRate'},
           {'expression': 'ga:sessionDuration'}]

    dimensions = [{'name': 'ga:pageTitle'},
                {'name': 'ga:pagePath'},
                {'name': 'ga:pageDepth'}]
    
    def __init__(self, data, redis_db):
        self.redis_db = redis_db
        self.data = data
    
    @classmethod
    def create_analytics_dataset(cls, redis_db, type_analysis="year", db=None):
        
        analytics_reporting_api = initialize_analyticsreporting_api(cls.google_key_file, 
                                                                    cls.SCOPES)
        if type_analysis == "year":
            total_reports = cls.get_total_reports_year(analytics_reporting_api)
            data = cls.get_popularity_dataframe(total_reports, type_analysis, db=db)
        elif type_analysis == "week":
            total_reports = get_week_data(analytics_reporting_api, cls.VIEW_ID, 
               cls.metrics, cls.dimensions)
            data = cls.get_popularity_dataframe(total_reports, type_analysis, db=db)
        return cls(data, redis_db)
    
    @classmethod
    def get_total_reports_year(cls, analytics_reporting_api):
        datetime_now = datetime.datetime.now()
        dates_ranges = [(datetime_now.year, month) for month in range(1, datetime_now.month + 1)]
        initial_bar = tqdm(dates_ranges, 
                   total = len(dates_ranges), 
                   desc="Getting reports within the dates ranges")
        total_reports = []
        for year, month in initial_bar:
            report = get_month_data(analytics_reporting_api, year, month, cls.VIEW_ID, 
               cls.metrics, cls.dimensions)
            total_reports += report
            initial_bar.set_postfix(year=year, month=month, lenght=len(total_reports))
            time.sleep(10)
        return total_reports
    

    @classmethod
    def get_popularity_dataframe(cls, total_reports, type_analysis, db=None):
        data = pd.DataFrame(total_reports)
        data.columns = [col.replace('ga:', '') for col in data.columns]

        index_blog = data[data['pagePath'].str.contains('blog')].index
        data = data.loc[index_blog].copy()
        data = data.loc[~data.pagePath.str.contains("__url_version__")].copy()
        data['datetime'] = pd.to_datetime(data['startDate'], infer_datetime_format=True)
        # Get blog_slug
        data['blog_slug'] = data['pagePath'].apply(get_slug_by_page_path)

        index_no_blog = data[data['blog_slug'] == ''].index

        data.drop(index = index_no_blog, inplace=True)
        data.reset_index(drop=True, inplace=True)
        blog_groups = data.groupby('blog_slug').groups
        
        # I need to sanity check the list of slugs in case of elimination from db
        slugs_error = sanity_check_slug(db, blog_groups.keys())
        logger.info(f"Deleting this slugs from trending and popular api: {json.dumps(slugs_error)}")
        data = data[~data['blog_slug'].isin(slugs_error)]
        blog_groups = data.groupby('blog_slug').groups
        
        # slugs_error = sanity_check_slug(db, blog_groups.keys())
        # print(len(slugs_error), slugs_error)

        page_view_sum = {}
        avg_time_mean = {}
        
        if type_analysis == "year":
            for slug, idx in blog_groups.items():
                page_view_sum[slug] = data.loc[idx]['uniquePageviews'].sum()
                avg_time_mean[slug] = data.loc[idx]['avgTimeOnPage'].mean()

            result = pd.DataFrame([{'slug': slug , 'sum_unique_page_views': page_view_sum[slug],
                                          'mean_avg_time': avg_time_mean[slug]} 
                                          for slug in page_view_sum.keys()])
            result["pageRank"] = np.log(result["sum_unique_page_views"]+1) + 8*np.log(result["mean_avg_time"]+1)
                                     
        elif type_analysis == "week":
            post_dates = {}
            for slug, idx in blog_groups.items():
                page_view_sum[slug] = data.loc[idx]['uniquePageviews'].sum()
                avg_time_mean[slug] = data.loc[idx]['avgTimeOnPage'].mean()
                post_dates[slug] = crud.get_blog_date_by_slug(db, slug)

            result = pd.DataFrame([{'slug': slug , 'sum_unique_page_views': page_view_sum[slug],
                                          'mean_avg_time': avg_time_mean[slug], 'post_date': post_dates[slug]}
                                          for slug in page_view_sum.keys()])
            
            today = datetime.datetime.now()
            if (today.month - 6)%12 != 0:
                start_month = (today.month - 6)%12
            elif np.abs(today.month - 6) == 6:
                start_month = 6
            else:
                start_month = 12
            start_date = '{}-{}-{}'.format(today.year if today.month > 6 else today.year - 1, 
                                           start_month, 1)
            end_date = '{}-{}-{}'.format(today.year, today.month, today.day)
            
            logger.info('start_date: {}'.format(start_date))
            logger.info('end_date: {}'.format(end_date))
            
            result = result[(result.post_date >= start_date) & (result.post_date <= end_date)]
            result["pageRank"] = np.log(result["sum_unique_page_views"]+1) + 2.7*np.log(result["mean_avg_time"]+1)
            
        # print(result.sort_values(by='pageRank', ascending=False).head(5))
        return result.sort_values(by='pageRank', ascending=False)
    
    def cache_populars(self):
        key='popular-blogs-api'
        value=self.data.slug.tolist()[:20]
        self.redis_db.set(key, json.dumps(value))
        logger.info("Finish cache populars")
        
    def cache_trending(self):
        key='trending-blogs-api'
        value=self.data.slug.tolist()[:20]
        self.redis_db.set(key, json.dumps(value))
        logger.info("Finish cache trending")
        
def sanity_check_slug(db, slugs):
    slug_quit = []
    for slug in tqdm(slugs, total=len(slugs), desc='sanity check for slugs from trending api'):
        date = crud.get_blog_date_by_slug(db, slug)
        if date is None:
            slug_quit.append(slug)
    return slug_quit

def get_term_industry_for_blog(blog_id, terms_taxonomy_id, db):
    term_id = crud.get_term_blog_taxonomy_id(db, int(blog_id))
    term_slug = crud.get_term_slug_by_term_id(db, term_id, terms_taxonomy_id)
    logger.info(term_slug)
    return term_slug

def get_blog_author_name_slug(post_author_id, db):
    return crud.get_author_slug_name_by_id(db, post_author_id)
    
def convert_date(date):
    return date.strftime("%b %d, %Y")

def tokenize(text, lang, removing_stops=False, lemmatize=False):
    tokens = preprocess_data(text, lang, removing_stops, lemmatize)
    return tokens

def get_slug_by_page_path(page_path):
    page_path_list = page_path.split("/")
    try:
        if page_path_list[1] == "blog":
            return page_path_list[2]
        else:
            return ''
    except:
        return ''
            
def initialize_analyticsreporting_api(KEY_FILE_LOCATION, SCOPES):
    """Initializes an Analytics Reporting API V4 service object.

    Returns:
        An authorized Analytics Reporting API V4 service object.
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
            KEY_FILE_LOCATION, SCOPES)

    # Build the service object.
    analytics = build('analyticsreporting', 'v4', credentials=credentials)

    return analytics

def check_load_model():
    health = False
    try:
        health = ScoringModel.check_is_loaded()
    except:
        pass
    return health
    

def detect_language(text):
    return 'es' if detect(text) != 'en' else 'en'

def transform_post_query_to_dataframe(post_query, param_query):
    posts = [vars(post) for post in post_query]
    # choose param_query
    posts = pd.DataFrame(posts)
    try:
        posts = posts[param_query].copy()
    except Exception as exception:
        logger.error("Error: {}".format(exception))
    return posts

def get_additional_data(get_db):
    for db in get_db():
        additional_data_artifacts = {
            "industries_info": crud.get_industries_info(db),
            "services_info": crud.get_services_info(db)
        }
        return list(additional_data_artifacts.values())
#     return ScoringModel.get_additional_artifacts()