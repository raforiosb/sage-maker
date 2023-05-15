import os
import json
import faiss
import pandas as pd
from faiss import normalize_L2
from fse.models import SIF
from gensim.models.phrases import Phraser
from app.utils import logger
from redis import Redis

PREFIX = "/opt/ml/"
ARTIFACTS_PATH = os.path.join(PREFIX, 'model')
RELOAD = True
# redis_db = Redis()

class ScoringModel(object):
    
    model_artifacts = { 
        "fse_model_en": None, 
        "fse_model_es": None,
        "phrases_model_en": None,
        "phrases_model_es": None,
    }
    index_artifacts = {
        # "index_slug_es": None,
        # "index_slug_en": None,
        "data_es": None,
        "data_en": None
    }
    index_faiss_artifacts = {
        "index_en": None,
        "index_es": None
    }
    additional_data_artifacts = {
        "industries_info": None,
        "services_info": None
    }
    popular_data_artifacts = {
        "popular_result": None,
        "trending_result": None
    }
    
    @classmethod
    def get_artifacts(cls):
        
        for name, obj in cls.model_artifacts.items():
            if obj is None:
                _name = name.split('_')
                if 'fse' in _name:
                    cls.model_artifacts[name] = SIF.load(os.path.join(ARTIFACTS_PATH, 
                                                                     name+'.pickle'))
                elif 'phrases' in _name:
                    cls.model_artifacts[name] = Phraser.load(os.path.join(ARTIFACTS_PATH,
                                                                         name+'.pickle'))
                
        for name, obj in cls.index_artifacts.items():
            if obj is None:
                cls.index_artifacts[name] = pd.read_json(os.path.join(ARTIFACTS_PATH,
                                                                           name+'.json'))
                cls.index_artifacts[name]["post_date"] = pd.to_datetime(cls.index_artifacts[name].post_date_str,
                                                                       infer_datetime_format=True)
            
        for name, obj in cls.index_faiss_artifacts.items():
            if obj is None:
                logger.info("loading faiss indexing " + name)
                _name = name.split('_')
                if 'es' in _name:
                    _temp_vectors = cls.model_artifacts["fse_model_es"].sv.vectors
                    normalize_L2(_temp_vectors)
                    cls.index_faiss_artifacts[name] = faiss.IndexFlatIP(_temp_vectors.shape[1])
                    cls.index_faiss_artifacts[name].add(_temp_vectors)
                elif 'en' in _name:
                    _temp_vectors = cls.model_artifacts["fse_model_en"].sv.vectors
                    normalize_L2(_temp_vectors)
                    cls.index_faiss_artifacts[name] = faiss.IndexFlatIP(_temp_vectors.shape[1])
                    cls.index_faiss_artifacts[name].add(_temp_vectors)
                    
        for name, obj in cls.additional_data_artifacts.items():
            if obj is None:
                logger.info("Loading additional data")
                cls.additional_data_artifacts[name] = json.load(open(os.path.join(
                    ARTIFACTS_PATH, name+".json"
                )))        
                
        for name, obj in cls.popular_data_artifacts.items():
            if obj is None:
                logger.info("Loading additional data")
                cls.popular_data_artifacts[name] = json.load(open(os.path.join(
                    ARTIFACTS_PATH, name+".json"
                )))   
                
        return list(cls.model_artifacts.values()) + list(cls.index_artifacts.values()) \
                 + list(cls.index_faiss_artifacts.values()) + list(cls.additional_data_artifacts.values()) \
                    + list(cls.popular_data_artifacts.values())
                
            
    @classmethod
    def check_is_loaded(cls):
        total = len(cls.model_artifacts.keys()) + \
        len(cls.index_artifacts.keys()) + \
        len(cls.index_faiss_artifacts.keys()) + \
        len(cls.additional_data_artifacts.keys()) + \
        len(cls.popular_data_artifacts.keys())
        
        return sum([artifact is not None for artifact in cls.get_artifacts()]) == total

    @classmethod
    def predict_new(cls, input_data, lang, top_k):
        """
        artifats:
            fse_model_en -> 0
            fse_model_es -> 1
            ph_model_en -> 2
            ph_model_es -> 3
            idx_slug_es  -> 4
            idx_slug_en -> 5
            idx_model_en -> 6
            idx_model_es -> 7
        """
        artifacts = cls.get_artifacts()
        ngram_model = artifacts[2] if lang == 'en' else artifacts[3]
        fse_model = artifacts[0] if lang == 'en' else artifacts[1]
        idx_slug = artifacts[5] if lang == 'en' else artifacts[4]
        idx_model = artifacts[6] if lang == 'en' else artifacts[7]
        input_data = list(ngram_model[input_data])
        print(input_data)
        vector_data = fse_model.infer([(input_data, 0)])
        normalize_L2(vector_data)
        # print(idx_model)
        _, I = idx_model.search(vector_data, top_k + 1)
        return [idx_slug.slug.values[_idx] for _idx in I.squeeze()[1:] if _idx < len(idx_slug)]
    
    @classmethod
    def get_vectors_cache(cls, input_data, lang):
        artifacts = cls.get_artifacts()
        ngram_model = artifacts[2] if lang == 'en' else artifacts[3]
        fse_model = artifacts[0] if lang == 'en' else artifacts[1]
        input_data = list(ngram_model[input_data])
        return fse_model.infer([(input_data, 0)])
    
    @classmethod
    def get_index_model_cache(cls, vectors):
        normalize_L2(vectors)
        index_model = faiss.IndexFlatIP(vectors.shape[1])
        index_model.add(vectors)
        return index_model
    
    @classmethod
    def predict_cache(cls, vector_slug, index_model):
        normalize_L2(vector_slug)
        _, I = index_model.search(vector_slug, 21)
        return list(I.squeeze()[1:])
    
    @classmethod
    def predict_cache_additional_data(cls, data, vector_slug, index_model, name, slug, blocked_slugs=None):
        data["term_slug"] = data["term_slug"].apply(lambda x: "".join(x.split("-")))
        normalize_L2(vector_slug)
        D, I = index_model.search(vector_slug, 100)
        # print(data.columns)
        results =  [(float(_val), data.loc[_idx]["post_date"].year, "".join(data.loc[_idx]["term_slug"].split("-")), 
                     data.loc[_idx]["slug"], data.loc[_idx]["user_nicename"], data.loc[_idx]["post_date_str"])
                    for _val, _idx in zip(D.squeeze(), I.squeeze())]
        if name == "industries":
            if slug in data.term_slug.values:
                results = [result for result in results if result[4] != "guest-author" and result[2] == slug]
            elif slug == "edtech":
                results = [result for result in results if result[4] != "guest-author" and result[2] == "hitech"]
        elif name == "services":
            results = [result for result in results if result[4] != "guest-author"]
        
        # ranking, year, industry_term_slug, slug_blog, guest, date_str
        max_rank = max(results, key=lambda x: x[0])[0]
        max_year = max([result for result in results if result[0] > max_rank - 0.15], key=lambda x: x[1])[1]
        results = [result for result in results if result[1] > max_year - 2]
        results.sort(key=lambda x: -x[0])
        
#         if blocked_slugs:
#             results = [result for result in results if not result[3] in blocked_slugs]
        
        return results
    
    @classmethod
    def predict_special_data(cls, input_data, top_k, key_value, slug, slug_blocked=None):
        input_vector = cls.get_vectors_cache(input_data, "en")
        index_model_en = cls.get_index_en_model()
        data_en = cls.get_data_en_artifact()
        # print(data_en.head())
        results = cls.predict_cache_additional_data(data_en, input_vector, index_model_en, key_value, slug)
        # print(pd.DataFrame(results))
        # results = [(val[3], val[-1]) for val in results]
        results = [val[3] for val in results]
        if slug_blocked:
            results = [result for result in results if not result in slug_blocked]
        return results[:top_k]
        
    @classmethod
    def get_additional_artifacts(cls):
        return cls.get_artifacts()[-4:-2]
    
    @classmethod
    def get_index_en_model(cls):
        return cls.get_artifacts()[6]
    
    @classmethod
    def get_data_en_artifact(cls):
        data_en = cls.get_artifacts()[5]
        data_en["user_nicename"] = data_en["author_slug_name"]
        return data_en
    
    @classmethod
    def predict_popular_blogs(cls, top_k, type_analysis):
        if type_analysis=="year":
            return cls.get_artifacts()[-2][:top_k]
        elif type_analysis=="week":
            return cls.get_artifacts()[-1][:top_k]