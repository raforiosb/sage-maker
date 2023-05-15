from pathlib import Path
from typing import Union
import dotenv
import os

CODE_CHANNEL = os.environ.get("SM_MODULE_DIR")

if not os.environ.get("STAGE"):
    dotenv.load_dotenv(os.path.join(CODE_CHANNEL, "vars.env"))
    dotenv.load_dotenv(os.path.join(CODE_CHANNEL, "vars.staging.env"))


from koombea_model.config.logger import logger
from koombea_model.config.settings import settings, PathSettings
from koombea_model.config.model_hyperparameters import HyperParameterModel
from koombea_model.general_utils import (
    parse_and_validate_arguments,
    get_channel_path,
    pretty_string,
    quiete_logs,
)
from koombea_model.model_analysis.model_visualizing import ModelVisualizeAnalysis
from koombea_model.data import Data
from koombea_model.vector_model.blogs_model import BlogsModel, Top
import argparse


def train_blogs_vectors_models(data: Data, model_dir: Path, **hyperparameters):
    # train english blogs vector model
    bv_model_en = BlogsModel("bv_model", lang="en", output_path=model_dir)
    bv_model_en.train(data=data.en_data, components=0, **hyperparameters)
    # train spanish blogs vector model
    bv_model_es = BlogsModel("bv_model", lang="es", output_path=model_dir)
    bv_model_es.train(data=data.es_data, components=0, **hyperparameters)
    # Cache predictions for english model and spanish model
    quiete_logs("fse")
    bv_model_en.cache_results(data.blogs_df, data.en_idx2ids)
    bv_model_es.cache_results(data.blogs_df, data.es_idx2ids, max_top=data.es_num_blogs)
    # return model artifacts
    return bv_model_en, bv_model_es


def make_analysis(
    bv_model_en: BlogsModel,
    bv_model_es: BlogsModel,
    data: Data,
):
    model_analysis = ModelVisualizeAnalysis(bv_model_en, bv_model_es, data)
    model_analysis.model_summary(
        targets=[top for top in Top], lang="en", top_distance_bad_good=5
    )
    model_analysis.model_summary(
        targets=[top for top in Top], lang="es", top_distance_bad_good=5
    )


def save_artifacts(*artifacts: Union[BlogsModel, Data]):
    for artifact in artifacts:
        artifact.save()
        artifact.build_metadata()


def main(hyperparameters: HyperParameterModel, path_settings: PathSettings):

    # Loading data, blogs df table and training data for english and spanish blogs
    data = Data(**path_settings.dict())
    # train models
    bv_model_en, bv_model_es = train_blogs_vectors_models(
        data, path_settings.model_dir, **hyperparameters.dict()
    )
    # Make analysis for english model
    make_analysis(bv_model_en=bv_model_en, bv_model_es=bv_model_es, data=data)
    # Save Model
    save_artifacts(bv_model_en, bv_model_es, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train run script to train a fast document embedding model"
    )
    # Gensim hyperparameters
    parser.add_argument(
        "--min_count",
        type=int,
        default=0,
        help="Ignores all words with total frequency lower than this",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help=" Dimensionality of the word vectors. And blogs vectors.",
    )
    parser.add_argument(
        "--sg",
        type=int,
        default=1,
        choices=[0, 1],
        help="Training algorithm: 1 for skip-gram; otherwise CBOW. values: {0, 1}",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=15,
        help="Maximum distance between the current and predicted word within a sentence.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=50,
        help="Number of iterations (epochs) over the corpus.",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=6e-5,
        help="The threshold for configuring which higher-frequency words are randomly downsampled,"
        "useful range is (0, 1e-5).",
    )
    parser.add_argument(
        "--hs",
        type=int,
        choices=[0, 1],
        default=0,
        help=" If 1, hierarchical softmax will be used for model training."
        "If 0, and `negative` is non-zero, negative sampling will be used.",
    )
    parser.add_argument(
        "--negative",
        type=int,
        default=15,
        help="If > 0, negative sampling will be used, the int for negative specifies how many "
        "noise words should be drawn (usually between 5-20)."
        "If set to 0, no negative sampling is used.",
    )
    parser.add_argument(
        "--ns_exponent",
        default=-0.5,
        help="The exponent used to shape the negative sampling distribution. A value of 1.0 samples exactly in "
        "proportion to the frequencies, 0.0 samples all words equally, while a negative value samples "
        "low-frequency words more than high-frequency words. The popular default value of 0.75 was chosen by "
        "the original Word2Vec paper.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.environ.get("SM_NUM_CPUS"),
        help="workers for multiprocessing training with gensim and fse.",
    )
    # Data Channels Path
    parser.add_argument(
        "--blogs_df_path",
        type=str,
        default=os.path.join(get_channel_path(0), settings.BLOGS_DF_NAME_FILE_CSV),
        help="Blogs df file, in order to get access to the corresponding data in our blogs",
    )
    parser.add_argument(
        "--en_data_path",
        type=str,
        default=os.path.join(get_channel_path(0), settings.EN_DATA_NAME_FILE_JSON),
        help="english proprocessed data mapping list of tokens with blogs id",
    )
    parser.add_argument(
        "--es_data_path",
        type=str,
        default=os.path.join(get_channel_path(0), settings.ES_DATA_NAME_FILE_JSON),
        help="spanish proprocessed data mapping list of tokens with blogs id",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
        help="output path directory where to save output artifacts. default: /opt/ml/",
    )

    # parse arguments
    args = parser.parse_args()
    # validate hyperparameters and pathsettings
    hyperparameters = parse_and_validate_arguments(
        HyperParameterModel, name="hyperparameters", args=args.__dict__
    )
    path_settings = parse_and_validate_arguments(
        PathSettings, name="path_input_settings", args=args.__dict__
    )
    # Log hyperparameters and path_settings
    logger.info("hyperparameters:\n{}".format(pretty_string(hyperparameters.dict())))
    logger.info("path_settings:\n{}".format(pretty_string(path_settings.dict())))
    # execute training
    main(hyperparameters, path_settings)
