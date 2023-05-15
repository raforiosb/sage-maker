import argparse
import json
import sys
from koombea_blogs.data_module.data_processing import DataProcessing
from koombea_blogs.data_module.data_extraction import DataExtraction
from koombea_blogs.data_analysis.data_visualizing import DataVisualizing
from koombea_blogs.config.settings import get_settings
from koombea_blogs.config.logger import get_logger
from koombea_blogs.db.deps import get_connection
from koombea_blogs.connection.database_tunnel import get_tunnel

logger = get_logger()

def analysis_process(english_data, spanish_data, settings):
    logger.info("Begins automatic analysis with visualization of our data")
    _, english_data = zip(*english_data)
    _, spanish_data = zip(*spanish_data)
    analyzing_process = DataVisualizing(english_data, spanish_data, settings = settings)

    logger.info("Analyze frequency data for english")
    analyzing_process.make_frequency_analysis(
        title="Frequency ditribution analysis on english data",
        sup_titles=[
            "Most repeated english words in blogs",
            "Less repeated english words in blogs",
        ],
        lang="en",
        n_top=30,
    )
    analyzing_process.make_frequency_analysis(
        title="Frequency ditribution analysis on spanish data",
        sup_titles=[
            "Most repeated spanish words in blogs",
            "Less repeated spanish words in blogs",
        ],
        lang="es",
        n_top=30,
    )

    logger.info("Analyze tfidf weights for our blogs data")
    analyzing_process.make_tfidf_mean_analysis(
        title="English tfidf mean weights across blogs",
        sup_titles=[
            "Highest tfidf weights english words in blogs",
            "Lowest tfidf weights english words in blogs",
        ],
        lang="en",
        n_top=30,
    )
    analyzing_process.make_tfidf_mean_analysis(
        title="Spanish tfidf mean weights across blogs",
        sup_titles=[
            "Highest tfidf weights spanish words in blogs",
            "Lowest tfidf weights spanish words in blogs",
        ],
        lang="es",
        n_top=25,
    )

    logger.info("Begins analysis of tfidf weights on topics")
    analyzing_process.make_tfidf_by_topics_analysis(
        title="English tfidf topic weights analysis",
        lang="en",
        n_top=4,
    )
    analyzing_process.make_tfidf_by_topics_analysis(
        title="Spanish tfidf topic weights analysis",
        lang="es",
        n_top=4,
    )

    logger.info("Begins analysis of wordcloud tfidf weights on topics")
    analyzing_process.make_tfidf_wordcloud_by_topics_analysis(
        title="Wordcloud for english topic analysis", lang="en", n_top=10
    )
    analyzing_process.make_tfidf_wordcloud_by_topics_analysis(
        title="Wordcloud for spanish topic analysis", lang="es", n_top=10
    )

def extraction_process(conn, settings):
    extraction_process = DataExtraction(conn, settings)
    logger.info("Begins extraction process from db")
    extraction_process.extract()
    logger.info("Begins cleaning process")
    processing_process = DataProcessing(extraction_process.blogs, settings)
    processing_process.preprocess_data()
    return (
        extraction_process.blogs,
        processing_process.english_data,
        processing_process.spanish_data,
    )


def saving_artifacts(settings, output_path, blogs_df, **languages_data):
    logger.info("Saving blogs_df file to {}".format(output_path))
    blogs_df.to_csv(output_path + f"/blogs_df_{settings.MYSQL_DBNAME}.csv", index=False)
    for lang_name, lang_data in languages_data.items():
        logger.info("Saving {} to {}".format(lang_name, output_path))
        with open(
            output_path + "/" + lang_name + f"_{settings.MYSQL_DBNAME}.json", "w"
        ) as file:
            json.dump(lang_data, file)


def main(output_path, conn, settings):
    logger.info("Preprocessing job begins")
    blogs_df, en_data, es_data = extraction_process(conn, settings)
    analysis_process(english_data=en_data, spanish_data=es_data, settings=settings)
    saving_artifacts(settings, output_path, blogs_df, en_data=en_data, es_data=es_data)
    logger.info("Finished processing job!")


if __name__ == "__main__":
    # Execute a processor job to download and process all data from koombea db
    # And upload to s3 bucket using sagemaker pipelines this will be part of a pipeline
    parser = argparse.ArgumentParser(description="Processor job for blogs data")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path where to save the processor's ouputs artifacts,"
        " this path is selected by sagemaker and"
        " it will automatically upload the artifacts to the selected s3 bucket",
        required=True,
    )
    args = parser.parse_args()
    settings = get_settings()
    tunnel = get_tunnel(settings)
    tunnel.start()    
    # Initialize db
    settings.set_assemble_db_url_connection(tunnel.local_bind_port)
    logger.info("Connect to the following sqlalchemy url: {}".format(settings.SQLALCHEMY_DATABASE_URL))
    for conn in get_connection(settings) :
        main(args.output_path, conn, settings)
    tunnel.close()
