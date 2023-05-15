from sqlalchemy.engine import Connection
import pandas as pd
from ..config.logger import get_logger
from ..data_module import data_utils
from typing import Dict, List
import json
import wandb
from ..config.deps import get_wandb_run_extraction

logger = get_logger()


class BlogsInformation:
    initial_blogs_columns: List[str] = [
        "ID",
        "post_author",
        "post_date",
        "post_content",
        "post_title",
        "post_excerpt",
        "post_name",
        "post_modified",
    ]
    final_blogs_columns: List[str] = initial_blogs_columns + [
        "author_name",
        "industry_id",
        "industry_name",
        "industry_slug",
        "category_id",
        "category_name",
        "category_slug",
        "lang",
        "image_alt",
        "image",
        "post_url",
    ]


class DBInformation:
    necessary_table_names: List[str] = [
        "wp_posts",
        "wp_users",
        "wp_term_taxonomy",
        "wp_terms",
        "wp_term_relationships",
        "wp_options",
        "wp_postmeta",
    ]


class DataExtraction:
    def __init__(self, conn: Connection, settings) -> None:
        self.settings = settings
        self.tables: Dict[str, pd.DataFrame] = {
            table_name: pd.read_sql_table(table_name, conn)
            for table_name in DBInformation.necessary_table_names
        }
        self.site_url: str = self.tables["wp_options"][
            self.tables["wp_options"]["option_name"] == "siteurl"
        ]["option_value"].item()
        content_wp: str = self.tables["wp_options"][
            self.tables["wp_options"]["option_name"] == "autoptimize_css_exclude"
        ]["option_value"].item()
        self.content_wp: str = content_wp.split(",")[1].strip().strip("/")
        logger.debug("Site url retrived from wp_options table is: " + self.site_url)
        logger.debug(
            "First part of the url for the images is: "
            + self.site_url
            + "/"
            + self.content_wp
            + "/"
        )

    def extract(self, test=False):
        # the test argument allow the process to check just for one blog
        logger.info(
            f"Extracting blogs data from db: testingmode = {test}, db_name = {self.settings.MYSQL_DBNAME}"
        )
        logger.info("getting published blogs from wp_posts table")
        self.blogs = data_utils.get_published_blogs(
            self.tables["wp_posts"], BlogsInformation.initial_blogs_columns, test
        )
        # Get users author_name information to blogs
        logger.info(
            "getting user display_name information for blogs dataset from wp_users table"
        )
        self.blogs = data_utils.join_users2blogs(self.blogs, self.tables["wp_users"])
        # Get industry information to blogs
        logger.info(
            "getting industry slug, and name information for blogs dataset"
            " from wp_terms, wp_term_relationships and wp_term_taxonomy tables"
        )
        self.blogs = data_utils.join_terms2blogs(
            self.blogs,
            self.tables["wp_terms"],
            self.tables["wp_term_taxonomy"],
            self.tables["wp_term_relationships"],
            target="industry",
        )
        # Get conten_type category information to blogs
        logger.info(
            "getting category slug and name information for blogs dataset"
            " from wp_terms, wp_term_relationships, and wp_term_taxonomy tables"
        )
        self.blogs = data_utils.join_terms2blogs(
            self.blogs,
            self.tables["wp_terms"],
            self.tables["wp_term_taxonomy"],
            self.tables["wp_term_relationships"],
            target="category",
        )
        # Getting blogs languages
        logger.info(
            "getting language for each blog given the category slug found before"
        )
        self.blogs["lang"] = self.blogs["category_slug"].apply(
            lambda category_slug_value: category_slug_value
            if category_slug_value == "es"
            else "en"
        )
        # Getting image_alt info to blogs
        logger.info("getting image_alt information to blogs dataset")
        self.blogs = data_utils.join_imagealt2blogs(
            self.blogs, self.tables["wp_postmeta"]
        )
        # getting image_url info to blogs
        logger.info("getting image_url information to blogs dataset")
        self.blogs = data_utils.join_imageurl2blogs(
            self.blogs,
            self.tables["wp_postmeta"],
            site_url=self.site_url,
            content_wp=self.content_wp,
        )
        # getting post_url info to blogs
        logger.info("getting post_url information to blogs dataset")
        self.blogs["post_url"] = (
            self.site_url
            + "/"
            + self.blogs["category_slug"]
            + "/"
            + self.blogs["post_name"]
            + "/"
        )
        # build metadata information for blogs dataset
        if not test:
            self.build_metadata()

    def build_metadata(self) -> None:
        # build some information about the data we gatther
        # and log this to wandb
        self.total_num_blogs = self.blogs.shape[0]
        self.spanish_num_blogs = self.blogs[self.blogs["lang"] == "es"].shape[0]
        self.english_num_blogs = self.total_num_blogs - self.spanish_num_blogs
        extra_data_dict = {
            "total_num_blogs": self.total_num_blogs,
            "spanish_num_blogs": self.spanish_num_blogs,
            "english_num_blogs": self.english_num_blogs,
        }
        extra_data = json.dumps(
            extra_data_dict,
            indent=True,
        )
        logger.debug(f"extra data information:\n{extra_data}")
        # Create dataset artifact for wandb_run_extraction job
        extraction_artifact = wandb.Artifact(
            f"blogs_{self.settings.MYSQL_DBNAME}",
            type="dataset-search",
            description="dataset search artifact",
            metadata={
                "db": self.settings.MYSQL_DBNAME,
                "stage": self.settings.STAGE,
            },
        )
        # add artifact
        extraction_artifact.add(
            wandb.Table(dataframe=self.blogs), name=f"{self.settings.MYSQL_DBNAME}_blogs_df"
        )
        # Initialize run
        logger.debug(
            "initializing wandb run extraction to keep track of artifacts and logs"
        )
        wandb_run_extraction = get_wandb_run_extraction(self.settings)
        # Log to wandb
        wandb_run_extraction.log(
            {
                "extra_data_bar": wandb.plot.bar(
                    wandb.Table(
                        data=[[label, val] for label, val in extra_data_dict.items()],
                        columns=["lang_blog", "#blogs"],
                    ),
                    "lang_blog",
                    "#blogs",
                    title="Language blogs distribution",
                )
            }
        )
        # Log table data to wandb
        wandb_run_extraction.log_artifact(extraction_artifact)
        # finish wandb run extraction
        logger.debug(
            "Finishing wandb run extraction to keep track of artifacts and logs"
        )
        wandb_run_extraction.finish()
