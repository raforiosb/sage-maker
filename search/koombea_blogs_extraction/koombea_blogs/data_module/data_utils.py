import pandas as pd
from typing import Any, Dict, List, Union
import phpserialize as phps


def get_published_blogs(
    wp_posts: pd.DataFrame, blogs_columns: List[str], test: bool
) -> pd.DataFrame:
    """filter post by post_type post and post_status publish,
    and select just the necessary columns, if test is true return just the first blog,
    that  means a dataframe os just one row.

    Args:
        wp_posts (pd.DataFrame): wp_posts table dataframe
        blogs_columns (List[str]): necessary columns
        test (bool): wether we are in test mode or not
    """
    blogs_df = wp_posts[
        (wp_posts["post_type"] == "post") & (wp_posts["post_status"] == "publish")
    ][blogs_columns]
    blogs_df.reset_index(drop=True, inplace=True)  # reset index
    return blogs_df if not test else blogs_df.head(1)


def join_users2blogs(blogs: pd.DataFrame, wp_users: pd.DataFrame) -> pd.DataFrame:
    """Join users necessary information from wp_users table to
    blogs dataset.
    Args:
        blogs (pd.DataFrame): blogs dataframe we are constructing
        wp_users (pd.DataFrame): wp_users dataframe from db table

    Returns:
        [pd.DataFrame]: blogs join information with wp_users
    """
    wp_users = wp_users[["ID", "display_name"]]  # selecting the information to join
    wp_users.columns = ["post_author", "author_name"]
    return blogs.join(wp_users.set_index("post_author"), on="post_author")


def join_terms2blogs(
    blogs: pd.DataFrame,
    wp_terms: pd.DataFrame,
    wp_term_taxonomy: pd.DataFrame,
    wp_term_relationships: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    """Join information from terms on blogs, selecting from target value

    Args:
        blogs (pd.DataFrame): pandas dataframe we are building for blogs
        wp_terms (pd.DataFrame): wp_terms pandas dataframe from db table
        wp_term_taxonomy (pd.DataFrame): wp_term_taxonomy pandas dataframe from db table
        wp_term_relationships (pd.DataFrame): wp_term_relationships pandas dataframe from db table
        target (str): specific target for type of term - industry or category - content_type

    Returns:
        [pd.DataFrame]: joined information from terms on blogs
    """
    # select necessary columns
    wp_terms = wp_terms[["term_id", "name", "slug"]]
    wp_term_taxonomy = wp_term_taxonomy[["term_taxonomy_id", "term_id", "taxonomy"]]
    wp_term_relationships = wp_term_relationships[["object_id", "term_taxonomy_id"]]
    # change columns for wp_term_relationships
    wp_term_relationships.columns = ["ID", "term_taxonomy_id"]
    # join wp_term_taxonomy and wp_terms on wp_term_taxonomy
    wp_term_taxonomy = wp_term_taxonomy.join(
        wp_terms.set_index("term_id"), on="term_id"
    )
    # filter wp_term_taxonomy by target industry or content_type
    wp_term_taxonomy = wp_term_taxonomy[wp_term_taxonomy["taxonomy"] == target]
    # select just the needed columns
    wp_term_taxonomy = wp_term_taxonomy[["term_taxonomy_id", "term_id", "name", "slug"]]
    # change columns name
    wp_term_taxonomy.columns = [
        "term_taxonomy_id",
        f"{target}_id",
        f"{target}_name",
        f"{target}_slug",
    ]
    # join wp_term_taxonomy and wp_term_relationships in wp_term_relationships
    wp_term_relationships = wp_term_relationships.join(
        wp_term_taxonomy.set_index("term_taxonomy_id"), on="term_taxonomy_id"
    )
    # Drop na values
    wp_term_relationships.dropna(inplace=True)
    # Drop repeated ID and keep first
    wp_term_relationships.drop_duplicates(subset=["ID"], keep="first", inplace=True)
    # Drop unnecesary columns
    wp_term_relationships.drop(columns=["term_taxonomy_id"], inplace=True)
    # joins wp_term_relationships and blogs on blogs
    return blogs.join(wp_term_relationships.set_index("ID"), on="ID")


def extract_image_url(site_url: str, content_wp: str, value: str) -> str:
    """Extract image url from php serialize information

    Args:
        site_url (str): site url from corresponding db enviroment,
            extracted from wp_options table
        content_wp (str): intermediate path to images content,
            extracted from wp_options table
        value (str): value from meta_value columns in wp_options table
            containing the php serialize _wp_attachment_metadata info

    Returns:
        str: image url site_url/content_wp/image_file
    """
    # deserialize with php
    attachment_metadata: Dict[str, Union[Any, str]] = phps.loads(
        value.encode(), decode_strings=True
    )
    # get original file name
    file_name_split = attachment_metadata["file"].split(
        "/"
    )  # split in case we need to change the last index for the specifig size if there exist
    files_size_info = attachment_metadata["sizes"]
    if files_size_info:
        medium_file_info = files_size_info.get("medium")
        thumb_file_info = files_size_info.get("thumbnail")
        if medium_file_info:
            file_name_split[-1] = medium_file_info["file"]
        elif thumb_file_info:
            file_name_split[-1] = thumb_file_info["file"]
    return site_url + "/" + content_wp + "/" + "/".join(file_name_split)


def get_blog2thumbnail_id_map(
    blogs: pd.DataFrame, wp_postmeta: pd.DataFrame
) -> pd.DataFrame:
    """Get pandas dataframe mapping from blog_id `ID` to
    thumbnaild_id `thumbnaild_id`

    Args:
        blogs (pd.DataFrame): Blogs dataset we are constructing
        wp_postmeta (pd.DataFrame): wp_postmeta pandas dataframe from db table

    Returns:
        pd.DataFrame: pandas dataframe mapping `ID` (blog id columns in blogs) to
            `thumbnail_id`
    """
    # Filter by blogs ID and meta key _thumbnail_id
    blog2thumbnail = wp_postmeta[
        (wp_postmeta["post_id"].isin(blogs["ID"]))
        & (wp_postmeta["meta_key"] == "_thumbnail_id")
    ]
    # Select the necessary columns
    blog2thumbnail = blog2thumbnail[["post_id", "meta_value"]]
    # cast _thumbnail_id meta_value to int64
    blog2thumbnail["meta_value"] = blog2thumbnail["meta_value"].astype("int")
    # change columns name to the corresponding
    blog2thumbnail.columns = ["ID", "thumbnail_id"]
    return blog2thumbnail


def get_postmeta_image(
    blogs: pd.DataFrame, wp_postmeta: pd.DataFrame, target: str
) -> pd.DataFrame:
    """Get postmeta target data from wp_postmeta

    Args:
        blogs (pd.DataFrame):  blogs dataset we are building
        wp_postmeta (pd.DataFrame): wp_postmeta pandas dataframe from db table
        target (str):  string that must be either `image_alt` or `metadata`

    Returns:
        pd.DataFrame:  pandas dataframe containing ID (blog id columns in blogs)
            to postmeta target data
    """
    # get auxiliary mapping from blog ID to thumbnaild_id
    blog2thumbnaild = get_blog2thumbnail_id_map(blogs, wp_postmeta)
    # get target metakey
    metakey_df = wp_postmeta[
        (wp_postmeta["post_id"].isin(blog2thumbnaild["thumbnail_id"]))
        & (wp_postmeta["meta_key"] == "_wp_attachment_{}".format(target))
    ]
    # select the necessary columns
    metakey_df = metakey_df[["post_id", "meta_value"]]
    # change columns
    col_name = target if target == "image_alt" else "image"
    metakey_df.columns = ["thumbnail_id", col_name]
    # joins metakey data frame containing the meta value to auxiliary mapping df
    return blog2thumbnaild.join(
        metakey_df.set_index("thumbnail_id"), on="thumbnail_id"
    ).drop(columns="thumbnail_id")


def join_imagealt2blogs(blogs: pd.DataFrame, wp_postmeta: pd.DataFrame) -> pd.DataFrame:
    """join image_alt to blogs dataframe on `ID` (blog_id)

    Args:
        blogs (pd.DataFrame): blogs dataset we are building
        wp_postmeta (pd.DataFrame): wp_postmeta pandas dataframe from db table

    Returns:
        pd.DataFrame: blogs dataframe with joining information from image_alt
    """
    # get image alt dataframe
    imagealt_df = get_postmeta_image(blogs, wp_postmeta, target="image_alt")
    # joins imagealt to blogs and fillna values
    blogs = blogs.join(imagealt_df.set_index("ID"), on="ID")
    blogs["image_alt"].fillna(value=blogs["post_title"], inplace=True)
    return blogs


def join_imageurl2blogs(
    blogs: pd.DataFrame, wp_postmeta: pd.DataFrame, site_url: str, content_wp: str
) -> pd.DataFrame:
    """Join image url to blogs dataframe on `ID` (blog_id)

    Args:
        blogs (pd.DataFrame): blogs dataset we are building
        wp_postmeta (pd.DataFrame): wp_postmeta pandas dataframe from db table
        site_url (str): site url from corresponding db enviroment,
            extracted from wp_options table
        content_wp (str): intermediate path to images content,
            extracted from wp_options table
    Returns:
        pd.DataFrame: joining df from imageurl information to blogs
    """
    # get image dataframe
    image_df = get_postmeta_image(blogs, wp_postmeta, target="metadata")
    # extract image_url
    image_df["image"] = image_df["image"].apply(
        lambda image_val: extract_image_url(site_url, content_wp, image_val)
    )
    # joins to blog
    return blogs.join(image_df.set_index("ID"), on="ID")
