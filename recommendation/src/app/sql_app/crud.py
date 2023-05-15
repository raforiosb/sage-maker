from sqlalchemy.orm import Session
from app.sql_app import models, schemas

def get_blog(db: Session, blog_id:int):
    return db.query(models.Blog).filter(models.Blog.id == blog_id).first()

def get_blog_by_slug(db: Session, slug:str):
    return db.query(models.Blog).filter(models.Blog.slug == slug).first()

def get_blog_date_by_slug(db: Session, slug:str):
    response = db.query(models.Blog.post_date).filter(models.Blog.slug == slug).first()
    try:
        return response[0]
    except Exception as e:
        # print('error: ' + slug)
        return None

def get_publish_blogs(db: Session):
    """Get The type: blogs and status: publish of wp_post tablet"""
    return db.query(models.Blog).filter(models.Blog.type == 'post', models.Blog.status == 'publish').all()

def get_industries_posts(db: Session):
    return db.query(models.Blog).filter(models.Blog.type == 'industries', models.Blog.status == 'publish').all()

def get_services_posts(db: Session):
    return db.query(models.Blog).filter(models.Blog.type == 'services', models.Blog.status == 'publish').all()

def get_services_info(db: Session):
    services_posts = get_services_posts(db)
    services_info = {blog.slug: " ".join(blog.slug.split("-")) + " " + blog.post_excerpt for blog in services_posts}
    return services_info

def get_industries_info(db: Session):
    industries_posts = get_industries_posts(db)
    industries_info = {blog.slug: " ".join(blog.slug.split("-")) + " " + blog.post_excerpt for blog in industries_posts}
    return industries_info

def get_industry_id_taxonomy(db: Session):
    return [ids[0] for ids in db.query(models.TermTaxonomy.id).filter(models.TermTaxonomy.taxonomy_type == "industry").all()]

def get_term_blog_taxonomy_id(db: Session, blog_id: int):
    return [ids[0] for ids in db.query(
                                    models.TermRelationship.term_taxonomy_id
                                ).filter(models.TermRelationship.blog_id == int(blog_id)).all()]

def get_term_slug_by_term_id(db: Session, term_id: list, industry_taxonomy_id: list):
    slugs = []
    for slug in db.query(models.Term.slug).filter(models.Term.id.in_(term_id),
                                                  models.Term.id.in_(industry_taxonomy_id)).all():
        if slug:
            slugs.append(slug[0])
    return slugs[0] if slugs else "other" # set other as a hotfix for empty term_slug

def get_author_slug_name_by_id(db: Session, author_blog_id: int):
    return db.query(models.Author.slug).filter(models.Author.id == author_blog_id).first()[0]