from sqlalchemy import (Boolean, Column, ForeignKey, 
                        Integer, String, DateTime)
from sqlalchemy.orm import relationship
from app.sql_app.database import Base
from datetime import datetime

class Blog(Base):
    __tablename__ = "wp_posts"
    """
    TABLE wp_posts
    Need Columns:
        ID: id
        post_title: title
        post_content: content
        post_name: slug
    """
    id = Column('ID', Integer, primary_key=True, index=True)
    title = Column('post_title', String)
    content = Column('post_content', String)
    slug = Column('post_name', String, index=True)
    status = Column('post_status', String)
    type = Column('post_type', String)
    post_date = Column('post_date', DateTime)
    post_excerpt = Column('post_excerpt', String)
    author_id = Column('post_author', Integer, ForeignKey("wp_users.ID"))
    
class Author(Base):
    """
    TABLE wp_users
    """
    __tablename__ = "wp_users"
    id = Column('ID', Integer, primary_key=True, index=True)
    name = Column('display_name', String, primary_key=True, index=True)
    slug = Column('user_nicename', String, primary_key=True, index=True)    

class Term(Base):
    """
    TABLE wp_terms
    """
    __tablename__ = "wp_terms"
    id = Column('term_id', Integer, primary_key=True, index=True)
    name = Column('name', String, index=True)
    slug = Column('slug', String, index=True)
    
class TermRelationship(Base):
    """
    TABLE wp_term_relationships
    """
    __tablename__ = "wp_term_relationships"
    blog_id = Column('object_id', Integer, ForeignKey("wp_posts.ID"), 
                     primary_key=True, index=True)
    term_taxonomy_id = Column('term_taxonomy_id', Integer,  
                              ForeignKey("wp_term_taxonomy.term_taxonomy_id"), index=True)
    
class TermTaxonomy(Base):
    """
    TABLE wp_term_taxonomy
    """
    __tablename__ = "wp_term_taxonomy"
    id = Column('term_taxonomy_id', Integer, primary_key=True, index=True)
    term_id = Column('term_id', Integer, ForeignKey("wp_terms.term_id"))
    taxonomy_type = Column('taxonomy', String)