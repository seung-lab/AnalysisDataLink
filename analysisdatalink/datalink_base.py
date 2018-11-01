from collections import defaultdict
from shapely import wkb
import geopandas as gpd
from emannotationschemas.base import flatten_dict
from emannotationschemas import get_schema
from emannotationschemas.models import make_annotation_model, make_cell_segment_model
from geoalchemy2.shape import to_shape, from_shape
from geoalchemy2 import func as geo_func
from geoalchemy2.elements import WKBElement
from shapely import geometry
from sqlalchemy import func
from sqlalchemy import and_
import neuroglancer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
import numpy as np
import pandas as pd
import os

def wkb_to_numpy(wkb):
    shp=to_shape(wkb)
    return np.array([shp.xy[0][0],shp.xy[1][0], shp.z], dtype=np.int)

def fix_wkb_columns(df):
    if len(df) > 0:
        for colname in df.columns:
            if isinstance(df.at[0,colname], WKBElement):
                df[colname] = df[colname].apply(wkb_to_numpy)
    return df

class DataLink():
    def __init__(self, dataset, version, database_uri=None):
        if database_uri is None:
            database_uri = os.getenv('DATABASE_URI')
            assert database_uri is not None
        self.database_uri = database_uri

        engine = create_engine(self.database_uri, echo=False)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.dataset = dataset
        self.data_version = version
        self.models = {}
        make_cell_segment_model(self.dataset, version=self.data_version)

    def add_annotation_model(self, model_name, schema_name, table_name, version=None):
        if model_name in self.models:
            print('Model name \'{}\' already exists'.format(model_name))
        if version is None:
            version = self.data_version
        self.models[model_name] = make_annotation_model(self.dataset, schema_name, table_name, version=version)

    def run_query(self, model_name, filter_obj=None):
        if filter_obj is None:
            data_qry = self.session.query(self.models[model_name])
        else:
            data_qry = self.session.query(self.models[model_name]).filter(filter_obj)
        data_df = pd.read_sql(data_qry.statement, self.session.bind, coerce_float=False)
        return data_df

    def run_query_raw(self, Model, filter_obj=None):
        if filter_obj is None:
            data_qry = self.session.query(Model)
        else:
            data_qry = self.session.query(Model).filter(filter_obj)
        data_df = pd.read_sql(data_qry.statement, self.session.bind, coerce_float=False)
        return data_df

    def query_synapses_by_id(self, model_name, pre_ids=None, post_ids=None):
        if pre_ids is None and post_ids is None:
            print('Please specify either pre or post ids')
            return

        if pre_ids is None:
            filter_obj = self.models[model_name].post_pt_root_id.in_(post_ids)
        elif post_ids is None:
            filter_obj = self.models[model_name].pre_pt_root_id.in_(pre_ids)
        else:
            filter_obj = and_(self.models[model_name].post_pt_root_id.in_(post_ids),
                              self.models[model_name].pre_pt_root_id.in_(pre_ids))

        data_df = self.run_query(model_name, filter_obj)
        data_df = fix_wkb_columns(data_df)
        return data_df

    def query_cell_type(self, model_name, oids=None, cell_type=None):
        if oids is None and cell_type is None:
            filter_obj = None
        elif oids is None:
            if isinstance(cell_type, str):
                cell_type = [cell_type]
            filter_obj = self.models[model_name].cell_type.in_(cell_type)
        elif cell_type is None:
            filter_obj = self.models[model_name].pt_root_id.in_(oids)
        else:
            filter_obj = and_(self.models[model_name].pt_root_id.in_(oids),
                              self.models[model_name].cell_type.in_(cell_type))

        data_df = self.run_query(model_name, filter_obj)
        data_df = fix_wkb_columns(data_df)
        return data_df

    def query_cell_id(self, model_name, oids=None, cell_ids=None):
        if isinstance(cell_ids, np.integer):
            cell_ids = [cell_ids]

        if oids is None and cell_ids is None:
            filter_obj = None
        elif oids is None:
            filter_obj = self.models[model_name].func_id.in_(cell_ids)
        elif cell_ids is None:
            filter_obj = self.models[model_name].pt_root_id.in_(oids)
        else:
            filter_obj = and_(self.models[model_name].pt_root_id.in_(oids),
                              self.models[model_name].func_id.in_(cell_ids))

        data_df = self.run_query(model_name, filter_obj)
        data_df = fix_wkb_columns(data_df)
        return data_df
