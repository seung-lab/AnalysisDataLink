from emannotationschemas import models as em_models, \
    mesh_models as em_mesh_models
from geoalchemy2.shape import to_shape, from_shape
from geoalchemy2.elements import WKBElement
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time
import numpy as np
import pandas as pd
import os
import json
import requests

import analysisdatalink


def wkb_to_numpy(wkb):
    """ Fixes single geometry column """
    shp=to_shape(wkb)
    return np.array([shp.xy[0][0],shp.xy[1][0], shp.z], dtype=np.int)


def fix_wkb_columns(df):
    """ Fixes geometry columns """
    if len(df) > 0:
        for colname in df.columns:
            if isinstance(df.at[0,colname], WKBElement):
                df[colname] = df[colname].apply(wkb_to_numpy)
    return df


def get_materialization_versions(dataset_name, materialization_endpoint=None):
    """ Gets materialization versions with timestamps """
    if materialization_endpoint is None:
        materialization_endpoint = analysisdatalink.materialization_endpoint

    url = '{}/api/dataset/{}'.format(materialization_endpoint, dataset_name)
    r = requests.get(url)
    assert r.status_code == 200
    versions = {d['version']:d['time_stamp'] for d in r.json()}
    return versions


def get_annotation_info(dataset_name, table_name, annotation_endpoint=None):
    """ Reads annotation info from annotation engine endpoint """

    if table_name is "postsynapsecompartment":
        return {"schema_name": "PostSynapseCompartment"}

    if annotation_endpoint is None:
        annotation_endpoint = analysisdatalink.annotation_endpoint

    url = "%s/dataset/%s/%s" % (annotation_endpoint, dataset_name, table_name)
    r = requests.get(url)

    assert r.status_code == 200

    return json.loads(r.content)


class AnalysisDataLinkBase(object):
    def __init__(self, dataset_name, materialization_version,
                 sqlalchemy_database_uri=None, verbose=True):

        if sqlalchemy_database_uri is None:
            sqlalchemy_database_uri = os.getenv('DATABASE_URI')
            assert sqlalchemy_database_uri is not None

        self._dataset_name = dataset_name
        self._materialization_version = materialization_version
        self._sqlalchemy_database_uri = sqlalchemy_database_uri
        self._models = {}

        self._models["cellsegment"] = em_models.make_cell_segment_model(
            dataset_name, version=self.materialization_version)

        self._sqlalchemy_engine = create_engine(sqlalchemy_database_uri,
                                                echo=verbose)
        em_models.Base.metadata.create_all(self.sqlalchemy_engine)

        self._sqlalchemy_session = sessionmaker(bind=self.sqlalchemy_engine)

        self._this_sqlalchemy_session = None

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def materialization_version(self):
        return self._materialization_version

    @property
    def sqlalchemy_database_uri(self):
        return self._sqlalchemy_database_uri

    @property
    def sqlalchemy_engine(self):
        return self._sqlalchemy_engine

    @property
    def sqlalchemy_session(self):
        return self._sqlalchemy_session

    @property
    def this_sqlalchemy_session(self):
        if self._this_sqlalchemy_session is None:
            self._this_sqlalchemy_session = self.sqlalchemy_session()
        return self._this_sqlalchemy_session

    def model(self, table_name, is_synapse_compartment=False):
        """ Returns annotation model for table

        :param table_name: str
        :return: em annotation model
        """
        if not is_synapse_compartment and self._add_annotation_model(table_name):
            return self._models[table_name]
        # elif is_synapse_compartment and self._add_synapse_compartment_model():
        #     return self.
        else:
            raise Exception("Could not make annotation model")

    def _add_synapse_compartment_model(self, synapse_table_name,
                                       table_name="postsynapsecompartment"):
        if table_name in self._models:
            print('Model name \'{}\' already exists'.format(table_name))
            return True

        try:
            self._models[table_name] = em_mesh_models.make_post_synaptic_compartment_model(
                dataset=self.dataset_name, synapse_table=synapse_table_name,
                version=self.materialization_version)
            return True
        except Exception as e:
            print(e)
            return False

    def _add_annotation_model(self, table_name):
        """ Loads database model for an annotation schema

        Args:
            table_name: Table name for the database
        """
        if table_name in self._models:
            return True

        schema_name = get_annotation_info(self.dataset_name,
                                          table_name)["schema_name"]
        try:
            self._models[table_name] = em_models.make_annotation_model(
                dataset=self.dataset_name, annotation_type=schema_name,
                table_name=table_name, version=self.materialization_version)

            if schema_name == 'synapse':
                self._add_synapse_compartment_model(synapse_table_name=table_name)
            
            return True
        except Exception as e:
            print(e)
            return False


    def _make_query(self, query_args, join_args=None, filter_args=None,
                    select_columns=None):
        """Constructs a query object with selects, joins, and filters

        Args:
            query_args: Iterable of objects to query
            join_args: Iterable of objects to set as a join (optional)
            filter_args: Iterable of iterables
            select_columns: None or Iterable of str

        Returns:
            SQLAchemy query object
        """

        query = self.this_sqlalchemy_session.query(*query_args)

        if join_args is not None:
            query = query.join(*join_args, full=True)

        if filter_args is not None:
            for f in filter_args:
                query = query.filter(*f)

        if select_columns is not None:
            query = query.with_entities(*select_columns)

        return query

    def _execute_query(self, query, fix_wkb=True, index_col=None):
        """ Query the database and make a dataframe out of the results

        Args:
            query: SQLAlchemy query object
            fix_wkb: Boolean to turn wkb objects into numpy arrays (optional, default is True)
            index_col: None or str

        Returns:
            Dataframe with query results
        """
        df = pd.read_sql(query.statement, self.sqlalchemy_engine,
                         coerce_float=False, index_col=index_col)
        if fix_wkb:
            df = fix_wkb_columns(df)

        return df


    def _query(self, query_args, join_args=None, filter_args=None,
               select_columns=None, fix_wkb=True, index_col=None):
        """ Wraps make_query and execute_query in one function

        :param query_args:
        :param join_args:
        :param filter_args:
        :param select_columns:
        :param fix_wkb:
        :param index_col:
        :return:
        """

        query = self._make_query(query_args=query_args,
                                 join_args=join_args,
                                 filter_args=filter_args,
                                 select_columns=select_columns)

        df = self._execute_query(query=query, fix_wkb=fix_wkb,
                                 index_col=index_col)

        return df