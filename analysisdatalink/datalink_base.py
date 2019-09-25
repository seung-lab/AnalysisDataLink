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
import re
import json
import requests
from decimal import Decimal
import analysisdatalink

def build_database_uri(base_uri, dataset_name, materialization_version):
    """Builds database name out of parameters"""
    qry_pg = re.search('/postgres$', base_uri)
    # Hack to convert old ids. Should be dropped when the new system is rolled out.
    if qry_pg is not None:
        base_uri = base_uri[0:qry_pg.start()]
    database_suffix = em_models.format_database_name(dataset_name, materialization_version)
    return base_uri + '/' + database_suffix


def wkb_to_numpy(wkb, convert_to_nm = None):
    """ Fixes single geometry column """
    shp=to_shape(wkb)
    xyz_voxel = np.array([shp.xy[0][0],shp.xy[1][0], shp.z], dtype=np.int)
    if convert_to_nm is not None:
        return xyz_voxel*convert_to_nm
    else:
        return xyz_voxel


def fix_wkb_columns(df, convert_to_nm = None):
    """ Fixes geometry columns 
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe of results to fix columns with WKB postgis columns
    convert_to_nm : len(3) iterable
        the x,y,z conversion factor to convert the coordinate system to nm
        default None leaves it as voxel resolution
    """
    if convert_to_nm is not None:
        if (len(convert_to_nm) != 3):
            raise ValueError("convert_to_nm must be length 3")
    if len(df) > 0:
        for colname in df.columns:
            if isinstance(df.at[0,colname], WKBElement):
                df[colname] = df[colname].apply(wkb_to_numpy, convert_to_nm)
    return df

def fix_decimal_columns(df):
    if len(df) > 0:
        is_decimal = np.vectorize(lambda x: isinstance(x, Decimal))
        is_integer_col = np.vectorize(lambda x: float(x).is_integer())
        for col in df.columns:
            if np.all(is_decimal(df[col])):
                if np.all(is_integer_col(df[col])):
                    df[col] = df[col].astype(np.uint64)
                else:
                    df[col] = df[col].apply(np.float)
    return df

def get_materialization_versions(dataset_name, materialization_endpoint=None):
    """ Gets materialization versions with timestamps """
    if materialization_endpoint is None:
        materialization_endpoint = analysisdatalink.materialization_endpoint
    
    url = '{}/api/dataset/{}'.format(materialization_endpoint, dataset_name)
    r = requests.get(url)
    assert r.status_code == 200
    versions = {d['version']:d['time_stamp'] for d in r.json() if d['valid']}
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
    """ Analysis query interface base class for dynamicannotationframework

    Parameters
    ----------
    dataset_name : str 
        name of dataset to query
    materialization_version : int or None
        what version to query (defaults to most recent), 
        use materialization service to query what is available
    sqlalchemy_database_uri : str
        sqlalchemy connection uri with db type, username, password, ip, and port
    verbose : bool
        whether to print debugging info
    annotation_endpoint : str or None
        url of annotation endpoint to retrieve metadata about tables
        (default to www.dynamicannotationframework.com/annotation)
    convert_to_nm : len(3) iterable or None
        conversion factor to convert spatial units into nm
        from what is stored in db as voxels
    """

    def __init__(self, dataset_name, materialization_version=None,
                 sqlalchemy_database_uri=None, verbose=True,
                 annotation_endpoint=None, convert_to_nm = None):

        if sqlalchemy_database_uri is None:
            sqlalchemy_database_uri = os.getenv('DATABASE_URI')
            assert sqlalchemy_database_uri is not None
        
        self._base_engine = create_engine(sqlalchemy_database_uri, echo=verbose)
        self._base_sqlalchemy_session = sessionmaker(bind=self._base_engine)
        self._this_sqlalchemy_base_session = None
        if materialization_version is None:
            version_query=self.this_sqlalchemy_base_session.query(em_models.AnalysisVersion)
            version_query=version_query.filter(em_models.AnalysisVersion.dataset == dataset_name)
            versions=version_query.filter(em_models.AnalysisVersion.valid == True).all()
            version_d = {v.version:v.time_stamp for v in versions}
            #version_d = get_materialization_versions(dataset_name=dataset_name)
            versions = np.array([v for v in version_d.keys()], dtype=np.uint32)
            materialization_version = int(np.max(versions))

        sqlalchemy_database_uri = build_database_uri(sqlalchemy_database_uri, dataset_name, materialization_version)
        if verbose == True:
            print('Using URI: {}'.format(sqlalchemy_database_uri))

        self._dataset_name = dataset_name
        self.convert_to_nm = convert_to_nm

            
        self._materialization_version = materialization_version
        self._annotation_endpoint = annotation_endpoint
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
    def this_sqlalchemy_base_session(self):
        if self._this_sqlalchemy_base_session is None:
            self._this_sqlalchemy_base_session  = self._base_sqlalchemy_session()
        return self._this_sqlalchemy_base_session 

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
        av = self.this_sqlalchemy_base_session.query(em_models.AnalysisVersion)\
            .filter(em_models.AnalysisVersion.version == self._materialization_version).first()

        base_query=self.this_sqlalchemy_base_session.query(em_models.AnalysisTable)
        base_query=base_query.filter(em_models.AnalysisTable.analysisversion == av)
        base_query=base_query.filter(em_models.AnalysisTable.tablename == table_name)

        schema = base_query.first()
        schema_name = schema.schema
        if schema_name is None:
            schema_name =  get_annotation_info(self.dataset_name, table_name,
                                       self._annotation_endpoint)

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

    def _execute_query(self, query, fix_wkb=True, fix_decimal=True, index_col=None, convert_to_nm=None):
        """ Query the database and make a dataframe out of the results

        Parameters
        ----------
        query: SQLAlchemy query object
        fix_wkb: bool
            whether to turn wkb objects into numpy arrays (optional, default is True)
        index_col: None or str
            column to use as index of dataframe
        fix_decimal : bool
            whether to convert Decimal columns to uint64 or floats depending on content

        Returns:
            Dataframe with query results
        """
        df = pd.read_sql(query.statement, self.sqlalchemy_engine,
                         coerce_float=False, index_col=index_col)
        if fix_wkb:
            df = fix_wkb_columns(df, convert_to_nm=self.convert_to_nm)

        if fix_decimal:
            df = fix_decimal_columns(df)

        return df

    def _query(self, query_args, join_args=None, filter_args=None,
               select_columns=None, fix_wkb=True, index_col=None):
        """ Wraps make_query and execute_query in one function

        Parameters
        ----------
        query_args:
        join_args:
        filter_args:
        select_columns:
        fix_wkb: bool
        index_col: str or None



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
