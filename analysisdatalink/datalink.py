from collections import defaultdict
from shapely import wkb
import geopandas as gpd
from emannotationschemas.base import flatten_dict
from emannotationschemas import get_schema
from emannotationschemas.models import make_annotation_model, make_cell_segment_model
from emannotationschemas.mesh_models import make_post_synaptic_compartment_model
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
from collections import defaultdict, Iterable

def wkb_to_numpy(wkb):
    shp=to_shape(wkb)
    return np.array([shp.xy[0][0],shp.xy[1][0], shp.z], dtype=np.int)

def fix_wkb_columns(df):
    if len(df) > 0:
        for colname in df.columns:
            if isinstance(df.at[0,colname], WKBElement):
                df[colname] = df[colname].apply(wkb_to_numpy)
    return df

def pre_post_dict():
    return {'post': [], 'pre': []}

class AnnotationDataLink():
    def __init__(self, dataset, version, database_uri=None, session=None):
        self.database_uri = database_uri
        self.data_version = version
        self.dataset = dataset
        self.session = session
        self.models = {}

    def add_annotation_model(self, schema_name, table_name, model_name=None, version=None):
        """Loads database model for an annotation schema

        Args:
            schema_name: Name of the schema for the table from emannotationschemas 
            table_name: Table name for the database
            model_name: Reference name for model (optional, defaults to same as table_name)
            version: Materialization version (Optional)
        """
        if model_name in self.models:
            print('Model name \'{}\' already exists'.format(model_name))
        if version is None:
            version = self.data_version
        self.models[model_name] = make_annotation_model(self.dataset, schema_name, table_name, version=version)

    def make_query(self, query_args, join_args=None, filter_args=None):
        """Constructs a query object with selects, joins, and filters

        Args:
            query_args: Iterable of objects to query
            join_args: Iterable of objects to set as a join (optional)
            filter_args: Iterable of iterables

        Returns:
            SQLAchemy query object
        """
        query = self.session.query(*query_args)
        if join_args is not None:
            query = query.join(*join_args)
        if filter_args is not None:
            for f in filter_args:
                query = query.filter(*f)
        return query

    def dataframe_from_query(self, query, fix_wkb=True):
        """ Query the database and make a dataframe out of the results

        Args:
            query: SQLAlchemy query object
            fix_wkb: Boolean to turn wkb objects into numpy arrays (optional, default is True)
        
        Returns:
            Dataframe with query results
        """
        df = pd.read_sql(query.statement, self.session.bind, coerce_float=False)
        if fix_wkb:
            df = fix_wkb_columns(df)
        return df


class CellTypeDataLink(AnnotationDataLink):
    def __init__(self, dataset, version, session):
        super(CellTypeDataLink, self).__init__(dataset, version, session=session)

    def add_model(self, table_name, schema_name='cell_type_local', model_name=None, version=None):
        self.add_annotation_model(schema_name, table_name, model_name=model_name, version=version)

    def query_cell_type(self, model_name, oids=None, cell_types=None):
        query_args = (self.models[model_name],)
        filter_args = []
        if oids is not None:
            filter_args.append( (self.models[model_name].pt_root_ids.in_(oids)) )
        if cell_types is not None:
            if isinstance(cell_types, str):
                filter_args.append((self.models[model_name].cell_type == cell_types,))
            elif isinstance(cell_types, Iterable):
                filter_args.append((self.models[model_name].cell_type.in_(cell_types),))
        qry = self.make_query(query_args=query_args, filter_args=filter_args)
        data_df = self.dataframe_from_query(qry)
        return data_df


class CellIdDataLink(AnnotationDataLink):
    def __init__(self, dataset, version, session):
        super(CellIdDataLink, self).__init__(dataset, version, session=session)

    def add_model(self, table_name, schema_name='microns_func_coreg', model_name=None, version=None):
        self.add_annotation_model(schema_name, table_name, model_name=model_name, version=version)

    def query_cell_ids(self, model_name, oids=None, cell_ids=None):
        query_args = (self.models[model_name],)
        filter_args = []
        if oids is not None:
            filter_args.append( (self.models[model_name].pt_root_ids.in_(oids)) )
        if cell_ids is not None:
            if isinstance(cell_ids, int):
                filter_args.append((self.models[model_name].func_id == cell_ids,))
            elif isinstance(cell_ids, Iterable):
                filter_args.append((self.models[model_name].func_id.in_(cell_ids),))

        qry = self.make_query(query_args=query_args, filter_args=filter_args)
        data_df = self.dataframe_from_query(qry)
        return data_df


class SynapseDataLink(AnnotationDataLink):
    def __init__(self, dataset, version, session):
        super(SynapseDataLink, self).__init__(dataset, version, session=session)
        self.compartment_models = defaultdict(pre_post_dict)

    def add_synapse_model(self, table_name, schema_name='synapse', model_name=None, version=None, set_compartments=False):
        self.add_annotation_model(schema_name, table_name, model_name=model_name, version=version)
        if set_compartments:
            if model_name is None:
                model_name = table_name
            try:
                self.add_postsynaptic_compartments(table_name, synapse_model_name=model_name, model_name=None)
            except:
                print('Could not add postsynaptic compartments')

    def add_postsynaptic_compartments(self, synapse_table, synapse_model_name=None, model_name=None):
        if (synapse_model_name is None) and (len(self.models) == 1):
            synapse_model_name = list(self.models.keys()).pop()            
        if model_name is None:
            model_name = '{}_post_compartments'.format(synapse_model_name)
        if model_name in self.models:
            raise ValueError('Model with that name already exists')    
        self.models[model_name] = make_post_synaptic_compartment_model(self.dataset, synapse_table, self.data_version)
        self.compartment_models[synapse_model_name]['post'].append( model_name )

    def query_synapses(self,
                       synapse_model_name=None,
                       pre_ids=None,
                       post_ids=None,
                       post_compartments=None,
                       return_post_compartments=True,
                       post_compartment_model_name=None):
        if synapse_model_name is None:
            all_models = set(self.models.keys())
            comp_models = []
            for syn_m in self.compartment_models.values():
                for m in syn_m['post']:
                    comp_models.append(m)
            comp_models = set(comp_models)
            syn_models = all_models.difference(comp_models)
            if len(syn_models)==1:
                synapse_model_name = syn_models.pop()
            else:
                print('Please specify which synapse model to use')
                return

        if pre_ids is None and post_ids is None and post_compartments is None:
            print('\tNo filters specified. This could take a while...')
        if post_compartments is not None:
            return_post_compartments = True
        if return_post_compartments:
            if (post_compartment_model_name is None) and (len(self.compartment_models[synapse_model_name]['post']) == 1):
                post_compartment_model_name = self.compartment_models[synapse_model_name]['post'][0]

        if return_post_compartments:
            query_args = (self.models[post_compartment_model_name], self.models[synapse_model_name])
            join_args = (self.models[synapse_model_name], self.models[synapse_model_name].id == self.models[post_compartment_model_name].synapse_id)
        else:
            query_args = (self.models[synapse_model_name], )
            join_args = None

        filter_args = []
        if pre_ids is not None:
            filter_args.append( (self.models[synapse_model_name].pre_pt_root_id.in_(pre_ids),) )
        if post_ids is not None:
            filter_args.append( (self.models[synapse_model_name].post_pt_root_id.in_(post_ids),) )
        if post_compartments is not None:
            filter_args.append( (self.models[post_compartment_model_name].label.in_(post_compartments),))

        qry = self.make_query(query_args, join_args, filter_args)
        data_df = self.dataframe_from_query(qry)

        return data_df


class DataLink(AnnotationDataLink):
    def __init__(self, dataset, version, database_uri=None):
        super(DataLink, self).__init__(dataset, version, database_uri=database_uri)        
        if database_uri is None:
            database_uri = os.getenv('DATABASE_URI')
            assert database_uri is not None
        self.database_uri = database_uri

        engine = create_engine(self.database_uri, echo=False)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        self.dataset = dataset
        self.data_version = version        
        make_cell_segment_model(self.dataset, version=self.data_version)

        self.synapses = SynapseDataLink(self.dataset, self.data_version, self.session)
        self.cell_types = CellTypeDataLink(self.dataset, self.data_version, self.session)
        self.cell_ids = CellIdDataLink(self.dataset, self.data_version, self.session)
        