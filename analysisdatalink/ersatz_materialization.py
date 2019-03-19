import tqdm
import pandas as pd
import numpy as np
import cloudvolume
import datetime

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from annotationframeworkclient.pychunkedgraph import PychunkedgraphClient
from emannotationschemas import models as em_models


def lookup_supervoxels(xyzs, cv_path, segmentation_scaling=[2,2,1]):
    '''
    Lookup supervoxel ids from a np array of points
    '''
    sv_ids = []
    xyzs = xyzs / np.array(segmentation_scaling)
    cv = cloudvolume.CloudVolumeFactory(cloudurl=cv_path,
                                        map_gs_to_https=True,
                                        progress=False)
    for xyz in xyzs:
        sv = cv._cv[xyz[0], xyz[1], xyz[2]]
        sv_ids.append(int(sv.flatten()))
    return sv_ids


def get_materialization_timestamp(materialization_version, sql_database_uri):
    '''
    Query the database for a materialization version and get the utc time stamp.
    '''
    Session = sessionmaker()
    engine = create_engine(sql_database_uri)
    Session.configure(bind=engine)
    session = Session()

    query = session.query(em_models.AnalysisVersion).filter(em_models.AnalysisVersion.version==materialization_version)
    materialization_dt = query.value(column='time_stamp')
    session.close()

    materialization_time_utc = materialization_dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return materialization_time_utc


def lookup_root_ids(supervoxel_ids, cv_path, materialization_time):
    '''
    Look up root ids from a list of supervoxels
    '''
    pcg_client = PychunkedgraphClient(cv_path=cv_path)
    root_ids = []
    for sv_id in supervoxel_ids:
        try:
            oid = pcg_client.get_root(sv_id, timestamp=materialization_time)
            root_ids.append(int(oid))
        except:
            root_ids.append(-1)
    return root_ids


def ersatz_point_query(xyzs,
                       materialization_version,
                       sql_database_uri,
                       cv_path,
                       segmentation_scaling=[2,2,1],
                       additional_columns={}):
    '''
    Given a set of points, returns a dataframe formatted like a database query.
    Aligned to a particular materialization version.
    :param xyzs: Nx3 array of point positions in supervoxels.
    :param materialization_version: Int, version in the materialized database.
    :param sql_database_uri: String, materialization database URI.
    :param cv_path: String, cloudvolume path.
    :param segmentation_scaling: 3 element array, Gives xyz scaling between segmentation and imagery for CloudVolume
    :param additional_columns: Dict with keys as strings and N-length array-likes as values. Extra columns in dataframe.
    '''
    sv_ids = lookup_supervoxels(xyzs, cv_path, segmentation_scaling=segmentation_scaling)
    materialization_time = get_materialization_timestamp(materialization_version, sql_database_uri)
    root_ids = lookup_root_ids(sv_ids, cv_path, materialization_time)
    pt_dict = {'pt_position': [list(xyz) for xyz in xyzs],
               'pt_supervoxel_id': sv_ids,
               'pt_root_id': root_ids}
    dat_dict = {**pt_dict, **additional_columns}
    df = pd.DataFrame(dat_dict)
    df[['pt_position', 'pt_supervoxel_id', 'pt_root_id']] = df[['pt_position', 'pt_supervoxel_id', 'pt_root_id']].astype('O')
    return df
