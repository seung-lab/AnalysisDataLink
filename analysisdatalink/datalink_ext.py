import numpy as np

from analysisdatalink import datalink
from collections import defaultdict

class AnalysisDataLinkExt(datalink.AnalysisDataLink):
    def __init__(self, dataset_name, materialization_version,
                 sqlalchemy_database_uri=None, verbose=True):
        super().__init__(dataset_name, materialization_version,
                         sqlalchemy_database_uri, verbose=verbose)

    def query_synapses(self, synapse_table, pre_ids=None, post_ids=None,
                       compartment_table="postsynapsecompartment"):
        """ Queries synapses

        :param synapse_table: str
            defines synapse table
        :param pre_ids: None, list or np.ndarray
        :param post_ids: None, list or np.ndarray
        :param compartment_table: None, str
            defines compartment table -- has to be 'postsynapsecompartment'
        :return:
        """

        filter_in_dict = {synapse_table: {}}
        if pre_ids is not None:
            filter_in_dict[synapse_table]["pre_pt_root_id"] = pre_ids
        if post_ids is not None:
            filter_in_dict[synapse_table]["post_pt_root_id"] = post_ids

        if compartment_table is not None:
            tables = [[synapse_table, "id"],
                      [compartment_table, "synapse_id"]]
        else:
            tables = [synapse_table]

        df = self.specific_query(tables, filter_in_dict=filter_in_dict)

        return df

    def query_cell_types(self, cell_type_table, cell_type_filter=None,
                         cell_type_exclude_filter=None, return_only_ids=False,
                         exclude_zero_root_ids=False):

        filter_in_dict = defaultdict(dict)
        if cell_type_filter is not None:
            filter_in_dict[cell_type_table]["cell_type"] = cell_type_filter

        filter_notin_dict = defaultdict(dict)
        if exclude_zero_root_ids:
            filter_notin_dict[cell_type_table]["pt_root_id"] = [0]
        if cell_type_exclude_filter is not None:
            filter_notin_dict[cell_type_table]['cell_type'] = cell_type_exclude_filter
        
        if return_only_ids:
            select_columns = ["pt_root_id"]
        else:
            select_columns = None

        df = self.specific_query(tables=[cell_type_table],
                                 filter_in_dict=filter_in_dict,
                                 filter_notin_dict=filter_notin_dict,
                                 select_columns=select_columns)

        if return_only_ids:
            return np.array(df, dtype = np.uint64).flatten()
        else:
            return df

    def query_cell_ids(self, cell_id_table, cell_id_filter=None,
                       cell_id_exclude_filter=None, return_only_ids=False,
                       exclude_zero_root_ids=False):
        filter_in_dict = defaultdict(dict)
        if cell_id_filter is not None:
            filter_in_dict[cell_id_table]['func_id'] = cell_id_filter

        filter_notin_dict = defaultdict(dict)
        if cell_id_exclude_filter is not None:
            filter_notin_dict[cell_id_table]['func_id'] = cell_id_exclude_filter
        if exclude_zero_root_ids is not None:
            filter_notin_dict[cell_id_table]['pt_root_id'] = [0]

        if return_only_ids:
            select_columns = ['pt_root_id']
        else:
            select_columns = None

        df = self.specific_query(tables=[cell_id_table],
                                 filter_in_dict=filter_in_dict,
                                 filter_notin_dict=filter_notin_dict,
                                 select_columns=select_columns)

        if return_only_ids:
            return np.array(df, dtype=np.uint64).flatten()
        else:
            return df