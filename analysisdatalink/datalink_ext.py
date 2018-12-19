import numpy as np

from analysisdatalink import datalink
from collections import defaultdict


class AnalysisDataLinkExt(datalink.AnalysisDataLink):
    def __init__(self, dataset_name, materialization_version,
                 sqlalchemy_database_uri=None, verbose=True,
                 annotation_endpoint=None):
        super().__init__(dataset_name, materialization_version,
                         sqlalchemy_database_uri, verbose=verbose,
                         annotation_endpoint=annotation_endpoint)

    def query_synapses(self, synapse_table, pre_ids=None, post_ids=None,
                       compartment_include_filter=None,
                       include_autapses=False,
                       compartment_table=None):
        """ Query synapses

        :param synapse_table: str
            table name without dataset prefix or version suffix
        :param pre_ids: None, list or np.ndarray
        :param post_ids: None, list or np.ndarray
        :param compartment_table: None, str
            defines compartment table -- has to be 'postsynapsecompartment'
        :param compartment_include_filter: list of str
        :param include_autapses: bool
        :param compartment_table: None, str
            DO NOT USE at the moment since there are no good compartment
            labels yet
        :return:
        """

        filter_in_dict = defaultdict(dict)
        filter_equal_dict = defaultdict(dict)
        if pre_ids is not None:
            filter_in_dict[synapse_table]["pre_pt_root_id"] = pre_ids
        if post_ids is not None:
            filter_in_dict[synapse_table]["post_pt_root_id"] = post_ids
        if not include_autapses:
            filter_equal_dict[synapse_table]["valid"] = True
        if compartment_table is not None:
            tables = [[synapse_table, "id"],
                      [compartment_table, "synapse_id"]]
            if compartment_include_filter is not None:
                filter_in_dict[compartment_table]['label'] = compartment_include_filter
        else:
            tables = [synapse_table]

        df = self.specific_query(tables,
                                 filter_in_dict=filter_in_dict,
                                 filter_equal_dict=filter_equal_dict)

        return df

    def query_cell_types(self, cell_type_table, cell_type_include_filter=None,
                         cell_type_exclude_filter=None, return_only_ids=False,
                         exclude_zero_root_ids=False):
        """ Query cell type tables

        :param cell_type_table: str
            table name without dataset prefix or version suffix
        :param cell_type_include_filter: list of str
        :param cell_type_exclude_filter: list of str
        :param return_only_ids: bool
        :param exclude_zero_root_ids: bool
        :return: pandas DataFrame or numpy array
        """

        filter_in_dict = defaultdict(dict)
        if cell_type_include_filter is not None:
            filter_in_dict[cell_type_table]["cell_type"] = cell_type_include_filter

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
            return np.array(df, dtype = np.uint64).squeeze()
        else:
            return df

    def query_cell_ids(self, cell_id_table, cell_id_filter=None,
                       cell_id_exclude_filter=None, return_only_ids=False,
                       exclude_zero_root_ids=False):
        """ Query cell ids

        :param cell_id_table: str
            table name without dataset prefix or version suffix
        :param cell_id_filter: list of uint64s
        :param cell_id_exclude_filter: list of uint64s
        :param return_only_ids: bool
        :param exclude_zero_root_ids:bool
        :return: pandas DataFrame or numpy array
        """
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
            return np.array(df, dtype=np.uint64).squeeze()
        else:
            return df

    def query_coreg(self, coreg_table, cell_id_filter=None,
                    cell_id_exclude_filter=None, return_only_mapping=False,
                    exclude_zero_root_ids=False):
        """ Queries coregistration

        :param coreg_table: str
            table name without dataset prefix or version suffix
        :param cell_id_filter: list of uint64s
        :param cell_id_exclude_filter: list of uint64s
        :param return_only_mapping: bool
            returns an array of [[root_id, f_id], ...]
        :param exclude_zero_root_ids: bool
            exclude zero root ids
        :return: pandas DataFrame or numpy array
        """
        filter_in_dict = defaultdict(dict)
        if cell_id_filter is not None:
            filter_in_dict[coreg_table]['func_id'] = cell_id_filter

        filter_notin_dict = defaultdict(dict)
        if cell_id_exclude_filter is not None:
            filter_notin_dict[coreg_table]['func_id'] = cell_id_exclude_filter
        if exclude_zero_root_ids is not None:
            filter_notin_dict[coreg_table]['pt_root_id'] = [0]

        if return_only_mapping:
            select_columns = ['pt_root_id', 'func_id']
        else:
            select_columns = None

        df = self.specific_query(tables=[coreg_table],
                                 filter_in_dict=filter_in_dict,
                                 filter_notin_dict=filter_notin_dict,
                                 select_columns=select_columns)

        if return_only_mapping:
            return np.array(df, dtype=np.uint64).squeeze()
        else:
            return df

