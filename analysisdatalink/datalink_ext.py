import numpy as np

from analysisdatalink import datalink
from collections import defaultdict


class AnalysisDataLinkExt(datalink.AnalysisDataLink):
    def __init__(self, dataset_name, materialization_version=None,
                 sqlalchemy_database_uri=None, verbose=True,
                 annotation_endpoint=None):
        super().__init__(dataset_name, materialization_version,
                         sqlalchemy_database_uri, verbose=verbose,
                         annotation_endpoint=annotation_endpoint)

    def query_synapses(self, synapse_table, pre_ids=None, post_ids=None,
                       compartment_include_filter=None,
                       include_autapses=False,
                       compartment_table=None, return_sql=False,
                       fix_wkb=True, fix_decimal=True, import_via_buffer=True,
                       n_threads=None):
        """Query a synapse table and return a dataframe
        
        Parameters
        ----------
        synapse_table : str
            Table name with a synapse schema
        pre_ids : collection of ints, optional
            Object ids for presynaptic neurons, by default None
        post_ids : collection of ints, optional
            Object ids for postsynaptic neurons, by default None
        compartment_include_filter : None, optional
            Not currently implemented. By default None
        include_autapses : bool, optional
            Include synapses whose pre- and post-synaptic objects are the same, by default False
        compartment_table : str, optional
            Not currently implemented. Would be a table name for synapse compartments. By default None
        return_sql : bool, optional
            Return the sqlalchemy query object instead of the data itself, by default False
        fix_wkb : bool, optional
            Convert wkb-formatted spatial location columns to numpy 3-vectors. Setting to False
            can be much faster, but spatial information is not easy to parse. These columns can be
            parsed after the fact with analysisdatalink.fix_wkb_column. Optional, by default True
        fix_decimal : bool, optional
            Convert Decimal columns to ints. Not used if import_via_buffer is True. By default True
        import_via_buffer : bool, optional
            Flag to determine whether to use a fast csv and tempfile based SQL import (if True) or the pandas
            native read_sql import (if False). If column formatting is odd, try setting to False. Optional, by default True.
        n_threads : int or None, optional
            Number of threads to use when parsing columns to convert wkb. Unused if fix_wkb is False. If set to 1,
            multiprocessing is not used and slower numpy vectorization is used instead.
            If None, uses the number of cpus available on the device. By default None
        
        Returns
        -------
        pandas.DataFrame
            DataFrame representation of the query results.
        """
        
        

        filter_in_dict = defaultdict(dict)
        filter_equal_dict = defaultdict(dict)
        if pre_ids is not None:
            filter_in_dict[synapse_table]["pre_pt_root_id"] = [int(pid) for pid in pre_ids]
        if post_ids is not None:
            filter_in_dict[synapse_table]["post_pt_root_id"] = [int(pid) for pid in post_ids]
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
                                 filter_equal_dict=filter_equal_dict,
                                 return_sql=return_sql,
                                 fix_wkb=fix_wkb, fix_decimal=fix_decimal,
                                 import_via_buffer=import_via_buffer,
                                 n_threads=n_threads)

        return df

    def query_cell_types(self, cell_type_table, cell_type_include_filter=None,
                         cell_type_exclude_filter=None, return_only_ids=False,
                         exclude_zero_root_ids=False, fix_wkb=True, fix_decimal=True,
                         return_sql=False, import_via_buffer=True, n_threads=None):
        """Query a synapse table and return a dataframe
        
        Parameters
        ----------
        cell_type_table : str
            Table name with a cell_type schema
        cell_type_include_filter : collection of str, optional
            Cell types to include
        cell_type_exclude_filter : collection of str, optional
            Cell types to exclude 
        return_only_ids : bool, optional
            Process to include only root ids matching the query.
        exclude_zero_root_ids : bool, optional
            Fitler out points with a null segmentation id. 
        return_sql : bool, optional
            Return the sqlalchemy query object instead of the data itself, by default False
        fix_wkb : bool, optional
            Convert wkb-formatted spatial location columns to numpy 3-vectors. Setting to False
            can be much faster, but spatial information is not easy to parse. These columns can be
            parsed after the fact with analysisdatalink.fix_wkb_column. Optional, by default True
        fix_decimal : bool, optional
            Convert Decimal columns to ints. Not used if import_via_buffer is True. By default True
        import_via_buffer : bool, optional
            Flag to determine whether to use a fast csv and tempfile based SQL import (if True) or the pandas
            native read_sql import (if False). If column formatting is odd, try setting to False. Optional, by default True.
        n_threads : int or None, optional
            Number of threads to use when parsing columns to convert wkb. Unused if fix_wkb is False. If set to 1,
            multiprocessing is not used and slower numpy vectorization is used instead.
            If None, uses the number of cpus available on the device. By default None
        
        Returns
        -------
        pandas.DataFrame
            DataFrame representation of the query results.
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
                                 select_columns=select_columns,
                                 fix_wkb=fix_wkb,
                                 fix_decimal=fix_decimal,
                                 return_sql=return_sql,
                                 import_via_buffer=import_via_buffer,
                                 n_threads=n_threads)

        if return_only_ids:
            return np.array(df, dtype = np.uint64).squeeze()
        else:
            return df

    def query_cell_ids(self, cell_id_table, cell_id_filter=None,
                       cell_id_exclude_filter=None, return_only_ids=False,
                       exclude_zero_root_ids=False, fix_wkb=True, fix_decimal=True,
                       return_sql=False, import_via_buffer=True, n_threads=None):

        """ Query cell id tables
        
        Parameters
        ----------
        cell_id_table : str
            Table name for a microns_functional_coregistration table
        cell_id_filter : list of uint64s, optional
            List of root ids to include. Default is None.
        cell_id_exclude_filter : list of uint64s, optional
            List of root ids to exclude. Default is None.
        return_only_ids : bool, optional
            Process to include only root ids matching the query. Default is False.
        exclude_zero_root_ids : bool, optional
            Fitler out points with a null segmentation id. Default is False.
        return_sql : bool, optional
            Return the sqlalchemy query object instead of the data itself, by default False
        fix_wkb : bool, optional
            Convert wkb-formatted spatial location columns to numpy 3-vectors. Setting to False
            can be much faster, but spatial information is not easy to parse. These columns can be
            parsed after the fact with analysisdatalink.fix_wkb_column. Optional, by default True
        fix_decimal : bool, optional
            Convert Decimal columns to ints. Not used if import_via_buffer is True. By default True
        import_via_buffer : bool, optional
            Flag to determine whether to use a fast csv and tempfile based SQL import (if True) or the pandas
            native read_sql import (if False). If column formatting is odd, try setting to False. Optional, by default True.
        n_threads : int or None, optional
            Number of threads to use when parsing columns to convert wkb. Unused if fix_wkb is False. If set to 1,
            multiprocessing is not used and slower numpy vectorization is used instead.
            If None, uses the number of cpus available on the device. By default None
        
        Returns
        -------
        pandas.DataFrame
            DataFrame representation of the query results.
        """
        filter_in_dict = defaultdict(dict)
        if cell_id_filter is not None:
            filter_in_dict[cell_id_table]['func_id'] = [int(pid) for pid in cell_id_filter]

        filter_notin_dict = defaultdict(dict)
        if cell_id_exclude_filter is not None:
            filter_notin_dict[cell_id_table]['func_id'] = [int(pid) for pid in cell_id_exclude_filter]
        if exclude_zero_root_ids is not None:
            filter_notin_dict[cell_id_table]['pt_root_id'] = [0]

        if return_only_ids:
            select_columns = ['pt_root_id']
        else:
            select_columns = None

        df = self.specific_query(tables=[cell_id_table],
                                 filter_in_dict=filter_in_dict,
                                 filter_notin_dict=filter_notin_dict,
                                 select_columns=select_columns,
                                 fix_wkb=fix_wkb,
                                 fix_decimal=fix_decimal,
                                 return_sql=return_sql,
                                 import_via_buffer=import_via_buffer, n_threads=n_threads)

        if return_only_ids:
            return np.array(df, dtype=np.uint64).squeeze()
        else:
            return df

    def query_coreg(self, coreg_table, cell_id_filter=None,
                    cell_id_exclude_filter=None, return_only_mapping=False,
                    exclude_zero_root_ids=False,
                    fix_wkb=True, fix_decimal=True,
                    return_sql=False, import_via_buffer=True, n_threads=None):
        """ Query cell id tables
        
        Parameters
        ----------
        coreg_table : str
            Table name for a microns_functional_coregistration table
        cell_id_filter : list of uint64s, optional
            List of root ids to include. Default is None.
        cell_id_exclude_filter : list of uint64s, optional
            List of root ids to exclude. Default is None.
        return_only_ids : bool, optional
            Process to include only root ids matching the query. Default is False.
        exclude_zero_root_ids : bool, optional
            Fitler out points with a null segmentation id. Default is False.
        return_sql : bool, optional
            Return the sqlalchemy query object instead of the data itself, by default False
        fix_wkb : bool, optional
            Convert wkb-formatted spatial location columns to numpy 3-vectors. Setting to False
            can be much faster, but spatial information is not easy to parse. These columns can be
            parsed after the fact with analysisdatalink.fix_wkb_column. Optional, by default True
        fix_decimal : bool, optional
            Convert Decimal columns to ints. Not used if import_via_buffer is True. By default True
        import_via_buffer : bool, optional
            Flag to determine whether to use a fast csv and tempfile based SQL import (if True) or the pandas
            native read_sql import (if False). If column formatting is odd, try setting to False. Optional, by default True.
        n_threads : int or None, optional
            Number of threads to use when parsing columns to convert wkb. Unused if fix_wkb is False. If set to 1,
            multiprocessing is not used and slower numpy vectorization is used instead.
            If None, uses the number of cpus available on the device. By default None
        
        Returns
        -------
        pandas.DataFrame
            DataFrame representation of the query results.
        """
        filter_in_dict = defaultdict(dict)
        if cell_id_filter is not None:
            filter_in_dict[coreg_table]['func_id'] = [int(pid) for pid in cell_id_filter]

        filter_notin_dict = defaultdict(dict)
        if cell_id_exclude_filter is not None:
            filter_notin_dict[coreg_table]['func_id'] = [int(pid) for pid in cell_id_exclude_filter]
        if exclude_zero_root_ids is not None:
            filter_notin_dict[coreg_table]['pt_root_id'] = [0]

        if return_only_mapping:
            select_columns = ['pt_root_id', 'func_id']
        else:
            select_columns = None

        df = self.specific_query(tables=[coreg_table],
                                 filter_in_dict=filter_in_dict,
                                 filter_notin_dict=filter_notin_dict,
                                 select_columns=select_columns,
                                 fix_wkb=fix_wkb,
                                 fix_decimal=fix_decimal,
                                 return_sql=return_sql,
                                 import_via_buffer=import_via_buffer, n_threads=n_threads)

        if return_only_mapping:
            return np.array(df, dtype=np.uint64).squeeze()
        else:
            return df

