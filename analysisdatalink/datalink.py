import numpy as np

import sqlalchemy
from analysisdatalink import datalink_base

class AnalysisDataLink(datalink_base.AnalysisDataLinkBase):
    """ Base analysis object

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
                 annotation_endpoint=None, convert_to_nm=None):
        super().__init__(dataset_name, materialization_version,
                         sqlalchemy_database_uri, verbose=verbose,
                         annotation_endpoint=annotation_endpoint, 
                         convert_to_nm=convert_to_nm)

    def specific_query(self, tables, filter_in_dict={}, filter_notin_dict={},
                       filter_equal_dict = {},
                       select_columns=None):
        """ Allows a more narrow query without requiring knowledge about the
            underlying data structures 

        Parameters
        ----------
        tables: list of lists
            standard: list of one entry: table_name of table that one wants to
                      query
            join: list of two lists: first entries are table names, second
                                     entries are the columns used for the join
        filter_in_dict: dict of dicts
            outer layer: keys are table names
            inner layer: keys are column names, values are entries to filter by
        filter_notin_dict: dict of dicts
            inverse to filter_in_dict
        select_columns: list of str

        Returns
        -------
        sqlalchemy query object:
        """
        tables = [[table] if not isinstance(table, list) else table
                  for table in tables]
        query_args = [self.model(table[0]) for table in tables]

        if len(tables) == 2:
            join_args = (self.model(tables[1][0]),
                         self.model(tables[1][0]).__dict__[tables[1][1]] ==
                         self.model(tables[0][0]).__dict__[tables[0][1]])
        elif len(tables) > 2:
            raise Exception("Currently, only single joins are supported")
        else:
            join_args = None

        filter_args = []

        for filter_table, filter_table_dict in filter_in_dict.items():
            for column_name in filter_table_dict.keys():
                filter_values = filter_table_dict[column_name]
                filter_values = np.array(filter_values, dtype="O")

                filter_args.append((self.model(filter_table).__dict__[column_name].
                                    in_(filter_values), ))

        for filter_table, filter_table_dict in filter_notin_dict.items():
            for column_name in filter_table_dict.keys():
                filter_values = filter_table_dict[column_name]
                filter_values = np.array(filter_values, dtype="O")

                filter_args.append((sqlalchemy.not_(self.model(filter_table).__dict__[column_name].
                                                    in_(filter_values)), ))
        
        for filter_table, filter_table_dict in filter_equal_dict.items():
            for column_name in filter_table_dict.keys():
                filter_value = filter_table_dict[column_name]
                filter_args.append((self.model(filter_table).__dict__[column_name]==filter_value, ))

        return self._query(query_args=query_args, filter_args=filter_args,
                           join_args=join_args, select_columns=select_columns)


