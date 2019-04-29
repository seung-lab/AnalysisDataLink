from .datalink import AnalysisDataLink

def annotation_version_mapping(table,
                               version_from,
                               version_to,
                               dataset_name,
                               sql_database_uri_base,
                               mapping_column='pt_root_id',
                               merge_column='id',
                               filter_in_dict={},
                               filter_notin_dict={},
                               filter_equal_dict={}):    
    df_merge = multiversion_merged_query(table,
                                         version_from,
                                         version_to,
                                         dataset_name,
                                         sql_database_uri_base,
                                         merge_column='id',
                                         filter_in_dict=filter_in_dict,
                                         filter_notin_dict=filter_notin_dict,
                                         filter_equal_dict=filter_equal_dict)
    
    mapping_column_from = mapping_column + '_{}'.format(version_from)
    mapping_column_to = mapping_column + '_{}'.format(version_to)

    if mapping_column_from not in df_merge.columns:
        raise ValueError('Mapping column ''{}'' not in annotation table'.format(mapping_column))
    
    return df_merge[[mapping_column_from, mapping_column_to]]


def multiversion_merged_query(table,
                              version_A,
                              version_B,
                              dataset_name,
                              sql_database_uri_base,
                              merge_column='id',
                              filter_in_dict={},
                              filter_notin_dict={},
                              filter_equal_dict={}):
    """
    Returns a merged dataframe of two materialization version queries.
    Columns other than annotation id get the suffix of the data version.
    Query filtering follows the structure of AnalysisDataLink.specific_query
    """

    df_A = _specific_version_query(dataset_name, sql_database_uri_base, version_A,
                               table, filter_in_dict, filter_notin_dict, filter_equal_dict)

    df_B = _specific_version_query(dataset_name, sql_database_uri_base, version_B,
                               table, filter_in_dict, filter_notin_dict, filter_equal_dict)
    
    return df_A.merge(df_B, on='id', how='outer',
                  suffixes=('_{}'.format(version_A), '_{}'.format(version_B)))


def _specific_version_query(dataset_name, sql_database_uri_base, data_version,
                            table, filter_in_dict={}, filter_notin_dict={}, filter_equal_dict={}):
    dl = AnalysisDataLink(dataset_name=dataset_name,
                          sqlalchemy_database_uri=sql_database_uri_base,
                          materialization_version=data_version,
                          verbose=False)
    df = dl.specific_query([table],
                           filter_in_dict=filter_in_dict,
                           filter_notin_dict=filter_notin_dict,
                           filter_equal_dict=filter_equal_dict)
    return df


