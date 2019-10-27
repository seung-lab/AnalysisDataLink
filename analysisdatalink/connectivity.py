import numpy as np
import pandas as pd


def edgelist_from_synapse_df(syn_df,
                             pre_column='pre_pt_root_id',
                             post_column='post_pt_root_id',
                             weight_column='size',
                             agg='count'):
    """Compute a list of pre to post edges from a synapse-query-style dataframe. Defaults to counting  synapses.

    Parameters
    ----------
    syn_df : pandas.DataFrame 
        DataFrame with columns for pre id, post id, and at least one additional column to use as weight. 
    pre_column : str, optional
        Column name of the presynaptic ids, by default 'pre_pt_root_id'
    post_column : str, optional
        Column name of the postsynaptic ids, by default 'pre_pt_root_id'
    weight_column : str, optional
        Column name for values to be aggregated, by default 'size''
    agg : str, optional
        Argument for the pandas groupby aggregation function, by default 'count'. Set to `sum` for using net synapse size instead.

    Returns
    -------
    pandas.DataFrame
        DataFrame with pre, post, and weight columns and a row for each edge in graph.
    """
    syn_gb = syn_df.groupby([pre_column, post_column])
    edge_list_wide = syn_gb.agg(agg).reset_index()    
    edge_list = edge_list_wide[[pre_column, post_column, weight_column]].rename(columns={weight_column: 'weight'})
    return edge_list

def adjacency_matrix_from_edgelist(edgelist,
                                   pre_column='pre_pt_root_id',
                                   post_column='post_pt_root_id',
                                   weight_column='weight',
                                   id_list=None):
    """Build an adjacency matrix dataframe from an edgelist dataframe.
    
    Parameters
    ----------
    edgelist : pandas.DataFrame
        Dataframe with pre id, post id, and weight columns and a row for each edge.
    pre_column : str, optional
        Name of the presynaptic column, by default 'pre_pt_root_id'
    post_column : str, optional
        Name of the postsynaptic column, by default 'post_pt_root_id'
    weight_column : str, optional
        Name of the weight column, by default 'weight'
    id_list : Collection, optional
        Collection of ids to use for the adjacency matrix indices, preserving order.
        If id_list is None, it uses exactly the ids in the edgelist.
        If id_list includes ids not in the edgelist, they become rows/columns with zeros.
        If id_list does not include ids that are in the edgelist, those edges are ignored.
        By default None
    
    Returns
    -------
    pandas.DataFrame
        Square dataframe with postsynaptic ids as index, presynaptic ids as columns, and
        values correspond to the weight column with 0s filled for unshown data.
    """
    el = edgelist.copy()

    if id_list is None:
        all_ids = np.unique(np.concatenate([edgelist[pre_column], edgelist[post_column]]))
    else:
        all_ids = id_list
        in_all_ids_pre = np.isin(edgelist[pre_column], all_ids)
        in_all_ids_post = np.isin(edgelist[post_column], all_ids)
        el = el[(in_all_ids_post) & (in_all_ids_pre)]
    
    el = edgelist.copy()
    pre_to_add = all_ids[~np.isin(all_ids, edgelist[pre_column])]
    concat_tuple = [el]
    if len(pre_to_add)>0:
        pre_add_df = pd.DataFrame(data={pre_column:pre_to_add,
                                        post_column:edgelist[post_column].iloc[0],
                                        weight_column:0})
        concat_tuple.append(pre_add_df)
    
    post_to_add = all_ids[~np.isin(all_ids, edgelist[post_column])]
    if len(post_to_add)>0:
        post_add_df = pd.DataFrame(data={pre_column:edgelist[pre_column].iloc[0],
                                        post_column:post_to_add,
                                        weight_column:0})
        concat_tuple.append(post_add_df)

    el = pd.concat(concat_tuple)
    Adf = el.pivot_table(index=post_column, columns=pre_column, values=weight_column, fill_value=0)
    return Adf[all_ids].loc[all_ids]

def adjacency_matrix_from_synapse_df(syn_df,
                                     pre_column='pre_pt_root_id',
                                     post_column='post_pt_root_id',
                                     weight_column='size',
                                     agg='count',
                                     id_list=None):
    """Convenience function making an adjacency matrix directly from a synapse dataframe.

    Parameters
    ----------
        syn_df : pandas.DataFrame 
            DataFrame with columns for pre id, post id, and at least one additional column to use as weight. 
        pre_column : str, optional
            Name of the presynaptic column, by default 'pre_pt_root_id'
        post_column : str, optional
            Name of the postsynaptic column, by default 'post_pt_root_id'
        weight_column : str, optional
            Name of the weight column, by default 'size'
        agg : str, optional
            Argument for the pandas groupby aggregation function, by default 'count'. Set to `sum` for using net synapse size instead.
        id_list : Collection, optional
            Collection of ids to use for the adjacency matrix indices, preserving order.
            If id_list is None, it uses exactly the ids in the edgelist.
            If id_list includes ids not in the edgelist, they become rows/columns with zeros.
            If id_list does not include ids that are in the edgelist, those edges are ignored.
            By default None
        
        Returns
        -------
        pandas.DataFrame
            Square dataframe with postsynaptic ids as index, presynaptic ids as columns, and
            values correspond to the weight column with 0s filled for unshown data.
    """
    el = edgelist_from_synapse_df(syn_df, pre_column, post_column, weight_column, agg)
    Adf = adjacency_matrix_from_edgelist(el, pre_column, post_column, weight_column, id_list)
    return Adf
