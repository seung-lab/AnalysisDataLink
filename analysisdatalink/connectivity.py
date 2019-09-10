import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    id_to_ind = {oid:ii for ii, oid in enumerate(all_idjs)}
    
    el = edgelist.copy()
    pre_to_add = all_ids[~np.isin(all_ids,edgelist[pre_column])]
    pre_add_df = pd.DataFrame(data={pre_column:pre_to_add,
                                    post_column:edgelist[post_column].iloc[0],
                                    weight_column:0})
    post_to_add = all_ids[~np.isin(all_ids,edgelist[post_column])]
    post_add_df = pd.DataFrame(data={pre_column:edgelist[pre_column].iloc[0],
                                     post_column:post_to_add,
                                     weight_column:0})
    
    el = pd.concat([el, pre_add_df, post_add_df])
    Adf = el.pivot_table(index=post_column, columns=pre_column, values=weight_column, fill_value=0)
    return Adf[id_list].loc[id_list]

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

def adjacency_rasterplot(Adf, id_order=None, ax=None, weight_col='weight',
                         pre_name='Presynaptic', post_name='Postsynaptic',
                         rows_are='pre', **kwargs):
    """Make a raster plot of an adjacency matrix from its dataframe with dots for nonzero values
        with appearance determined by seaborn scatterplot. All kwargs are passed to the seaborn scatterplot fucntion.
    
    Parameters
    ----------
    Adf : pandas.DataFrame
        Dataframe with postsynaptic elements as rows, presynaptic elements as columns, and edge weight as values.
    id_order : Collection, optional
        An ordered list of ids to order the plot by, by default None.
    ax : matplotlib.Axis, optional
        Axis object for the plot, by default None
    pre_name: str, optional
        Column name to give to the presynaptic indices, default is 'Presynaptic'.
    post_name: str, optional
        Column name to give to teh postsynaptic indices, default is 'Postsynaptic'.
    weight_col : str, optional
        Column name to give to the values, by default 'weight'. Used for legends as well as visualization style.
    rows_are : ['pre' or 'post'], optional
        If 'pre', rows are presynaptic to columns (Wij with j->i).
        If 'post', columns are presynaptic to rows (Wij with i->j).
        By default 'pre'
    
    Returns
    -------
    matplotlib.Axis 
        Axis object for the plot
    """
    if id_order is not None:
        Adf = Adf[id_order].loc[id_order]
    ii, jj = np.where(Adf.values>0)
    w = Adf.values[ii,jj]

    data_df = pd.DataFrame(data={'Postsynaptic':ii, 'Presynaptic':jj, weight_col:w})
    if rows_are == 'pre':
        ax=sns.scatterplot(x='Postsynaptic', y='Presynaptic', size=weight_col, data=data_df, ax=ax, **kwargs)
    else:
        ax=sns.scatterplot(y='Postsynaptic', x='Presynaptic', size=weight_col, data=data_df, ax=ax, **kwargs)
    ax.set_aspect(1)
    if ax.get_legend() is not None:
        ax.legend(bbox_to_anchor=(1.01,1))
    return ax
