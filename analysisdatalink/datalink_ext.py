from analysisdatalink import datalink


class AnalysisDataLinkExt(datalink.AnalysisDataLink):
    def __init__(self, dataset_name, materialization_version,
                 sqlalchemy_database_uri=None):
        super().__init__(dataset_name, materialization_version,
                         sqlalchemy_database_uri)

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