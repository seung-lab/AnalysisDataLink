# AnalysisDataLink

This repostitory facilitates easy access to the materialized (SQL) database tables. Have a look at https://github.com/seung-lab/AnnotationPipelineOverview to get a better overview of the system. 

The database can be accessed directly as described in [below](https://github.com/seung-lab/AnalysisDataLink#accessing-the-sql-databases-directly). However, it is recommended to use this repository as it not only helps with querying the database but also sets datatypes and converts the geometry coordinates which are stored in a postgis string format.

## Access through the DataLink

The DataLink has three hierarchy levels:
- low level: `datalink_base.py`
- query level: `datalink.py`
- abstract level: `datalink_ext.py`

We anticpate that most users operate on the highest level where queries to the different table schemas are predefined for convenient access. However, these functions might be too limited in some cases and require more low level access. We hope that users contribute to this repo by formulating their currently unsupported queries with the means of the lower level modules and adding them to `datalink_ext.py`. 

### Example

Accessing synapses from all pyramidal cells onto all other cells:

```
from analysisdatalink import datalink_ext as de
adle = de.AnalysisDataLinkExt("pinky100", 50, sqlalchemy_database_uri)

# Read all pyramidal cell ids
pyc_ids = adle.query_cell_types("soma_valence", cell_type_include_filter=["e"], return_only_ids=True, exclude_zero_root_ids=True)

# Read synapses restricted to pyramidal cells (takes ~11s and returns 17571 synapses)
synapse_df = adle.query_synapses("pni_synapses_i3", pre_ids=pyc_ids)
```

See below for how to build the `sqlalchemy_database_uri`. For convenience, one can define `DATABASE_URI` as global system variable which will be read if `sqlalchemy_database_uri` is undefined.


## Accessing the SQL databases directly

The SQL database can be accessed in many ways, sqlAlchemy and pandas are a good place to start. Adminer is a good tool to view the database content.

### Table naming

All tables are called following a convention:
```
{dataset_name}_{table_name}_v{materialization_version}
```
For instance, a synapse table might be called: `pinky100_pni_synapses_i3_v38`.

### Pandas examples

Getting all the cell segment ids (also called root ids):

```
import pandas as pd
sql_query = "SELECT * FROM pinky100_cellsegment_v38"
df = pd.read_sql(sql_query, database_uri, index_col="id")
```

where `database_uri` is build as follows:

```
database_uri = "postgresql://{user_name}:{password}@{database_ip}/postgres"
```

