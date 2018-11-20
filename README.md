# AnalysisDataLink

This repostitory will facilitate easy access to the materialized (SQL) databases once it is build up. Have a look at https://github.com/seung-lab/AnnotationPipelineOverview to get a better overview of the system. 


## Accessing the SQL databases

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

