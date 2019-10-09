from sqlalchemy.orm import Query
from geoalchemy2.elements import WKBElement
from geoalchemy2.types import Geometry
from sqlalchemy.sql.sqltypes import Boolean
from decimal import Decimal
from multiwrapper import multiprocessing_utils as mu
import numpy as np
from geoalchemy2.shape import to_shape
from datetime import date, timedelta
from datetime import datetime
from functools import partial
import shapely


def fix_wkb_column(df_col, wkb_data_start_ind=2, n_threads=None):
    """Convert a column with 3-d point data stored as in WKB format
    to list of arrays of integer point locations. The series can not be
    mixed.
    
    Parameters
    ----------
    df_col : pandas.Series
        N-length Series (representing a column of a dataframe) to convert. All elements
        should be either a hex-string or a geoalchemy2 WKBElement object.
    wkb_data_start_ind : int, optional
        When the WKB data is represented as a hex string, sets the first character
        of the actual data. By default 2, since the current implementation has
        a prefix when the data is imported as text. Set to 0 if the data is just
        an exact hex string already. This value is ignored if the series data is in
        WKBElement object form.
    n_threads : int or None, optional
        Sets number of threads. If None, uses as many threads as CPUs.
        If n_threads is set to 1, multiprocessing is not used.
        Optional, by default None.
    
    Returns
    -------
    list
        N-length list of arrays of 3d points
    """

    if len(df_col) == 0:
        return df_col.tolist()

    if isinstance(df_col.loc[0], str):
        wkbstr = df_col.loc[0]
        shp = shapely.wkb.loads(wkbstr[wkb_data_start_ind:], hex=True)
        if isinstance(shp, shapely.geometry.point.Point):
            return _fix_wkb_hex_point_column(df_col, n_threads=n_threads)
    elif isinstance(df_col.loc[0], WKBElement):
        return _fix_wkb_object_point_column(df_col, n_threads=n_threads)
    return df_col.tolist()


def fix_columns_with_query(df, query, n_threads=None, fix_decimal=True, fix_wkb=True, wkb_data_start_ind=2):
    """ Use a query object to suggest how to convert columns imported from csv to correct types.
    """
    if len(df) > 0:
        schema_model = query.column_descriptions[0]['type']
        for colname in df.columns:
            coltype = type(getattr(schema_model, colname).type)

            if coltype is Boolean:
                df[colname] = _fix_boolean_column(df[colname])

            elif coltype is Geometry and fix_wkb is True:
                df[colname] = fix_wkb_column(
                    df[colname], wkb_data_start_ind=wkb_data_start_ind, n_threads=n_threads)

            elif isinstance(df[colname].loc[0], Decimal) and fix_decimal is True:
                df[colname] = _fix_decimal_column(df[colname])
            else:
                continue
    return df


def _wkb_object_point_to_numpy(wkb):
    """ Fixes single geometry element """
    shp = to_shape(wkb)
    return shp.xy[0][0], shp.xy[1][0], shp.z


def _fix_wkb_object_point_column(df_col, n_threads=None):
    if n_threads != 1:
        xyz = mu.multiprocess_func(_wkb_object_point_to_numpy, df_col.tolist(), n_threads=n_threads)
    else:
        func = np.vectorize(_wkb_object_point_to_numpy)
        xyz = np.vstack(func(df_col.values)).T
    return list(np.array(xyz, dtype=int))

def _wkb_hex_point_to_numpy(wkbstr, wkb_data_start_ind=2):
    shp = shapely.wkb.loads(wkbstr[wkb_data_start_ind:], hex=True)
    return shp.xy[0][0], shp.xy[1][0], shp.z

def _fix_wkb_hex_point_column(df_col, wkb_data_start_ind=2, n_threads=None):
    func = partial(_wkb_hex_point_to_numpy, wkb_data_start_ind=wkb_data_start_ind)
    if n_threads != 1:
        xyz = mu.multiprocess_func(func, df_col.tolist(), n_threads)
    else:
        func = np.vectorize(func)
        xyz = np.vstack(func(df_col.values)).T
    return list(np.array(xyz, dtype=int))


def _fix_boolean_column(df_col):
    return df_col.apply(lambda x: True if x == 't' else False)


def _fix_decimal_column(df_col):
    is_integer_col = np.vectorize(lambda x: float(x).is_integer())
    if np.all(is_integer_col(df_col)):
        return df_col.apply(int)
    else:
        return df_col.apply(np.float)

def render_query(statement, dialect=None):
    """
    Based on https://stackoverflow.com/questions/5631078/sqlalchemy-print-the-actual-query#comment39255415_23835766
    Generate an SQL expression string with bound parameters rendered inline
    for the given SQLAlchemy statement.
    """
    if isinstance(statement, Query):
        if dialect is None:
            dialect = statement.session.bind.dialect
        statement = statement.statement
    elif dialect is None:
        dialect = statement.bind.dialect

    class LiteralCompiler(dialect.statement_compiler):

        def visit_bindparam(self, bindparam, within_columns_clause=False,
                            literal_binds=False, **kwargs):
            return self.render_literal_value(bindparam.value, bindparam.type)

        def render_array_value(self, val, item_type):
            if isinstance(val, list):
                return "{%s}" % ",".join([self.render_array_value(x, item_type) for x in val])
            return self.render_literal_value(val, item_type)

        def render_literal_value(self, value, type_):
            if isinstance(value, int):
                return str(value)
            if isinstance(value, bool):
                return bool(value)
            elif isinstance(value, (str, date, datetime, timedelta)):
                return "'%s'" % str(value).replace("'", "''")
            elif isinstance(value, list):
                return "'{%s}'" % (",".join([self.render_array_value(x, type_.item_type) for x in value]))
            return super(LiteralCompiler, self).render_literal_value(value, type_)

    return LiteralCompiler(dialect, statement).process(statement)
