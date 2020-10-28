# -*- coding: utf-8 -*-
from typing import List

from bw2data.backends.peewee import (
    Exchange, ActivityDataset as AD, ExchangeDataset as ED
)
import numpy as np
import pandas as pd


def constuct_ad_data(row) -> tuple:
    """Take a namedtuple from the method below and convert it into two tuples."""
    key = (row.database, row.code)
    if row.type == "process":  # Technosphere
        data = (row.name, row.product, row.location, None, row.database)
    elif "categories" in row.data:  # Biosphere
        data = (row.name, np.NaN, np.NaN, row.data["categories"], row.database)
    else:  # Unknown, give incomplete information.
        data = (row.name, np.NaN, np.NaN, np.NaN, row.database)
    return key, data


def convert_key_to_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the process fields to its actual key by matching the database.
    """
    keys = set(df.iloc[:, 5])
    dbs, codes = zip(*keys)
    query = (AD
             .select()
             .where((AD.database.in_(set(dbs)))
                    & (AD.code.in_(set(codes))))
             .namedtuples())
    key_data = dict(constuct_ad_data(x) for x in query.iterator())
    subdf = pd.DataFrame([key_data[x] for x in df.iloc[:, 5]], columns=df.columns[0:5])
    return subdf


# Exchanges
def select_superstructure_indexes(struct: str) -> set:
    query = (ED.select(ED.input_code, ED.output_code)
             .where(ED.output_database == struct)
             .tuples())
    indexes = set(x for x in query.iterator())
    return indexes


def select_exchanges_by_database_codes(db_name: str, codes: set):
    inputs = set(x[0] for x in codes)
    outputs = set(x[1] for x in codes)
    query = (ED.select()
             .where((ED.output_database == db_name) &
                    (ED.input_code.in_(inputs)) &
                    (ED.output_code.in_(outputs)))
             .namedtuples())
    return query


def find_missing_exchanges(superstruct: set, delta: str) -> (set, list):
    query = (ED.select(ED.input_code, ED.output_code)
             .where(ED.output_database == delta)
             .distinct()
             .tuples())
    diff = set(x for x in query.iterator()).difference(superstruct)
    # Now query again, and create a list of exchanges of the diff.
    query = select_exchanges_by_database_codes(delta, diff)
    diff_list = [x for x in query.iterator()]
    return diff, diff_list


def select_exchange_data(db_name: str):
    query = (ED.select(ED.data)
             .where((ED.output_database == db_name) &
                    (ED.type.in_(["biosphere", "technosphere"])))
             .tuples())
    data = (x[0] for x in query.iterator())
    return data


def nullify_exchanges(data: List[dict]) -> List[dict]:
    """Take a list of exchange dictionaries, extract all the amounts
    and set the 'amount' in the dictionaries to 0."""
    def set_null(d):
        d["amount"] = 0
        return d
    nulled = list(map(set_null, data))
    return nulled


def swap_exchange_activities(data: dict, super_db: str, delta_set: set) -> Exchange:
    """Take the exchange data and replace one or two activities inside with
    new ones containing the same information.

    This works best with activities constructed like those of ecoinvent.
    """
    in_key = data.get("input", ("",))
    out_key = data.get("output", ("",))
    if in_key[0] in delta_set:
        data["input"] = (super_db, in_key[1])
    if out_key[0] in delta_set:
        data["output"] = (super_db, out_key[1])
    # Constructing the Exchange this way will cause a new row to be written
    e = Exchange(**data)
    return e
