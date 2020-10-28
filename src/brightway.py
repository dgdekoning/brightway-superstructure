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


def check_for_invalid_codes(df: pd.DataFrame, struct_db: str) -> set:
    """Check if the given superstructure contains keys for the superstructure
    database that do not exist.

    Return a set of codes where the keys are invalid.
    """
    codes = set(x[1] for x in df["from key"] if x[0] == struct_db).union(
        x[1] for x in df["to key"] if x[0] == struct_db
    )
    missing_codes = set()
    query = (AD.select(AD.code)
             .where((AD.code.in_(codes)) & (AD.database == struct_db))
             .tuples())
    if not len(codes) == query.count():
        # This means not all of the codes exist in the superstructure.
        missing_codes = codes.difference(x[0] for x in query.iterator())
    return missing_codes


def handle_code_weirdness(codes: set, dbs: set, struct_db: str) -> dict:
    """Sometimes, we might be working with weird data, where the codes of
    activities no longer match, while the rest of the data absolutely does.

    So, here we takes these codes and original databases and yoink the
    other important data (location, name, product, etc.), using that
    to find a match in the superstructure database.

    The return value is a dictionary where the invalid keys are linked
    to the valid ones.
    """
    query = (AD.select(AD.name, AD.product, AD.location, AD.code)
             .where((AD.code.in_(codes)) & (AD.database.in_(dbs)))
             .tuples())
    combo_dict = {x[:-1]: x[-1] for x in query.iterator()}

    names, products, locations = zip(*combo_dict.keys())
    query = (AD.select(AD.name, AD.product, AD.location, AD.code)
             .where((AD.name.in_(set(names))) & (AD.product.in_(set(products)))
                    & (AD.location.in_(set(locations))) & (AD.database == struct_db))
             .tuples())
    match_dict = {x[:-1]: x[-1] for x in query.iterator()}
    final = {(struct_db, combo_dict[k]): (struct_db, v) for k, v in match_dict.items()}
    return final


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
