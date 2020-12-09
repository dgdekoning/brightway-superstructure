# -*- coding: utf-8 -*-
from typing import List, Tuple

from bw2data.backends.peewee import (
    Activity, Exchange, ActivityDataset as AD, ExchangeDataset as ED
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


def select_superstructure_codes(struct: str) -> set:
    query = (AD.select(AD.code)
             .where(AD.database == struct)
             .distinct()
             .tuples())
    codes = set(x[0] for x in query.iterator())
    return codes


def find_missing_activities(existing_codes: set, delta: str) -> Tuple[set, list]:
    query = (AD.select(AD.code)
             .where(AD.database == delta)
             .distinct()
             .tuples())
    diff = set(x[0] for x in query.iterator()).difference(existing_codes)
    # Now query again, and create a list of Activities of the diff.
    query = (AD.select()
             .where((AD.database == delta) & (AD.code.in_(diff))))
    diff_list = [Activity(x) for x in query.iterator()]
    return diff, diff_list


def structure_activities(data: List[Activity], db_name: str) -> List[Activity]:
    """Takes a list of activity objects and generates a list of new Activity
    objects with the (output) database name altered.
    """
    altered = []
    for act in data:
        activity = Activity()
        for key, value in act.items():
            activity[key] = value
        # Alter the database.
        activity._data["database"] = db_name
        altered.append(activity)
    return altered


# Exchanges
def select_superstructure_indexes(struct: str) -> set:
    query = (ED.select(ED.input_code, ED.output_code)
             .where(ED.output_database == struct)
             .tuples())
    indexes = set(x for x in query.iterator())
    return indexes


def find_missing_exchanges(superstruct: set, delta: str) -> Tuple[set, list]:
    query = (ED.select(ED.input_code, ED.output_code)
             .where(ED.output_database == delta)
             .distinct()
             .tuples())
    diff = set(x for x in query.iterator()).difference(superstruct)
    # Now query again, and create a list of exchanges of the diff.
    inputs = set(x[0] for x in diff)
    outputs = set(x[1] for x in diff)
    query = (ED.select(ED.data)
             .where((ED.output_database == delta) &
                    (ED.input_code.in_(inputs)) &
                    (ED.output_code.in_(outputs)))
             .tuples())
    diff_list = [x[0] for x in query.iterator()]
    return diff, diff_list


def select_exchange_data(db_name: str):
    query = (ED.select(ED.data)
             .where((ED.output_database == db_name) &
                    (ED.type.in_(["biosphere", "technosphere", "production"])))
             .tuples())
    data = (x[0] for x in query.iterator())
    return data


def structure_exchanges(data: List[dict], super_db: str, deltas: set) -> List[Exchange]:
    """Take a list of dictionaries and structure them into a list of Exchange
    objects, adjusted for the superstructure database.
    """
    def alter_data(d):
        amount = d.get("amount")
        d["amount"] = amount if d["type"] == "production" else 0
        key = d["input"]
        d["input"] = (super_db, key[1]) if key[0] in deltas else key
        key = d["output"]
        d["output"] = (super_db, key[1]) if key[0] in deltas else key
        return d

    altered_exchanges = map(alter_data, data)
    new_exchanges = [Exchange(**exc) for exc in altered_exchanges]
    return new_exchanges
