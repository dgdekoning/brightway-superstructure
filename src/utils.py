# -*- coding: utf-8 -*-
from typing import NamedTuple, Optional

import brightway2 as bw
from bw2data.backends.peewee import ActivityDataset, ExchangeDataset
from bw2data.utils import TYPE_DICTIONARY
import pandas as pd

FROM_ACT = pd.Index([
    "from activity name", "from reference product", "from location",
    "from database", "from key"
])
TO_ACT = pd.Index([
    "to activity name", "to reference product", "to location", "to database",
    "to key"
])
FROM_BIOS = pd.Index([
    "from activity name", "from categories", "from database", "from key"
])
TO_BIOS = pd.Index([
    "to activity name", "from categories", "to database", "to key"
])
FROM_ALL = pd.Index([
    "from activity name", "from reference product", "from location",
    "from categories", "from database", "from key"
])
TO_ALL = pd.Index([
    "to activity name", "to reference product", "to location", "to categories",
    "to database", "to key"
])
EXCHANGE_KEYS = pd.Index(["from key", "to key"])
SUPERSTRUCTURE = pd.Index([
    "from activity name",
    "from reference product",
    "from location",
    "from categories",
    "from database",
    "from key",
    "to activity name",
    "to reference product",
    "to location",
    "to categories",
    "to database",
    "to key",
    "flow type",
])


class Key(NamedTuple):
    database: str
    code: str

    @property
    def database_type(self) -> str:
        return "biosphere" if self.database == bw.config.biosphere else "technosphere"


class Index(NamedTuple):
    input: Key
    output: Key
    flow_type: Optional[str] = None

    @classmethod
    def build_from_exchange(cls, exc: ExchangeDataset) -> 'Index':
        return cls(
            input=Key(exc.input_database, exc.input_code),
            output=Key(exc.output_database, exc.output_code),
            flow_type=exc.type,
        )

    @property
    def input_document_id(self) -> int:
        return ActivityDataset.get(
            ActivityDataset.code == self.input.code,
            ActivityDataset.database == self.input.database
        ).id

    @property
    def output_document_id(self) -> int:
        return ActivityDataset.get(
            ActivityDataset.code == self.output.code,
            ActivityDataset.database == self.output.database
        ).id

    @property
    def exchange_type(self) -> int:
        if self.flow_type:
            return TYPE_DICTIONARY.get(self.flow_type, -1)
        exc_type = ExchangeDataset.get(
            ExchangeDataset.input_code == self.input.code,
            ExchangeDataset.input_database == self.input.database,
            ExchangeDataset.output_code == self.output.code,
            ExchangeDataset.output_database == self.output.database).type
        return TYPE_DICTIONARY.get(exc_type, -1)

    @property
    def ids_exc_type(self) -> (int, int, int):
        return self.input_document_id, self.output_document_id, self.exchange_type
