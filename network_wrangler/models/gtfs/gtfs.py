"""Data Model for Pure GTFS Feed (not wrangler-flavored)."""

from typing import ClassVar

from ...models._base.db import DBModelMixin
from .tables import (
    AgenciesTable,
    FrequenciesTable,
    RoutesTable,
    ShapesTable,
    StopsTable,
    StopTimesTable,
    TripsTable,
)


class GtfsValidationError(Exception):
    """Exception raised for errors in the GTFS feed."""


class GtfsModel(DBModelMixin):
    """Wrapper class around GTFS feed.

    This is the pure GTFS model version of [Feed][network_wrangler.transit.feed.feed.Feed]

    Most functionality derives from mixin class
    [`DBModelMixin`][network_wrangler.models._base.db.DBModelMixin] which provides:

    - validation of tables to schemas when setting a table attribute (e.g. self.trips = trips_df)
    - validation of fks when setting a table attribute (e.g. self.trips = trips_df)
    - hashing and deep copy functionality
    - overload of __eq__ to apply only to tables in table_names.
    - convenience methods for accessing tables

    Attributes:
        table_names (list[str]): list of table names in GTFS feed.
        tables (list[DataFrame]): list tables as dataframes.
        agency (DataFrame[AgenciesTable]): agency dataframe
        stop_times (DataFrame[StopTimesTable]): stop_times dataframe
        stops (DataFrame[WranglerStopsTable]): stops dataframe
        shapes (DataFrame[ShapesTable]): shapes dataframe
        trips (DataFrame[TripsTable]): trips dataframe
        frequencies (Optional[DataFrame[FrequenciesTable]]): frequencies dataframe
        routes (DataFrame[RoutesTable]): route dataframe
        net (Optional[TransitNetwork]): TransitNetwork object
    """

    # the ordering here matters because the stops need to be added before stop_times if
    # stop times needs to be converted
    _table_models: ClassVar[dict] = {
        "agency": AgenciesTable,
        "frequencies": FrequenciesTable,
        "routes": RoutesTable,
        "shapes": ShapesTable,
        "stops": StopsTable,
        "trips": TripsTable,
        "stop_times": StopTimesTable,
    }

    table_names: ClassVar[list[str]] = [
        "agency",
        "routes",
        "shapes",
        "stops",
        "trips",
        "stop_times",
    ]

    optional_table_names: ClassVar[list[str]] = ["frequencies"]

    def __init__(self, **kwargs):
        """Initialize GTFS model."""
        self.initialize_tables(**kwargs)

        # Set extra provided attributes.
        extra_attr = {k: v for k, v in kwargs.items() if k not in self.table_names}
        for k, v in extra_attr.items():
            self.__setattr__(k, v)
