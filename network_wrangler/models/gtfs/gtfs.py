"""Data Model for Pure GTFS Feed (not wrangler-flavored)."""

from typing import ClassVar

import geopandas as gpd

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
from .types import RouteType

# Constants for display
MAX_AGENCIES_DISPLAY = 3

# Route type categorizations
MIXED_TRAFFIC_ROUTE_TYPES = [
    RouteType.TRAM,
    RouteType.BUS,
    RouteType.CABLE_TRAM,
    RouteType.TROLLEYBUS,
]
"""GTFS route types that operate in mixed traffic so stops are nodes that are drive-accessible.

See [GTFS routes.txt](https://gtfs.org/documentation/schedule/reference/#routestxt)

- TRAM = Tram, Streetcar, Light rail, operates in mixed traffic AND at stations
- CABLE_TRAM = street-level rail with underground cable
- TROLLEYBUS = electric buses with overhead wires
"""

STATION_ROUTE_TYPES = [
    RouteType.TRAM,  # TODO: This is partial...
    RouteType.SUBWAY,
    RouteType.RAIL,
    RouteType.FERRY,
    RouteType.CABLE_TRAM,  # TODO: This is partial...
    RouteType.AERIAL_LIFT,
    RouteType.FUNICULAR,
    RouteType.MONORAIL,
]
"""GTFS route types that operate at stations.
"""

RAIL_ROUTE_TYPES = [
    RouteType.TRAM,
    RouteType.SUBWAY,
    RouteType.RAIL,
    RouteType.CABLE_TRAM,
    RouteType.AERIAL_LIFT,
    RouteType.FUNICULAR,
    RouteType.MONORAIL,
]
"""GTFS route types which trigger 'rail_only' link creation in add_stations_and_links_to_roadway_network()
"""

FERRY_ROUTE_TYPES = [RouteType.FERRY]
"""GTFS route types which trigger 'ferry_only' link creation in add_stations_and_links_to_roadway_network()
"""


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

    def __repr__(self) -> str:  # noqa: PLR0912
        """Return a string representation of the GtfsModel with table summaries."""
        lines = ["GtfsModel:"]

        # Add agency info if available
        if hasattr(self, "agency") and self.agency is not None and len(self.agency) > 0:
            agency_names = self.agency.agency_name.tolist()[:MAX_AGENCIES_DISPLAY]
            if len(self.agency) > MAX_AGENCIES_DISPLAY:
                agency_names.append(f"... and {len(self.agency) - MAX_AGENCIES_DISPLAY} more")
            lines.append(
                f"  Agencies ({len(self.agency)}): {', '.join(str(a) for a in agency_names)}"
            )

        # Add summary for each table with type info
        table_summaries = []
        if hasattr(self, "routes") and self.routes is not None:
            table_summaries.append(f"{len(self.routes)} routes")
        if hasattr(self, "trips") and self.trips is not None:
            table_summaries.append(f"{len(self.trips)} trips")
        if hasattr(self, "stops") and self.stops is not None:
            table_summaries.append(f"{len(self.stops)} stops")
        if hasattr(self, "stop_times") and self.stop_times is not None:
            table_summaries.append(f"{len(self.stop_times)} stop_times")
        if hasattr(self, "shapes") and self.shapes is not None:
            n_shapes = len(self.shapes.shape_id.unique()) if len(self.shapes) > 0 else 0
            table_summaries.append(f"{n_shapes} shapes ({len(self.shapes)} points)")
        if (
            hasattr(self, "frequencies")
            and self.frequencies is not None
            and len(self.frequencies) > 0
        ):
            table_summaries.append(f"{len(self.frequencies)} frequencies")

        if table_summaries:
            lines.append(f"  Tables: {', '.join(table_summaries)}")

        # Add type information for each table
        type_info = []
        for table_name in [
            "agency",
            "routes",
            "trips",
            "stops",
            "stop_times",
            "shapes",
            "frequencies",
        ]:
            if hasattr(self, table_name) and getattr(self, table_name) is not None:
                table = getattr(self, table_name)
                if isinstance(table, gpd.GeoDataFrame):
                    type_info.append(f"{table_name}: GeoDataFrame")
                else:
                    type_info.append(f"{table_name}: DataFrame")

        if type_info:
            lines.append(f"  Types: {', '.join(type_info)}")

        # Add route type breakdown if routes exist
        if hasattr(self, "routes") and self.routes is not None and len(self.routes) > 0:
            route_type_counts = self.routes.route_type.value_counts().sort_index()
            route_type_names = {
                0: "Tram",
                1: "Metro",
                2: "Rail",
                3: "Bus",
                4: "Ferry",
                5: "Cable",
                6: "Gondola",
                7: "Funicular",
            }
            route_types = []
            for rt, count in route_type_counts.items():
                name = route_type_names.get(rt, f"Type{rt}")
                route_types.append(f"{name}:{count}")
            lines.append(f"  Route types: {', '.join(route_types)}")

        return "\n".join(lines)
