"""Functions for reading and writing transit feeds and networks."""

import os
from pathlib import Path
from typing import Literal, Optional, Union

import geopandas as gpd
import pandas as pd
import pyarrow as pa

from ..configs import DefaultConfig, WranglerConfig
from ..errors import FeedReadError
from ..logger import WranglerLogger
from ..models._base.db import RequiredTableError
from ..models._base.types import TransitFileTypes
from ..models.gtfs.gtfs import GtfsModel
from ..utils.geo import to_points_gdf
from ..utils.io_table import unzip_file, write_table
from .feed.feed import Feed
from .network import TransitNetwork

# Constants
MAX_INVALID_STOPS_DISPLAY = 20


def _feed_path_ref(path: Path) -> Path:
    if path.suffix == ".zip":
        path = unzip_file(path)
    if not path.exists():
        msg = f"Feed path does not exist: {path}"
        raise FileExistsError(msg)

    return path


def load_feed_from_path(  # noqa: PLR0915
    feed_path: Union[Path, str],
    file_format: TransitFileTypes = "txt",
    wrangler_flavored: bool = True,
    service_ids_filter: pd.DataFrame = None,
) -> Union[Feed, GtfsModel]:
    """Create a Feed or GtfsModel object from the path to a GTFS transit feed.

    Args:
        feed_path (Union[Path, str]): The path to the GTFS transit feed.
        file_format: the format of the files to read. Defaults to "txt"
        wrangler_flavored: If True, creates a Wrangler-enhanced Feed object.
                          If False, creates a pure GtfsModel object. Defaults to True.
        service_ids_filter (DataFrame): If not None, filter to these service_ids. Assumes service_id is a str.

    Returns:
        Union[Feed, GtfsModel]: The Feed or GtfsModel object created from the GTFS transit feed.
    """
    feed_path = _feed_path_ref(Path(feed_path))  # unzips if needs to be unzipped

    if not feed_path.is_dir():
        msg = f"Feed path not a directory: {feed_path}"
        raise NotADirectoryError(msg)

    WranglerLogger.info(f"Reading GTFS feed tables from {feed_path}")

    # Use the appropriate table names based on the model type
    model_class = Feed if wrangler_flavored else GtfsModel
    feed_possible_files = {
        table: list(feed_path.glob(f"*{table}.{file_format}")) for table in model_class.table_names
    }
    WranglerLogger.debug(f"model_class={model_class}  feed_possible_files={feed_possible_files}")

    # make sure we have all the tables we need
    _missing_files = [t for t, v in feed_possible_files.items() if not v]

    if _missing_files:
        WranglerLogger.debug(f"!!! Missing transit files: {_missing_files}")
        model_name = "Feed" if wrangler_flavored else "GtfsModel"
        msg = f"Required GTFS {model_name} table(s) not in {feed_path}: \n  {_missing_files}"
        raise RequiredTableError(msg)

    # but don't want to have more than one file per search
    _ambiguous_files = [t for t, v in feed_possible_files.items() if len(v) > 1]
    if _ambiguous_files:
        WranglerLogger.warning(
            f"! More than one file matches following tables. "
            + f"Using the first on the list: {_ambiguous_files}"
        )

    feed_files = {t: f[0] for t, f in feed_possible_files.items()}
    feed_dfs = {table: _read_table_from_file(table, file) for table, file in feed_files.items()}

    if service_ids_filter is not None:
        WranglerLogger.info(f"Filtering trips to {len(service_ids_filter)} service_ids")
        WranglerLogger.debug(f"Filtering service_ids: {service_ids_filter}")

        # filter to trips for these service_ids
        original_trip_count = len(feed_dfs["trips"])
        feed_dfs["trips"]["service_id"] = feed_dfs["trips"]["service_id"].astype(
            str
        )  # make service_id a string
        feed_dfs["trips"] = feed_dfs["trips"].merge(
            right=service_ids_filter, on="service_id", how="left", indicator=True
        )
        WranglerLogger.debug(
            f"feed_dfs['trips']._merge.value_counts():\n{feed_dfs['trips']._merge.value_counts()}"
        )
        feed_dfs["trips"] = (
            feed_dfs["trips"]
            .loc[feed_dfs["trips"]._merge == "both"]
            .drop(columns=["_merge"])
            .reset_index(drop=True)
        )
        WranglerLogger.info(
            f"Filtered trips from {original_trip_count:,} to {len(feed_dfs['trips']):,}"
        )
        WranglerLogger.debug(f"feed_dfs['trips']:\n{feed_dfs['trips']}")

        # filter stop_times for these trip_ids
        feed_dfs["trips"]["trip_id"] = feed_dfs["trips"]["trip_id"].astype(
            str
        )  # make trips.trip_id a string
        trip_ids = feed_dfs["trips"][["trip_id"]].drop_duplicates().reset_index(drop=True)
        WranglerLogger.debug(
            f"After filtering trips to trip_ids (len={len(trip_ids):,}), trip_ids=\n{trip_ids}"
        )

        feed_dfs["stop_times"]["trip_id"] = feed_dfs["stop_times"]["trip_id"].astype(
            str
        )  # make stop_times.trip_id a string
        feed_dfs["stop_times"] = feed_dfs["stop_times"].merge(
            right=trip_ids, how="left", indicator=True
        )
        WranglerLogger.debug(
            f"feed_dfs['stop_times']._merge.value_counts():\n{feed_dfs['stop_times']._merge.value_counts()}"
        )
        feed_dfs["stop_times"] = (
            feed_dfs["stop_times"]
            .loc[feed_dfs["stop_times"]._merge == "both"]
            .drop(columns=["_merge"])
            .reset_index(drop=True)
        )
        WranglerLogger.debug(f"feed_dfs['stop_times']:\n{feed_dfs['stop_times']}")

        # filter stops for these stop_ids
        feed_dfs["stop_times"]["stop_id"] = feed_dfs["stop_times"]["stop_id"].astype(
            str
        )  # make stop_times.stop_id a string
        stop_ids = feed_dfs["stop_times"][["stop_id"]].drop_duplicates().reset_index(drop=True)
        WranglerLogger.debug(f"After filtering stop_times to stop_ids (len={len(stop_ids):,})")

        feed_dfs["stops"]["stop_id"] = feed_dfs["stops"]["stop_id"].astype(
            str
        )  # make stops.stop_id a string

        # Save a copy of all stops before filtering
        all_stops_df = feed_dfs["stops"].copy()

        # First filter to stops referenced in stop_times
        feed_dfs["stops"] = feed_dfs["stops"].merge(right=stop_ids, how="left", indicator=True)
        WranglerLogger.debug(
            f"feed_dfs['stops']._merge.value_counts():\n{feed_dfs['stops']._merge.value_counts()}"
        )
        feed_dfs["stops"] = (
            feed_dfs["stops"]
            .loc[feed_dfs["stops"]._merge == "both"]
            .drop(columns=["_merge"])
            .reset_index(drop=True)
        )

        # Now check if any of these stops reference parent stations
        if "parent_station" in feed_dfs["stops"].columns:
            # Get parent stations that are referenced by kept stops
            parent_stations = feed_dfs["stops"]["parent_station"].dropna().unique()
            parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings

            if len(parent_stations) > 0:
                WranglerLogger.info(
                    f"Found {len(parent_stations)} parent stations referenced by kept stops"
                )

                # Find parent stations that aren't already in our filtered stops
                existing_stop_ids = set(feed_dfs["stops"]["stop_id"])
                missing_parent_stations = [
                    ps for ps in parent_stations if ps not in existing_stop_ids
                ]

                if len(missing_parent_stations) > 0:
                    WranglerLogger.debug(
                        f"Adding back {len(missing_parent_stations)} missing parent stations"
                    )

                    # Get the parent station records from our saved copy
                    parent_station_records = all_stops_df[
                        all_stops_df["stop_id"].isin(missing_parent_stations)
                    ]

                    # Append parent stations to filtered stops
                    feed_dfs["stops"] = pd.concat(
                        [feed_dfs["stops"], parent_station_records], ignore_index=True
                    )

                    WranglerLogger.debug(
                        f"After adding parent stations, stops count: {len(feed_dfs['stops']):,}"
                    )

        # Now check for stop_times with invalid stop_ids after all filtering is complete
        valid_stop_ids = set(feed_dfs["stops"]["stop_id"])
        invalid_mask = ~feed_dfs["stop_times"]["stop_id"].isin(valid_stop_ids)
        invalid_stop_times = feed_dfs["stop_times"][invalid_mask]
        WranglerLogger.info(
            f"Found {len(invalid_stop_times):,} stop_times entries with invalid stop_ids after filtering"
        )

        # This shouldn't happen if the data is valid but leaving the logging in just in case
        if len(invalid_stop_times) > 0:
            WranglerLogger.warning(
                f"Found {len(invalid_stop_times):,} stop_times entries with invalid stop_ids after filtering"
            )

            # Join with trips to get route_id
            invalid_with_routes = invalid_stop_times.merge(
                feed_dfs["trips"][["trip_id", "route_id"]], on="trip_id", how="left"
            )

            # Log unique invalid stop_ids
            invalid_stop_ids = invalid_stop_times["stop_id"].unique()
            WranglerLogger.warning(
                f"Invalid stop_ids ({len(invalid_stop_ids)} unique): {invalid_stop_ids[:MAX_INVALID_STOPS_DISPLAY].tolist()}..."
            )
            if len(invalid_stop_ids) > MAX_INVALID_STOPS_DISPLAY:
                WranglerLogger.warning(
                    f"  ... and {len(invalid_stop_ids) - MAX_INVALID_STOPS_DISPLAY} more invalid stop_ids"
                )

            # Log sample of invalid entries with trip and route context
            sample_invalid = invalid_with_routes.head(10)
            WranglerLogger.warning(
                f"Sample invalid stop_times entries:\n{sample_invalid[['trip_id', 'route_id', 'stop_id', 'stop_sequence']]}"
            )

            # Log summary by route
            route_summary = (
                invalid_with_routes.groupby("route_id")["stop_id"]
                .agg(["count", "nunique"])
                .sort_values("count", ascending=False)
            )
            route_summary.columns = ["invalid_stop_times_count", "unique_invalid_stops"]
            WranglerLogger.warning(
                f"Invalid stop_times by route (top 20):\n{route_summary.head(20)}"
            )

            # Log full details to debug
            WranglerLogger.debug(
                f"All invalid stop_times entries with routes:\n{invalid_with_routes}"
            )

            # Optionally save to CSV for investigation
            if Path(feed_path).exists():
                invalid_csv_path = Path(feed_path).parent / "invalid_stop_times_after_filtering.csv"
                invalid_with_routes.to_csv(invalid_csv_path, index=False)
                WranglerLogger.info(f"Saved invalid stop_times to {invalid_csv_path}")

            # Remove invalid entries
            feed_dfs["stop_times"] = feed_dfs["stop_times"][~invalid_mask].reset_index(drop=True)
            WranglerLogger.info(f"Removed {len(invalid_stop_times):,} invalid stop_times entries")

    return load_feed_from_dfs(feed_dfs, wrangler_flavored=wrangler_flavored)


def _read_table_from_file(table: str, file: Path) -> pd.DataFrame:
    WranglerLogger.debug(f"...reading {file}.")
    try:
        if file.suffix in [".csv", ".txt"]:
            return pd.read_csv(file, low_memory=False)
        if file.suffix == ".parquet":
            return pd.read_parquet(file)
    except Exception as e:
        msg = f"Error reading table {table} from file: {file}.\n{e}"
        WranglerLogger.error(msg)
        raise FeedReadError(msg) from e


def load_feed_from_dfs(feed_dfs: dict, wrangler_flavored: bool = True) -> Union[Feed, GtfsModel]:
    """Create a Feed or GtfsModel object from a dictionary of DataFrames representing a GTFS feed.

    Args:
        feed_dfs (dict): A dictionary containing DataFrames representing the tables of a GTFS feed.
        wrangler_flavored: If True, creates a Wrangler-enhanced Feed] object.
                           If False, creates a pure GtfsModel object. Defaults to True.

    Returns:
        Union[Feed, GtfsModel]: A Feed or GtfsModel object representing the transit network.

    Raises:
        ValueError: If the feed_dfs dictionary does not contain all the required tables.

    Example usage:
    ```python
    feed_dfs = {
        "agency": agency_df,
        "routes": routes_df,
        "stops": stops_df,
        "trips": trips_df,
        "stop_times": stop_times_df,
    }
    feed = load_feed_from_dfs(feed_dfs)  # Creates Feed by default
    gtfs_model = load_feed_from_dfs(feed_dfs, wrangler_flavored=False)  # Creates GtfsModel
    ```
    """
    # Use the appropriate model class based on the parameter
    model_class = Feed if wrangler_flavored else GtfsModel

    if not all(table in feed_dfs for table in model_class.table_names):
        model_name = "Feed" if wrangler_flavored else "GtfsModel"
        msg = f"feed_dfs must contain the following tables for {model_name}: {model_class.table_names}"
        raise ValueError(msg)

    feed = model_class(**feed_dfs)

    return feed


def load_transit(
    feed: Union[Feed, GtfsModel, dict[str, pd.DataFrame], str, Path],
    file_format: TransitFileTypes = "txt",
    config: WranglerConfig = DefaultConfig,
) -> TransitNetwork:
    """Create a [`TransitNetwork`][network_wrangler.transit.network.TransitNetwork] object.

    This function takes in a `feed` parameter, which can be one of the following types:

    - `Feed`: A Feed object representing a transit feed.
    - `dict[str, pd.DataFrame]`: A dictionary of DataFrames representing transit data.
    - `str` or `Path`: A string or a Path object representing the path to a transit feed file.

    Args:
        feed: Feed boject, dict of transit data frames, or path to transit feed data
        file_format: the format of the files to read. Defaults to "txt"
        config: WranglerConfig object. Defaults to DefaultConfig.

    Returns:
        (TransitNetwork): object representing the loaded transit network.

    Raises:
    ValueError: If the `feed` parameter is not one of the supported types.

    Example usage:
    ```python
    transit_network_from_zip = load_transit("path/to/gtfs.zip")

    transit_network_from_unzipped_dir = load_transit("path/to/files")

    transit_network_from_parquet = load_transit("path/to/files", file_format="parquet")

    dfs_of_transit_data = {"routes": routes_df, "stops": stops_df, "trips": trips_df...}
    transit_network_from_dfs = load_transit(dfs_of_transit_data)
    ```

    """
    if isinstance(feed, (Path, str)):
        feed = Path(feed)
        feed_obj = load_feed_from_path(feed, file_format=file_format)
        feed_obj.feed_path = feed
    elif isinstance(feed, dict):
        feed_obj = load_feed_from_dfs(feed)
    elif isinstance(feed, GtfsModel):
        feed_obj = Feed(**feed.__dict__)
    else:
        if not isinstance(feed, Feed):
            msg = f"TransitNetwork must be seeded with a Feed, dict of dfs or Path. Found {type(feed)}"
            raise ValueError(msg)
        feed_obj = feed

    return TransitNetwork(feed_obj, config=config)


def write_transit(
    transit_obj: Union[TransitNetwork, Feed, GtfsModel],
    out_dir: Union[Path, str] = ".",
    prefix: Optional[Union[Path, str]] = None,
    file_format: Literal["txt", "csv", "parquet"] = "txt",
    overwrite: bool = True,
) -> None:
    """Writes a transit network, feed, or GTFS model to files.

    Args:
        transit_obj: a TransitNetwork, Feed, or GtfsModel instance
        out_dir: directory to write the network to
        file_format: the format of the output files. Defaults to "txt" which is csv with txt
            file format.
        prefix: prefix to add to the file name
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    out_dir = Path(out_dir)
    prefix = f"{prefix}_" if prefix else ""

    # Determine the data source based on input type
    if isinstance(transit_obj, TransitNetwork):
        data_source = transit_obj.feed
        source_type = "TransitNetwork"
    elif isinstance(transit_obj, (Feed, GtfsModel)):
        data_source = transit_obj
        source_type = type(transit_obj).__name__
    else:
        msg = (
            f"transit_obj must be a TransitNetwork, Feed, or GtfsModel instance, "
            f"not {type(transit_obj).__name__}"
        )
        raise TypeError(msg)

    # Write tables
    tables_written = 0
    for table in data_source.table_names:
        df = data_source.get_table(table)
        if df is not None and len(df) > 0:  # Only write non-empty tables
            outpath = out_dir / f"{prefix}{table}.{file_format}"
            write_table(df, outpath, overwrite=overwrite)
            tables_written += 1

    WranglerLogger.info(f"Wrote {tables_written} {source_type} tables to {out_dir}")


def convert_transit_serialization(
    input_path: Union[str, Path],
    output_format: TransitFileTypes,
    out_dir: Union[Path, str] = ".",
    input_file_format: TransitFileTypes = "csv",
    out_prefix: str = "",
    overwrite: bool = True,
):
    """Converts a transit network from one serialization to another.

    Args:
        input_path: path to the input network
        output_format: the format of the output files. Should be txt, csv, or parquet.
        out_dir: directory to write the network to. Defaults to current directory.
        input_file_format: the file_format of the files to read. Should be txt, csv, or parquet.
            Defaults to "txt"
        out_prefix: prefix to add to the file name. Defaults to ""
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    WranglerLogger.info(
        f"Loading transit net from {input_path} with input type {input_file_format}"
    )
    net = load_transit(input_path, file_format=input_file_format)
    WranglerLogger.info(f"Writing transit network to {out_dir} in {output_format} format.")
    write_transit(
        net,
        prefix=out_prefix,
        out_dir=out_dir,
        file_format=output_format,
        overwrite=overwrite,
    )


def write_feed_geo(
    feed: Feed,
    ref_nodes_df: gpd.GeoDataFrame,
    out_dir: Union[str, Path],
    file_format: Literal["geojson", "shp", "parquet"] = "geojson",
    out_prefix=None,
    overwrite: bool = True,
) -> None:
    """Write a Feed object to a directory in a geospatial format.

    Args:
        feed: Feed object to write
        ref_nodes_df: Reference nodes dataframe to use for geometry
        out_dir: directory to write the network to
        file_format: the format of the output files. Defaults to "geojson"
        out_prefix: prefix to add to the file name
        overwrite: if True, will overwrite the files if they already exist. Defaults to True
    """
    from .geo import shapes_to_shape_links_gdf  # noqa: PLC0415

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        if out_dir.parent.is_dir():
            out_dir.mkdir()
        else:
            msg = f"Output directory {out_dir} ands its parent path does not exist"
            raise FileNotFoundError(msg)

    prefix = f"{out_prefix}_" if out_prefix else ""
    shapes_outpath = out_dir / f"{prefix}trn_shapes.{file_format}"
    shapes_gdf = shapes_to_shape_links_gdf(feed.shapes, ref_nodes_df=ref_nodes_df)
    write_table(shapes_gdf, shapes_outpath, overwrite=overwrite)

    stops_outpath = out_dir / f"{prefix}trn_stops.{file_format}"
    stops_gdf = to_points_gdf(feed.stops, ref_nodes_df=ref_nodes_df)
    write_table(stops_gdf, stops_outpath, overwrite=overwrite)
