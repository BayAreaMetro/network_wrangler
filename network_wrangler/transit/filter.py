"""Functions to filter transit feeds by various criteria.

Filtered transit feeds are subsets of the original feed based on selection criteria
like service_ids, route_types, etc.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from ..logger import WranglerLogger
from ..models.gtfs.gtfs import GtfsModel
from .feed.feed import Feed


def filter_feed_by_service_ids(
    feed: Union[Feed, GtfsModel],
    service_ids: list[str],
) -> Union[Feed, GtfsModel]:
    """Filter a transit feed to only include trips for specified service_ids.
    
    Filters trips, stop_times, and stops to only include data related to the
    specified service_ids. Also ensures parent stations are retained if referenced.
    
    Args:
        feed: Feed or GtfsModel object to filter
        service_ids: List of service_ids to retain
        
    Returns:
        Union[Feed, GtfsModel]: Filtered copy of feed with only trips/stops/stop_times 
            for specified service_ids. Returns same type as input.
    """
    WranglerLogger.info(f"Filtering feed to {len(service_ids):,} service_ids")
    
    # Remember the input type to return the same type
    is_feed = isinstance(feed, Feed)
    
    # Extract dataframes to work with them directly (avoiding validation during filtering)
    feed_dfs = {}
    for table_name in feed.table_names:
        if hasattr(feed, table_name) and getattr(feed, table_name) is not None:
            feed_dfs[table_name] = getattr(feed, table_name).copy()
    
    # Filter trips for these service_ids
    original_trip_count = len(feed_dfs["trips"])
    feed_dfs["trips"]["service_id"] = feed_dfs["trips"]["service_id"].astype(str)
    
    # Create a DataFrame from the list for merging
    service_ids_df = pd.DataFrame({"service_id": service_ids})
    feed_dfs["trips"] = feed_dfs["trips"].merge(
        right=service_ids_df, on="service_id", how="left", indicator=True
    )
    WranglerLogger.debug(
        f"trips._merge.value_counts():\n{feed_dfs['trips']._merge.value_counts()}"
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
    
    # Filter stop_times for these trip_ids
    feed_dfs["trips"]["trip_id"] = feed_dfs["trips"]["trip_id"].astype(str)
    trip_ids = feed_dfs["trips"][["trip_id"]].drop_duplicates().reset_index(drop=True)
    WranglerLogger.debug(
        f"After filtering trips to trip_ids (len={len(trip_ids):,})"
    )
    
    feed_dfs["stop_times"]["trip_id"] = feed_dfs["stop_times"]["trip_id"].astype(str)
    feed_dfs["stop_times"] = feed_dfs["stop_times"].merge(
        right=trip_ids, how="left", indicator=True
    )
    WranglerLogger.debug(
        f"stop_times._merge.value_counts():\n{feed_dfs['stop_times']._merge.value_counts()}"
    )
    feed_dfs["stop_times"] = (
        feed_dfs["stop_times"]
        .loc[feed_dfs["stop_times"]._merge == "both"]
        .drop(columns=["_merge"])
        .reset_index(drop=True)
    )
    
    # Filter stops for these stop_ids
    feed_dfs["stop_times"]["stop_id"] = feed_dfs["stop_times"]["stop_id"].astype(str)
    stop_ids = feed_dfs["stop_times"][["stop_id"]].drop_duplicates().reset_index(drop=True)
    WranglerLogger.debug(f"After filtering stop_times to stop_ids (len={len(stop_ids):,})")
    
    feed_dfs["stops"]["stop_id"] = feed_dfs["stops"]["stop_id"].astype(str)
    
    # Save a copy of all stops before filtering
    all_stops_df = feed_dfs["stops"].copy()
    
    # First filter to stops referenced in stop_times
    feed_dfs["stops"] = feed_dfs["stops"].merge(right=stop_ids, how="left", indicator=True)
    WranglerLogger.debug(
        f"stops._merge.value_counts():\n{feed_dfs['stops']._merge.value_counts()}"
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
    
    # Check for stop_times with invalid stop_ids after all filtering is complete
    valid_stop_ids = set(feed_dfs["stops"]["stop_id"])
    invalid_mask = ~feed_dfs["stop_times"]["stop_id"].isin(valid_stop_ids)
    invalid_stop_times = feed_dfs["stop_times"][invalid_mask]
    
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
            f"Invalid stop_ids ({len(invalid_stop_ids)} unique): {invalid_stop_ids.tolist()}"
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
        
        WranglerLogger.debug(
            f"All invalid stop_times entries with routes:\n{invalid_with_routes}"
        )
    else:
        WranglerLogger.info(
            "All stop_times entries have valid stop_ids after filtering"
        )
    
    # Filter other tables to match filtered trips
    if "shapes" in feed_dfs:
        shape_ids = feed_dfs["trips"]["shape_id"].dropna().unique()
        feed_dfs["shapes"] = feed_dfs["shapes"][
            feed_dfs["shapes"]["shape_id"].isin(shape_ids)
        ].reset_index(drop=True)
        WranglerLogger.debug(f"Filtered shapes to {len(feed_dfs['shapes']):,} records")
    
    if "routes" in feed_dfs:
        route_ids = feed_dfs["trips"]["route_id"].unique()
        feed_dfs["routes"] = feed_dfs["routes"][
            feed_dfs["routes"]["route_id"].isin(route_ids)
        ].reset_index(drop=True)
        WranglerLogger.debug(f"Filtered routes to {len(feed_dfs['routes']):,} records")
    
    # Feed has frequencies, GtfsModel doesn't
    if is_feed and "frequencies" in feed_dfs:
        feed_dfs["frequencies"]["trip_id"] = feed_dfs["frequencies"]["trip_id"].astype(str)
        feed_dfs["frequencies"] = feed_dfs["frequencies"][
            feed_dfs["frequencies"]["trip_id"].isin(feed_dfs["trips"]["trip_id"])
        ].reset_index(drop=True)
        WranglerLogger.debug(f"Filtered frequencies to {len(feed_dfs['frequencies']):,} records")
    
    # Create the appropriate object type with the filtered dataframes
    if is_feed:
        return Feed(**feed_dfs)
    else:
        return GtfsModel(**feed_dfs)