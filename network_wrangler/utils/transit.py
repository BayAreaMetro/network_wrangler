"""Utilities for getting GTFS into wrangler"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path
import pprint

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import networkx as nx

from ..logger import WranglerLogger
from ..models.gtfs.converters import (
    convert_stop_times_to_wrangler_stop_times,
    convert_stops_to_wrangler_stops,
)
from ..models.gtfs.gtfs import GtfsModel
from ..models.gtfs.types import RouteType
from ..roadway.network import RoadwayNetwork
from ..transit.feed.feed import Feed
from ..errors import NodeNotFoundError, TransitValidationError
from .time import time_to_seconds
from ..params import LAT_LON_CRS

MIXED_TRAFFIC_ROUTE_TYPES = [
    RouteType.TRAM, 
    RouteType.BUS, 
    RouteType.CABLE_TRAM, 
    RouteType.TROLLEYBUS
]
"""GTFS route types that operate in mixed traffic so stops are nodes that are drive-accessible.

See [GTFS routes.txt](https://gtfs.org/documentation/schedule/reference/#routestxt)

- TRAM = Tram, Streetcar, Light rail, operates in mixed traffic AND at stations
- CABLE_TRAM = street-level rail with underground cable
- TROLLEYBUS = electric buses with overhead wires
"""

STATION_ROUTE_TYPES = [
    RouteType.TRAM,       # TODO: This is partial...
    RouteType.SUBWAY, 
    RouteType.RAIL, 
    RouteType.FERRY,
    RouteType.CABLE_TRAM,  # TODO: This is partial...
    RouteType.AERIAL_LIFT, 
    RouteType.FUNICULAR, 
    RouteType.MONORAIL
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
    RouteType.MONORAIL
]
"""GTFS route types which trigger 'rail_only' link creation in add_stations_and_links_to_roadway_network()
"""

FERRY_ROUTE_TYPES = [
    RouteType.FERRY
]
"""GTFS route types which trigger 'ferry_only' link creation in add_stations_and_links_to_roadway_network()
"""

CONNECTIVITY_MATCH_ROUTE_TYPES = [
    RouteType.SUBWAY, 
    RouteType.RAIL, 
    RouteType.FERRY
]
"""GTFS route types that should use connectivity-aware matching.
"""

STATION_ONLY_ROUTE_TYPES = list(set(STATION_ROUTE_TYPES) - set(MIXED_TRAFFIC_ROUTE_TYPES))
"""GTFS route types that operate ONLY at stations (not in mixed traffic).

This is STATION_ROUTE_TYPES minus MIXED_TRAFFIC_ROUTE_TYPES.
"""

MIXED_TRAFFIC_ONLY_ROUTE_TYPES = list(set(MIXED_TRAFFIC_ROUTE_TYPES) - set(STATION_ROUTE_TYPES))
"""GTFS route types that operate ONLY in mixed traffic.

This is MIXED_TRAFFIC_ROUTE_TYPES minus STATION_ROUTE_TYPES.
"""

MIXED_TRAFFIC_AND_STATION_ROUTE_TYPES = list(set(MIXED_TRAFFIC_ROUTE_TYPES).intersection(STATION_ROUTE_TYPES))
"""Route types that operate in both mixed traffic and at stations."""

FEET_PER_MILE = 5280.0
"""Feet to miles conversion constant."""

METERS_PER_KILOMETER = 1000.0
"""Meters to kilometers conversion constant."""

MAX_DISTANCE_STOP = {
    'feet': 0.25 * FEET_PER_MILE,
    'meters': 0.40 * METERS_PER_KILOMETER,
}
"""Maximum distance for a stop to match to a node."""

def create_feed_frequencies(
    feed_tables: Dict[str, pd.DataFrame],
    timeperiods: Dict[str, Tuple[str, str]],
    frequency_method: str,
    default_frequency_for_onetime_route: int = 10800,
    trace_shape_ids: Optional[List[str]] = None
):
    """Create frequencies table and convert GTFS-style tables to Wrangler-style Feed tables.

    This function transforms detailed GTFS trip schedules into a frequency-based representation
    used by the Wrangler Feed object. While GTFS specifies each individual transit trip with
    exact times, Wrangler uses representative trips with headways/frequencies for time periods.

    Process Steps:
    1. Adds route_id, direction_id, shape_id to stop_times from trips table
    2. Adds departure_minutes column (minutes after midnight) to stop_times
    3. Groups trips by stop pattern to verify consistency per shape_id
    4. Assigns each trip to a time period based on departure time
    5. Calculates headways using the specified frequency_method
    6. Creates new trip_ids based on shape_id (one trip per unique shape)
    7. Updates stop_times and trips tables to use new consolidated trip definitions

    See [GTFS frequencies.txt](https://gtfs.org/documentation/schedule/reference/#frequenciestxt)

    Args:
        feed_tables (Dict[str, pd.DataFrame]): Dictionary of feed tables to modify in place:
            Required input columns:
            - 'stop_times': trip_id, stop_id, stop_sequence, departure_time
            - 'trips': trip_id, route_id, direction_id, shape_id, service_id
            
            Columns added/modified:
            - 'stop_times': Adds route_id, direction_id, shape_id, departure_minutes;
                           Renames trip_id to orig_trip_id, creates new trip_id from shape_id
            - 'trips': Consolidated to one row per shape_id with new trip_id
            - 'frequencies': Created with trip_id, start_time, end_time, headway_secs
            
        timeperiods (Dict[str, Tuple[str, str]]): Time period labels to time spans.
            Example: {'EA': ('03:00','06:00'), 'AM': ('06:00','10:00')}
        frequency_method (str): Method for calculating headways:
            - 'uniform_headway': timeperiod_duration / number_of_trips
            - 'mean_headway': mean of time gaps between consecutive trips
            - 'median_headway': median of time gaps between consecutive trips
        default_frequency_for_onetime_route (int, optional): Default headway in seconds
            for routes with only one trip in a period. Defaults to 10800 (3 hours).
        trace_shape_ids (Optional[List[str]]): Shape IDs to log detailed debug output for.

    Returns:
        None - Modifies feed_tables in place

    Notes:
        - Assumes shape_id uniquely identifies a route/direction/stop pattern combination
        - Handles time periods crossing midnight (e.g., '19:00' to '03:00')
        - For single-trip routes, uses default_frequency_for_onetime_route
        - Original trip_ids are preserved as 'orig_trip_id' for reference
    """
    WranglerLogger.info("Creating frequencies table from feed stop_times data")

    VALID_FREQUENCY_METHOD = ["uniform_headway", "mean_headway", "median_headway" ]
    if frequency_method not in VALID_FREQUENCY_METHOD:
        raise ValueError(f"frequency_method must be on of {VALID_FREQUENCY_METHOD}; received {frequency_method}")

    # convert timeperiods to { timeperiod_label: [start_time, end_time]} where times are in minutes from midnight
    timeperiod_minutes_after_midnight = {}
    for label in timeperiods.keys():
        timeperiod_minutes_after_midnight[label] = [
            time_to_seconds(f"{timeperiods[label][0]}:00")/60.0,
            time_to_seconds(f"{timeperiods[label][1]}:00")/60.0,
        ]
    # sort by the start_time of the timeperiod
    WranglerLogger.debug(f"timeperiod_minutes_after_midnight:\n{timeperiod_minutes_after_midnight}")

    # Add route_id, direction_id, and shape_id to stop_times
    feed_tables["stop_times"] = pd.merge(
        left=feed_tables["stop_times"],
        right=feed_tables["trips"][["trip_id","shape_id","direction_id","route_id"]].drop_duplicates(),
        how="left",
        validate="many_to_one"
    )
    # Add departure_minutes, which is departure_time in minutes after midnight (easier to read than seconds)
    feed_tables["stop_times"]["departure_minutes"] = feed_tables["stop_times"]["departure_time"].apply(time_to_seconds)
    feed_tables["stop_times"]["departure_minutes"] = feed_tables["stop_times"]["departure_minutes"]/60.0 # convert to minutes
    feed_tables["stop_times"]["departure_minutes"] = feed_tables["stop_times"]["departure_minutes"].round(decimals=0)
    WranglerLogger.debug(f"feed_tables['stop_times']:\n{feed_tables['stop_times']}")

    # In GTFS, each trip_id is a vehicle trip with a single set of times for the stops.
    # In NetworkWrangler, each trip is a set of vehicle trips with a frequency per time period.
    # Assuming that the GTFS data includes shape_ids, which follow the shape of a route/direction,
    # we'll transform shape_ids to be the new trip_id, and determine frequencies for them
    WranglerLogger.info(f"There are {feed_tables['stop_times']['shape_id'].nunique():,} unique shape_ids.")
    WranglerLogger.info(f"These will serve as our new trip_ids.")

    # Check that stop patterns are the same for all trips for a given route_id, direction_id, shape_id
    # Group by trip_id and get ordered stop sequences
    stop_patterns_df = (
        feed_tables["stop_times"]
        .sort_values(['trip_id', 'stop_sequence'])
        .groupby('trip_id')
    ).aggregate(
        trip_depart_time = pd.NamedAgg(column='departure_minutes', aggfunc='first'),
        stop_pattern     = pd.NamedAgg(column='stop_id', aggfunc=list)
    ).reset_index(drop=False)
    WranglerLogger.debug(f"stop_patterns_df:\n{stop_patterns_df}")
    # columns are trip_id, trip_depart_time, stop_pattern

    # Merge with trips to get route_id, direction_id, shape_id
    stop_patterns_df = pd.merge(
        stop_patterns_df,
        feed_tables["trips"][['trip_id', 'route_id', 'direction_id', 'shape_id']].drop_duplicates(),
        on='trip_id',
        how='left',
        validate='one_to_one'
    )
    # Assign each trip to a timeperiod using the label
    # And calculate timeperiod_duration_minutes 
    timeperiod_duration_minutes = {}
    stop_patterns_df['timeperiod'] = 'not_set'
    for timeperiod in timeperiod_minutes_after_midnight.keys():
        # for the time period, in minutes after midnight
        start_time = timeperiod_minutes_after_midnight[timeperiod][0]
        end_time   = timeperiod_minutes_after_midnight[timeperiod][1]

        assert(start_time >= 0)
        assert(start_time < 24*60)
        assert(end_time >= 0)
        assert(end_time < 24*60)
        assert(start_time != end_time)
        # doesn't roll over midnight
        if start_time < end_time:
            stop_patterns_df.loc[ 
                (stop_patterns_df['trip_depart_time'] >= start_time) & 
                (stop_patterns_df['trip_depart_time'] < end_time),
                 'timeperiod' ] = timeperiod
            timeperiod_duration_minutes[timeperiod] = end_time - start_time
        # rolls over
        else:
            stop_patterns_df.loc[ 
                (stop_patterns_df['trip_depart_time'] >= start_time) |
                (stop_patterns_df['trip_depart_time'] < end_time),
                 'timeperiod' ] = timeperiod
            timeperiod_duration_minutes[timeperiod] = ((24*60) - start_time) + (end_time)

    WranglerLogger.debug(f"timeperiod_duration_minutes: {timeperiod_duration_minutes}")

    # Make sure all trips are assigned
    WranglerLogger.debug(f"trip_id by timeperiod:\n{stop_patterns_df['timeperiod'].value_counts()}")
    assert(len(stop_patterns_df.loc[ stop_patterns_df.timeperiod=='not_set']) == 0)

    WranglerLogger.debug(f"stop_patterns_df:\n{stop_patterns_df}")
    #                   trip_id  trip_depart_time                                       stop_pattern route_id direction_id           shape_id timeperiod
    # 0        3D:1090:20230930            261.00  [811425, 810752, 810747, 819910, 810641, 81063...  3D:300X            1     3D:35:20230930         EA
    # 1        3D:1091:20230930            321.00  [811425, 810752, 810747, 819910, 810641, 81063...  3D:300X            1     3D:35:20230930         EA
    # 2        3D:1092:20230930            381.00  [811425, 810752, 810747, 819910, 810641, 81063...  3D:300X            1     3D:35:20230930         AM
    # 3        3D:1093:20230930            438.00  [817028, 812735, 812719, 817899, 814801, 81454...   3D:391            0     3D:51:20230930         AM
    # 4        3D:1094:20230930            542.00  [817754, 813644, 811043, 819910, 810948, 81093...  3D:300X            0     3D:36:20230930         AM

    # First, group without timeperiods to make sure there is one shape_id per route_id and direction_id
    shape_per_route_dir_df = stop_patterns_df.groupby(by=['route_id','direction_id','shape_id'], observed=False).aggregate(
        num_trip_ids      = pd.NamedAgg(column='trip_id',          aggfunc='nunique'),
        trip_ids          = pd.NamedAgg(column='trip_id',          aggfunc=list),
        first_trip_id     = pd.NamedAgg(column='trip_id',          aggfunc='first'),
        trip_depart_time  = pd.NamedAgg(column='trip_depart_time', aggfunc=list),
        stop_pattern      = pd.NamedAgg(column='stop_pattern',     aggfunc='first')
    ).reset_index(drop=False)
    # drop lines with no relationship between shape_id and the stop_pattern
    shape_per_route_dir_df = shape_per_route_dir_df.loc[ shape_per_route_dir_df.stop_pattern.notna() ].reset_index(drop=True)
    WranglerLogger.debug(f"shape_per_route_dir_df:\n{shape_per_route_dir_df}")

    # Check if there are shape_ids that appears for more than one (e.g. one shape_id for 1+ unique route_id, direction_id)
    # This is what showed that 'PE:t263-sl17-p182-r1A:20230930' appears to have the wrong direction_id
    shape_id_multiple_rows = shape_per_route_dir_df.loc[ shape_per_route_dir_df.duplicated(subset='shape_id', keep=False) ]
    if len(shape_id_multiple_rows) > 0:
        WranglerLogger.error(f"shape_id_multiple_rows:\n{shape_id_multiple_rows}")
        # Hopefully there are none!
        assert(len(shape_id_multiple_rows) == 0)

    # Use shape_per_route_dir_df to create a mapping from the new trip_id (shape_id + '_trip') and 
    # the representative (first) orig_trip_id
    new_old_trip_id_df = shape_per_route_dir_df[['shape_id','first_trip_id']].copy()
    new_old_trip_id_df['trip_id'] = new_old_trip_id_df['shape_id'] + '_trip'
    new_old_trip_id_df.rename(columns={'first_trip_id':'orig_trip_id'}, inplace=True)
    new_old_trip_id_df.drop(columns=['shape_id'], inplace=True)
    WranglerLogger.debug(f"new_old_trip_id_df:\n{new_old_trip_id_df}")

    # Now, groupby route_id, direction_id, shape_id, timeperiod and count unique trip_ids, stop_patterns
    shape_patterns_df = stop_patterns_df.groupby(by=['route_id','direction_id','shape_id','timeperiod'], observed=False).aggregate(
        num_trip_ids      = pd.NamedAgg(column='trip_id',          aggfunc='nunique'),
        trip_ids          = pd.NamedAgg(column='trip_id',          aggfunc=list),
        trip_depart_time  = pd.NamedAgg(column='trip_depart_time', aggfunc=lambda x: sorted(list(x))),
        stop_pattern      = pd.NamedAgg(column='stop_pattern',     aggfunc='first')
    ).reset_index(drop=False)
    # WranglerLogger.debug(f"shape_patterns_df:\n{shape_patterns_df}")

    # drop lines with no relationship between shape_id and the stop_pattern
    shape_patterns_df = shape_patterns_df.loc[ shape_patterns_df.stop_pattern.notna() ].reset_index(drop=True)
    WranglerLogger.debug(f"shape_patterns_df:\n{shape_patterns_df}")
    #      route_id direction_id           shape_id timeperiod  num_trip_ids                                           trip_ids        trip_depart_time                                       stop_pattern
    # 0     3D:200X            0     3D:83:20230930         AM             3  [3D:1118:20230930, 3D:1130:20230930, 3D:1166:2...   [445.0, 505.0, 588.0]  [819191, 818955, 819209, 813979, 814568, 81399...
    # 1     3D:200X            0     3D:83:20230930         PM             3  [3D:1100:20230930, 3D:1354:20230930, 3D:1430:2...  [912.0, 972.0, 1032.0]  [819191, 818955, 819209, 813979, 814568, 81399...
    # 2     3D:200X            1     3D:82:20230930         AM             3  [3D:1117:20230930, 3D:1129:20230930, 3D:1165:2...   [380.0, 440.0, 520.0]  [818889, 816287, 816481, 819221, 819213, 81456...
    # 3     3D:200X            1     3D:82:20230930         MD             1                                 [3D:1099:20230930]                 [856.0]  [818889, 816287, 816481, 819221, 819213, 81456...
    # 4     3D:200X            1     3D:82:20230930         PM             2               [3D:1353:20230930, 3D:1429:20230930]          [916.0, 976.0]  [818889, 816287, 816481, 819221, 819213, 81456...
    
    # Determine frequencies for the shape_id, timeperiod
    # if num_trip_ids == 1, use default_frequency_for_onetime_route
    # if num_trip_ids > 1, we can use:
    #    timeperiod duration / num_trip_ids OR
    #    mean time between trip_depart_times OR
    #    median time between trip_depar_times
    # Let's set all of those

    shape_patterns_df['timeperiod_duration_minutes'] = shape_patterns_df['timeperiod'].map(timeperiod_duration_minutes)
    shape_patterns_df['uniform_headway'] = shape_patterns_df['timeperiod_duration_minutes'] / shape_patterns_df['num_trip_ids']

    # Mean of differences (as integers)
    shape_patterns_df['mean_headway'] = shape_patterns_df['trip_depart_time'].apply(
        lambda x: int(np.mean(np.diff(x))) if len(x) > 1 else None
    ) 

    # Median of differences (as integers)
    shape_patterns_df['median_headway'] = shape_patterns_df['trip_depart_time'].apply(
        lambda x: int(np.median(np.diff(x))) if len(x) > 1 else None
    )

    # set it based on frequency_method argument
    shape_patterns_df['headway_mins'] = shape_patterns_df[frequency_method]
    # Make it default_frequency_for_onetime_route if needed
    shape_patterns_df.loc[ shape_patterns_df["num_trip_ids"] <= 1, 'headway_mins'] = default_frequency_for_onetime_route
    WranglerLogger.debug(f"After calculating different versions of headways, shape_patterns_df:\n{shape_patterns_df}")
    #      route_id direction_id           shape_id timeperiod  num_trip_ids                                           trip_ids         trip_depart_time                                       stop_pattern  timeperiod_duration_minutes  uniform_headway  mean_headway  median_headway  headway_mins
    # 0     3D:200X            0     3D:83:20230930         AM             3  [3D:1118:20230930, 3D:1130:20230930, 3D:1166:2...    [445.0, 505.0, 588.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         71.00           71.00         71.00
    # 1     3D:200X            0     3D:83:20230930         PM             3  [3D:1100:20230930, 3D:1354:20230930, 3D:1430:2...   [912.0, 972.0, 1032.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         60.00           60.00         60.00
    # 2     3D:200X            1     3D:82:20230930         AM             3  [3D:1117:20230930, 3D:1129:20230930, 3D:1165:2...    [380.0, 440.0, 520.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00            80.00         70.00           70.00         70.00
    # 3     3D:200X            1     3D:82:20230930         MD             1                                 [3D:1099:20230930]                  [856.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       300.00           300.00           NaN             NaN      10800.00
    # 4     3D:200X            1     3D:82:20230930         PM             2               [3D:1353:20230930, 3D:1429:20230930]           [916.0, 976.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00           120.00         60.00           60.00         60.00
    if trace_shape_ids:
        WranglerLogger.debug(f"trace_shape_ids shape_patterns_df:\n{shape_patterns_df.loc[ shape_patterns_df.shape_id.isin(trace_shape_ids)]}")

    # trip_id is now the same as shape_id (but let's add suffix '_trip')
    # Create feed_tables['frequencies'] compatible with WranglerFrequenciesTable
    timeperiod_start_times = {period: times[0] for period, times in timeperiods.items()}
    timeperiod_end_times   = {period: times[1] for period, times in timeperiods.items()}
    WranglerLogger.debug(f"timeperiod_start_times:\n{timeperiod_start_times}")

    shape_patterns_df['trip_id'] = shape_patterns_df['shape_id'] + '_trip'
    shape_patterns_df['start_time'] = shape_patterns_df['timeperiod'].map(timeperiod_start_times)
    shape_patterns_df['end_time'] = shape_patterns_df['timeperiod'].map(timeperiod_end_times)
    shape_patterns_df['headway_secs'] = shape_patterns_df['headway_mins']*60
    WranglerLogger.debug(f"shape_patterns_df:\n{shape_patterns_df}")
    #      route_id direction_id           shape_id timeperiod  num_trip_ids                                           trip_ids          trip_depart_time                                       stop_pattern  timeperiod_duration_minutes  uniform_headway  mean_headway  median_headway  headway_mins                 trip_id start_time end_time  headway_secs
    # 0     3D:200X            0     3D:83:20230930         AM             3  [3D:1118:20230930, 3D:1130:20230930, 3D:1166:2...     [445.0, 505.0, 588.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         71.00           71.00         71.00     3D:83:20230930_trip      06:00    10:00       4260.00
    # 1     3D:200X            0     3D:83:20230930         PM             3  [3D:1100:20230930, 3D:1354:20230930, 3D:1430:2...    [912.0, 972.0, 1032.0]  [819191, 818955, 819209, 813979, 814568, 81399...                       240.00            80.00         60.00           60.00         60.00     3D:83:20230930_trip      15:00    19:00       3600.00
    # 2     3D:200X            1     3D:82:20230930         AM             3  [3D:1117:20230930, 3D:1129:20230930, 3D:1165:2...     [380.0, 440.0, 520.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00            80.00         70.00           70.00         70.00     3D:82:20230930_trip      06:00    10:00       4200.00
    # 3     3D:200X            1     3D:82:20230930         MD             1                                 [3D:1099:20230930]                   [856.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       300.00           300.00           NaN             NaN      10800.00     3D:82:20230930_trip      10:00    15:00     648000.00
    # 4     3D:200X            1     3D:82:20230930         PM             2               [3D:1353:20230930, 3D:1429:20230930]            [916.0, 976.0]  [818889, 816287, 816481, 819221, 819213, 81456...                       240.00           120.00         60.00           60.00         60.00     3D:82:20230930_trip      15:00    19:00       3600.00
    if trace_shape_ids:
        WranglerLogger.debug(f"trace_shape_ids shape_patterns_df:\n{shape_patterns_df.loc[ shape_patterns_df.shape_id.isin(trace_shape_ids)]}")

    feed_tables['frequencies'] = shape_patterns_df[['trip_id','start_time','end_time','headway_secs']]
    WranglerLogger.debug(f"feed_tables['frequencies']:\n{feed_tables['frequencies']}")

    # Update feed_tables['trips'] to be a WranglerTripsTable
    # Only one trip_id per shape_id, so drop the remaining trip_ids
    feed_tables['trips'].rename(columns={'trip_id':'orig_trip_id'}, inplace=True)
    feed_tables['trips']['trip_id'] = feed_tables['trips']['shape_id'] + '_trip'
    # make trip_id unique
    TRIP_TABLE_COLUMNS = [
        'shape_id','direction_id','service_id','route_id','trip_short_name',
        'trip_headsign','block_id','wheelchair_accessible','bikes_allowed'
    ]
    aggregate_dict = {}
    for colname in TRIP_TABLE_COLUMNS:
        if colname not in feed_tables['trips'].columns: continue
        aggregate_dict[colname       ] = pd.NamedAgg(column=colname, aggfunc='first')
        aggregate_dict[f"{colname}_n"] = pd.NamedAgg(column=colname, aggfunc='nunique')


    feed_tables['trips'] = feed_tables['trips'].groupby('trip_id').agg(**aggregate_dict)
    # move trip_id back to regular column from index
    feed_tables['trips'].reset_index(drop=False, inplace=True)
    WranglerLogger.debug(
        f"After aggregating to new shape_id version of trip_id, "
        f"feed_tables['trips']:\n{feed_tables['trips']}"
    )
    # Log summary
    WranglerLogger.debug(
        f"After aggregating to new shape_id version of trip_id, "
        f"feed_tables['trips'].describe():\n{feed_tables['trips'].describe()}"
    )
    feed_tables['trips'] = feed_tables['trips'][['trip_id'] + TRIP_TABLE_COLUMNS]
    WranglerLogger.debug(f"feed_tables['trips']:\n{feed_tables['trips']}")

    # Update feed_tables['stop_times'] to be a WranglerStopTimesTable
    feed_tables['stop_times'].rename(columns={'trip_id':'orig_trip_id'}, inplace=True)
    feed_tables['stop_times']['trip_id'] = feed_tables['stop_times']['shape_id'] + '_trip'
    WranglerLogger.debug(f"Before trimming, feed_tables['stop_times']:\n{feed_tables['stop_times']}")
    # There are stop_times for every original trip_id but we have fewer now, so filter out the extra
    feed_tables['stop_times'] = pd.merge(
        left=feed_tables['stop_times'],
        right=new_old_trip_id_df, # trip_id, orig_trip_id
        on=['trip_id','orig_trip_id'],
        how='inner',
    )
    # drop columns no long relevant
    feed_tables['stop_times'].drop(columns=[
        'orig_trip_id','arrival_time','departure_time', # not relevant for frequency-based trips
        'departure_minutes'
    ], inplace=True)
    WranglerLogger.debug(f"feed_tables['stop_times']:\n{feed_tables['stop_times']}")


def build_transit_graph(transit_links_df: pd.DataFrame) -> Dict[int, set]:
    """Build directed adjacency graph from transit links.
    
    Creates a dictionary-based graph representation where each node maps to a set of
    directly reachable nodes. The graph is directed, respecting the A->B direction
    of links. Self-loops are not included by default.
    
    Args:
        transit_links_df: DataFrame of transit links with required columns:
            - A (int): Source node ID
            - B (int): Target node ID
        
    Returns:
        Dict[int, set]: Adjacency dictionary where:
            - Keys are node IDs that appear in the links
            - Values are sets of node IDs directly reachable from that node
            - Empty set for nodes with no outgoing edges
    
    Example:
        >>> links_df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 3, 1]})
        >>> graph = build_transit_graph(links_df)
        >>> graph
        {1: {2}, 2: {3}, 3: {1}}
    """
    graph = {}
    for _, link in transit_links_df.iterrows():
        node_a, node_b = link['A'], link['B']
        if node_a not in graph:
            graph[node_a] = set()
        if node_b not in graph:
            graph[node_b] = set()
        # Only add edge from A to B (directed) - no self-loops by default
        graph[node_a].add(node_b)
    return graph

def match_bus_stops_to_roadway_nodes(
    feed_tables: Dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    max_distance: float,
    trace_shape_ids: Optional[List[str]] = None
):
    """Match bus stops to bus-accessible nodes in the roadway network.
    
    Matches bus and trolleybus stops to the nearest bus-accessible nodes in the roadway
    network using spatial proximity. Updates stop and shape locations to snap to road nodes.
    
    Process Steps:
    1. Identifies bus stops (route_types BUS or TROLLEYBUS) in feed_tables['stops']
    2. Builds bus network graph from roadway to find accessible nodes
    3. Projects geometries to local CRS for accurate distance calculations
    4. Uses BallTree spatial index to find nearest bus-accessible node for each stop
    5. Updates stop locations to matched road node locations (if within max_distance)
    6. Updates shape point locations for matched bus stops
    
    Modifies feed_tables in place:
    
    feed_tables['stops'] - Adds/modifies columns:
        - is_bus_stop (bool): True if stop serves BUS or TROLLEYBUS routes
        - model_node_id (int): Matched roadway node ID (None if no valid match)
        - match_distance_{crs_units} (float): Distance to matched node
        - valid_match (bool): True if match found within max_distance
        - stop_lon, stop_lat, geometry: Updated to road node location if valid_match
    
    feed_tables['shapes'] - Adds/modifies columns:
        - shape_model_node_id (int): Matched roadway node ID for bus stops
        - match_distance_{crs_units} (float): Distance to matched node
        - shape_pt_lon, shape_pt_lat, geometry: Updated to road node location if valid match
    
    feed_tables['stop_times'] - If GeoDataFrame, updates:
        - geometry: Updated to matched road node location for bus stops
    
    Args:
        feed_tables: Dictionary of GTFS feed tables. Expects:
            - 'stops': Must have route_types column (list of RouteType enums)
            - 'shapes': Shape points to update
            - 'stop_times': Optional, updated if present as GeoDataFrame
        roadway_net: RoadwayNetwork with nodes to match against.
            Will be converted to GeoDataFrame if needed.
        local_crs: Coordinate reference system for projections (e.g., "EPSG:2227")
        crs_units: Distance units for local_crs ('feet' or 'meters')
        max_distance: Maximum matching distance in crs_units
        trace_shape_ids: Optional list of shape_ids for debug logging
        
    Raises:
        TransitValidationError: If no bus-accessible nodes found near any bus stops
        
    Notes:
        - Only matches stops that serve BUS or TROLLEYBUS routes
        - Uses bus modal graph to ensure matched nodes are bus-accessible
        - Preserves original locations for non-bus stops
    """
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    # Make roadway network nodes a GeoDataFrame if it's not already
    if not isinstance(roadway_net.nodes_df, gpd.GeoDataFrame):
        # Convert to GeoDataFrame if needed
        roadway_net.nodes_df = gpd.GeoDataFrame(
            roadway_net.nodes_df, 
            geometry=gpd.points_from_xy(roadway_net.nodes_df.X, roadway_net.nodes_df.Y),
            crs=LAT_LON_CRS
        )

    # Collect bus stops to match
    feed_tables["stops"]["is_bus_stop"] = False
    feed_tables["stops"].loc[ 
        feed_tables["stops"]['route_types'].apply(lambda x: RouteType.BUS in x if isinstance(x, list) else False) | 
        feed_tables["stops"]['route_types'].apply(lambda x: RouteType.TROLLEYBUS in x if isinstance(x, list) else False),
        "is_bus_stop"] = True
    WranglerLogger.debug(f"feed_tables['stops'].value_counts():\n{feed_tables['stops']}")
    bus_stops_gdf = feed_tables["stops"].loc[ feed_tables["stops"]["is_bus_stop"] == True ].copy(deep=True)
    
    # Reset index to ensure continuous indices from 0 to n-1
    bus_stops_gdf.reset_index(drop=True, inplace=True)
    
    WranglerLogger.info(f"Matching {len(bus_stops_gdf):,} bus stops to roadway nodes")
    WranglerLogger.debug(f"bus_stops_gdf:\n{bus_stops_gdf}")
    WranglerLogger.debug(f"feed_tables['shapes']:\n{feed_tables['shapes']}")
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace shapes for {trace_shape_id}:\n"
                f"{feed_tables['shapes'].loc[ feed_tables['shapes']['shape_id']==trace_shape_id]}"
            )

    # Build BallTree for bus-access nodes that are in the bus graph
    # This ensures we only match to connected nodes
    G_bus = roadway_net.get_modal_graph('bus')
    bus_graph_nodes = set(G_bus.nodes())
    roadway_net.nodes_df["bus_access"] = False
    roadway_net.nodes_df.loc[
        roadway_net.nodes_df['model_node_id'].isin(bus_graph_nodes),
        'bus_access'] = True
    WranglerLogger.debug(f"bus-accessible nodes from graph: {roadway_net.nodes_df['bus_access'].sum():,}")

    # Extract bus-accessible nodes
    bus_accessible_nodes_gdf = roadway_net.nodes_df[roadway_net.nodes_df["bus_access"] == True].copy()
    WranglerLogger.debug(
        f"Found {len(bus_accessible_nodes_gdf):,} bus-accessible nodes (for mixed-traffic transit) "
        f"out of {len(roadway_net.nodes_df):,} total"
    )
    
    # Project bus nodes and bus_stops_gdf to specified CRS and bus_stops
    bus_accessible_nodes_gdf.to_crs(local_crs, inplace=True)
    bus_stops_gdf.to_crs(local_crs, inplace=True)

    # Initialize results
    bus_stops_gdf["model_node_id"] = None
    bus_stops_gdf[f"match_distance_{crs_units}"] = np.inf # in crs_units
    
    from sklearn.neighbors import BallTree

    # Build spatial index for bus nodes
    bus_node_coords = np.array([(geom.x, geom.y) for geom in bus_accessible_nodes_gdf.geometry])
    bus_nodes_tree = BallTree(bus_node_coords)
    WranglerLogger.debug(f"Created BallTree for {len(bus_node_coords):,} bus nodes")
    
    # Get coordinates of stops to match
    bus_stop_coords = np.array([(geom.x, geom.y) for geom in bus_stops_gdf.geometry])
    
    # Query nearest neighbors
    match_distances, match_indices = bus_nodes_tree.query(bus_stop_coords, k=1)
    # Create a matches dataframe
    matches_df = pd.DataFrame({
        'stop_idx': bus_stops_gdf.index,  # Use actual index from bus_stops_gdf
        # Flatten the arrays since k=1
        'match_node_idx': match_indices.flatten(), 
        'match_distance':  match_distances.flatten()
    })
    
    # Check for valid matches
    matches_df['valid_match'] = False
    matches_df.loc[matches_df['match_distance'] <= max_distance, 'valid_match'] = True
    WranglerLogger.debug(f"matches_df:\n{matches_df}")

    WranglerLogger.info(f"Found {matches_df.valid_match.sum():,} valid matches out of {len(bus_stops_gdf):,} total bus stops")

    if matches_df.valid_match.sum() == 0:
        exception = TransitValidationError("Found no bus-accessible nodes near bus stops.")
        raise exception
    
    # Get matched node information
    matched_nodes_gdf = bus_accessible_nodes_gdf.iloc[matches_df['match_node_idx']]
    WranglerLogger.debug(f"matched_nodes_gdf:\n{matched_nodes_gdf}")
    
    # Update bus stops with matched node information (vectorized)
    # Update all matches with their corresponding node information
    bus_stops_gdf.loc[matches_df['stop_idx'], f"match_distance_{crs_units}"] = matches_df['match_distance'].values
    bus_stops_gdf.loc[matches_df['stop_idx'], "valid_match"]                 = matches_df["valid_match"].values
    bus_stops_gdf.loc[matches_df['stop_idx'], "model_node_id"]               = matched_nodes_gdf["model_node_id"].values
    bus_stops_gdf.loc[matches_df['stop_idx'], "stop_lon"]                    = matched_nodes_gdf["X"].values
    bus_stops_gdf.loc[matches_df['stop_idx'], "stop_lat"]                    = matched_nodes_gdf["Y"].values
    bus_stops_gdf.loc[matches_df['stop_idx'], "geometry"]                    = matched_nodes_gdf["geometry"].values

    WranglerLogger.debug(f"bus_stops_gdf:\n{bus_stops_gdf}")
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace bus_stops_gdf for {trace_shape_id}:\n"
                f"{bus_stops_gdf.loc[ bus_stops_gdf['shape_ids'].apply(lambda x: trace_shape_id in x) ]}"
            )
    
    # verify model_node_id, f'match_distance_{crs_units}' and 'valid_match' are not in feed_tables['stops']
    assert('model_node_id' not in feed_tables['stops'].columns)
    assert(f'match_distance_{crs_units}' not in feed_tables['stops'].columns)
    assert('valid_match' not in feed_tables['stops'].columns)

    # Update feed_tables['stops'] by merging the updates
    feed_tables['stops'] = feed_tables['stops'].merge(
        bus_stops_gdf[['stop_id','model_node_id',f'match_distance_{crs_units}', 'stop_lon', 'stop_lat', 'geometry', 'valid_match']],
        on='stop_id',
        how='left',
        suffixes=('', '_bus'),
        validate='one_to_one'
    )
    # Only update stop location for valid match
    feed_tables['stops'].loc[ feed_tables['stops']['valid_match'] == True, 'stop_lon'] = feed_tables['stops']['stop_lon_bus']
    feed_tables['stops'].loc[ feed_tables['stops']['valid_match'] == True, 'stop_lat'] = feed_tables['stops']['stop_lat_bus']
    feed_tables['stops'].loc[ feed_tables['stops']['valid_match'] == True, 'geometry'] = feed_tables['stops']['geometry_bus']
    # Drop bus-specific columns
    feed_tables['stops'].drop(columns=['stop_lon_bus','stop_lat_bus','geometry_bus'], inplace=True)

    WranglerLogger.debug(f"After merging with bus_stops_gdf, feed_tables['stops']:\n{feed_tables['stops']}")
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['stops'] for {trace_shape_id}:\n"
                f"{feed_tables['stops'].loc[ feed_tables['stops']['shape_ids'].apply(lambda x: isinstance(x, list) and trace_shape_id in x) ]}"
            )
    
    # summary of match distances
    WranglerLogger.debug(f"\n{feed_tables['stops'][['valid_match',f'match_distance_{crs_units}']].describe()}")

    # Update shapes table similarly
    WranglerLogger.debug(f"feed_tables['shapes']:\n{feed_tables['shapes']}")
    # columns: shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled, geometry
    #   trip_id, direction_id, route_id, agency_id, route_short_name, route_type, agency_name,
    #   match_distance_feet, stop_id, stop_name, stop_sequence
    # Note: this is adding: 'model_node_id'
    feed_tables['shapes'] = pd.merge(
        left=feed_tables['shapes'],
        right=bus_stops_gdf[['stop_id','model_node_id',f'match_distance_{crs_units}', 'stop_lon', 'stop_lat', 'geometry', 'valid_match']],
        on='stop_id',
        how='left',
        suffixes=('', '_bus'),
        validate='many_to_one'
    ).rename(columns={'model_node_id':'shape_model_node_id'})
    # Only update stop location for valid match
    feed_tables['shapes'].loc[ feed_tables['shapes']['valid_match'] == True, 'shape_pt_lon'] = feed_tables['shapes']['stop_lon']
    feed_tables['shapes'].loc[ feed_tables['shapes']['valid_match'] == True, 'shape_pt_lat'] = feed_tables['shapes']['stop_lat']
    feed_tables['shapes'].loc[ feed_tables['shapes']['valid_match'] == True, 'geometry'] = feed_tables['shapes']['geometry_bus']
    feed_tables['shapes'].loc[ feed_tables['shapes']['valid_match'] == True, f'match_distance_{crs_units}'] = \
        feed_tables['shapes'][f'match_distance_{crs_units}_bus']

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['shapes'] for {trace_shape_id} at the end of match_bus_stops_to_roadway_nodes:\n"
                f"{feed_tables['shapes'].loc[ feed_tables['shapes']['shape_id']==trace_shape_id]}"
            )

    # Drop bus-specific columns
    feed_tables['shapes'].drop(columns=['stop_lon','stop_lat','geometry_bus',f'match_distance_{crs_units}_bus','valid_match'], inplace=True)

    # update geometry of feed_tables['stop_times'] if needed
    if isinstance(feed_tables['stop_times'], gpd.GeoDataFrame):
        feed_tables['stop_times'] = pd.merge(
                left=feed_tables['stop_times'],
                right=bus_stops_gdf[['stop_id','geometry']],
                how='left',
                on='stop_id',
                validate='many_to_one',
                suffixes=('', '_bus'),
                indicator=True
            )
        WranglerLogger.debug(f"Updating feed['stop_times'] stop locations for bus stops:\n{feed_tables['stop_times']._merge.value_counts()}")
        feed_tables['stop_times'].loc[ feed_tables['stop_times']['geometry_bus'].notna(), 'geometry'] = feed_tables['stop_times']['geometry_bus']
        feed_tables['stop_times'].drop(columns=['geometry_bus','_merge'], inplace=True)
    return

def build_shape_links_gdf(
    shapes_df: pd.DataFrame,
    roadway_links_df: pd.DataFrame,
    include_merge_indicator: bool = True
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of shape segment links from consecutive shape points.
    
    Creates LineString geometries connecting consecutive shape points and identifies
    which segments correspond to actual roadway links. Filters out self-loops where
    consecutive points have the same node ID.
    
    Process Steps:
    1. Sorts shape points by shape_id and shape_pt_sequence
    2. Groups consecutive points within each shape using shift operations
    3. Filters out self-loops (where shape_model_node_id == next_node)
    4. Creates LineString geometries for each segment
    5. Left joins with roadway links to identify valid road segments
    
    Args:
        shapes_df: DataFrame with shape points, required columns:
            - shape_id (str): Shape identifier
            - shape_pt_sequence (int): Order of points in shape
            - shape_model_node_id (int): Roadway node ID for this shape point
            - shape_pt_lat (float): Latitude of shape point
            - shape_pt_lon (float): Longitude of shape point
        roadway_links_df: DataFrame of roadway links, required columns:
            - A (int): Source node ID
            - B (int): Target node ID
            - model_link_id (int): Unique link identifier
        include_merge_indicator: If True, includes _merge column from join
            
    Returns:
        GeoDataFrame with one row per shape segment:
            - All columns from input shapes_df
            - next_node (int): Next node ID in sequence
            - next_pt_lat (float): Latitude of next point
            - next_pt_lon (float): Longitude of next point
            - A (int): Source node ID (copy of shape_model_node_id)
            - B (int): Target node ID (copy of next_node)
            - model_link_id (float): Road link ID if segment matches, NaN otherwise
            - _merge (str): Join result ('both', 'left_only') if requested
            - geometry: LineString connecting consecutive points in LAT_LON_CRS
            
    Notes:
        - Last point of each shape is excluded (no next point to connect to)
        - Self-loops are automatically filtered out
        - Segments not matching roadway links will have NaN model_link_id
    """
    # Sort shapes by shape_id and sequence
    shape_links_df = shapes_df.sort_values(['shape_id', 'shape_pt_sequence']).copy()
    
    # Create consecutive node pairs for each shape
    shape_links_df['next_node'] = shape_links_df.groupby('shape_id')['shape_model_node_id'].shift(-1)
    shape_links_df['next_pt_lat'] = shape_links_df.groupby('shape_id')['shape_pt_lat'].shift(-1)
    shape_links_df['next_pt_lon'] = shape_links_df.groupby('shape_id')['shape_pt_lon'].shift(-1)
    
    # Filter to only rows that have a next node (excludes last point of each shape)
    # and filter out self-loops where shape_model_node_id == next_node
    has_next = shape_links_df['next_node'].notna()
    is_self_loop = shape_links_df['shape_model_node_id'] == shape_links_df['next_node']
    
    num_self_loops = (has_next & is_self_loop).sum()
    if num_self_loops > 0:
        WranglerLogger.debug(f"Filtering out {num_self_loops:,} self-loop segments where shape_model_node_id == next_node")
    
    shape_links_df = shape_links_df[has_next & ~is_self_loop]
    shape_links_df['A'] = shape_links_df['shape_model_node_id']
    shape_links_df['B'] = shape_links_df['next_node'].astype(int)
    
    # Create LineString geometries for each segment
    segment_lines = []
    for _, row in shape_links_df.iterrows():
        line = shapely.geometry.LineString([
            (row['shape_pt_lon'], row['shape_pt_lat']),
            (row['next_pt_lon'], row['next_pt_lat'])
        ])
        segment_lines.append(line)
    
    # Convert to GeoDataFrame with LAT_LON_CRS
    shape_links_gdf = gpd.GeoDataFrame(
        shape_links_df,
        geometry=segment_lines,
        crs=LAT_LON_CRS
    )
    
    # Left join shape segments with roadway links to find invalid segments
    shape_links_gdf = pd.merge(
        shape_links_gdf,
        roadway_links_df[['A','B','model_link_id']],
        on=['A', 'B'],
        how='left',
        validate="many_to_one",
        suffixes=('','_rd'),
        indicator=include_merge_indicator
    )
    
    # Ensure the result is still a GeoDataFrame with correct CRS
    shape_links_gdf = gpd.GeoDataFrame(shape_links_gdf, crs=LAT_LON_CRS)
    
    return shape_links_gdf


def create_bus_routes(
    bus_stop_links_gdf: gpd.GeoDataFrame,
    feed_tables: Dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    trace_shape_ids: Optional[List[str]] = None
):
    """Find shortest paths through the bus network between consecutive bus stops.
    
    Replaces original bus route shapes with new shapes that follow the actual bus network
    by finding shortest paths between consecutive stops through bus-accessible roads.
    
    Process Steps:
    1. Sorts bus stop links by shape_id and stop_sequence
    2. Gets bus modal graph from roadway network
    3. For each consecutive stop pair in each shape:
       - Finds shortest path through bus network using NetworkX
       - Creates shape points for all nodes in the path
       - Preserves stop information at stop nodes
    4. Replaces bus shapes in feed_tables['shapes'] with new routed shapes
    
    Modifies feed_tables['shapes'] in place:
    - Removes existing bus/trolleybus shapes
    - Adds new shapes with points following road network paths
    - Each shape point has shape_model_node_id from roadway network
    - Stop points retain stop_id, stop_name, stop_sequence
    
    Args:
        bus_stop_links_gdf: GeoDataFrame of bus stop pairs, required columns:
            - shape_id (str): Shape identifier  
            - stop_sequence (int): Stop order in route
            - stop_id (str): Current stop ID
            - stop_name (str): Current stop name
            - next_stop_id (str): Next stop ID
            - next_stop_name (str): Next stop name
            - A (int): Current stop's model_node_id
            - B (int): Next stop's model_node_id
            - route_id, route_type, trip_id, direction_id: Route metadata
        feed_tables: Dictionary with required tables:
            - 'stops': Must have is_bus_stop column
            - 'shapes': Will be modified to replace bus shapes
        roadway_net: RoadwayNetwork with bus modal graph
        local_crs: Coordinate reference system for projections
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional shape IDs for debug logging
    
    Raises:
        TransitValidationError: If no path exists between any consecutive stops.
            Exception includes no_bus_path_gdf with failed stop sequences.
            
    Notes:
        - Uses NetworkX shortest_path for routing
        - Intermediate nodes between stops are added as shape points
        - Original shape geometry is replaced with routed paths
    """
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    WranglerLogger.info(f"Creating bus routes between bus stops")
    WranglerLogger.debug(f"bus stops:\n{feed_tables['stops'].loc[ feed_tables['stops']['is_bus_stop'] == True]}")
    WranglerLogger.debug(f"bus shapes:\n{feed_tables['shapes'].loc[ feed_tables['shapes']['route_type'].isin([RouteType.BUS, RouteType.TROLLEYBUS]) ]}")
    WranglerLogger.debug(f"bus_stop_links_gdf:\n{bus_stop_links_gdf}")
    if trace_shape_ids:
        WranglerLogger.debug(f"trace bus_stop_links_gdf:\n{bus_stop_links_gdf.loc[ bus_stop_links_gdf.shape_id.isin(trace_shape_ids)]}")

    # Create a shortest path through the bus network between each consecutive bus stop for a given shape_id,
    # traversing through intermediate roadway nodes
    # Create new shape links between those nodes
    # Check how far away the new shape links are from the given stop-to-stop shape links
    bus_stop_links_gdf.sort_values(by=['shape_id','stop_sequence'], inplace=True)
    G_bus = roadway_net.get_modal_graph('bus')
    
    # collect node sequences for these shapes
    bus_node_sequence = []
    # also collect failed stop sequences
    no_path_sequence = []

    current_shape_id = None
    current_shape_pt_sequence = None
    for idx, row in bus_stop_links_gdf.iterrows():
        # restart for each shape_id
        if current_shape_id != row['shape_id']:
            current_shape_pt_sequence = 1
            current_shape_id = row['shape_id']

        # WranglerLogger.debug(f"{idx} Looking for path from {row['A']} to {row['B']}")
        # WranglerLogger.debug(f"{row}")
        try:
            # Try to find shortest path
            path = nx.shortest_path(G_bus, row['A'], row['B'])
            path_length = len(path) - 1  # Number of edges
            # WranglerLogger.debug(f"Found path for {row['A']} to {row['B']}: len={path_length} {path}")

            # Create shape point rows for that path
            for path_node_id in path:
                bus_node_dict = {
                    'shape_id'            : row['shape_id'],
                    'route_id'            : row['route_id'],
                    'route_type'          : row['route_type'],
                    'trip_id'             : row['trip_id'],
                    'direction_id'        : row['direction_id'],
                    'shape_pt_sequence'   : current_shape_pt_sequence,
                    'shape_model_node_id' : path_node_id,
                }
                # set these for the stops but leave blank for intermediate nodes
                if path_node_id == row['A']:
                    bus_node_dict['stop_id']       = row['stop_id']
                    bus_node_dict['stop_name']     = row['stop_name']
                    bus_node_dict['stop_sequence'] = row['stop_sequence']
                elif path_node_id == row['B']:
                    bus_node_dict['stop_id']       = row['next_stop_id']
                    bus_node_dict['stop_name']     = row['next_stop_name']
                    bus_node_dict['stop_sequence'] = last_stop_sequence + 1

                bus_node_sequence.append(bus_node_dict)
                current_shape_pt_sequence += 1
                last_stop_sequence = row['stop_sequence']

        except nx.NetworkXNoPath as e:
            WranglerLogger.warning(f"No path exists from {row['A']} to {row['B']}")
            WranglerLogger.warning(e)
            # No path exists
            no_path_sequence.append({
                'shape_id'      : row['shape_id'],
                'stop_id'       : row['stop_id'],
                'next_stop_id'  : row['next_stop_id'],
                'stop_sequence' : row['stop_sequence'],
            })
            pass
        except nx.NodeNotFound as e:
            WranglerLogger.fatal(f"No node found: {e}")
            pass

    bus_node_sequence_df = pd.DataFrame(bus_node_sequence)
    WranglerLogger.debug(f"bus_node_sequence_df:\n{bus_node_sequence_df}")

    if len(no_path_sequence) == 0:
        WranglerLogger.info(f"All bus route shapes mapped to roadway nodes")
    else:
        no_path_sequence_df = pd.DataFrame(no_path_sequence)
        WranglerLogger.debug(f"no_path_sequence_df:\n{no_path_sequence_df}")

        # join with bus_stop_links_gdf
        no_bus_path_gdf = gpd.GeoDataFrame(
            pd.merge(
                left=no_path_sequence_df,
                right=bus_stop_links_gdf,
                how='left',
                on=['stop_id','next_stop_id','stop_sequence','shape_id'],
                validate='one_to_one'
            ), crs=bus_stop_links_gdf.crs)
        WranglerLogger.debug(f"no_bus_path_gdf:\n{no_bus_path_gdf}")
        e = TransitValidationError("Some bus stop sequences failed to find paths. See e.no_bus_path_gdf")
        e.no_bus_path_gdf = no_bus_path_gdf
        raise e
    
    # create bus shapes
    # current shapes columns:
    #  shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled, geometry, 
    #  trip_id, direction_id, route_id, agency_id, route_short_name, route_type, agency_name, match_distance_feet,
    #  stop_id, stop_name, stop_sequence, shape_model_node_id

    # we have:
    #  shape_id, route_id, route_type, trip_id, direction_id, shape_pt_sequence, shape_model_node_id, stop_id, stop_name

    # Reorder to be similar
    bus_node_sequence_df = bus_node_sequence_df[[
        'shape_id', 'shape_pt_sequence',
        'trip_id', 'direction_id', 'route_id', 'route_type',
        'stop_id', 'stop_name', 'stop_sequence', 'shape_model_node_id' ]]
    # get agency_id, route_short_name, agency_name from existing feed_tables['shapes']
    bus_node_sequence_df = pd.merge(
        left=bus_node_sequence_df,
        right=feed_tables['shapes'][['shape_id','agency_id','route_short_name','agency_name']].drop_duplicates(),
        on='shape_id',
        how='left',
        validate='many_to_one'
    )

    # get lon, lat and geometry from roadway_net.nodes
    bus_node_sequence_gdf = gpd.GeoDataFrame(
        pd.merge(
            left=bus_node_sequence_df,
            right=roadway_net.nodes_df[['model_node_id','X','Y','geometry']].rename(columns={'model_node_id':'shape_model_node_id'}),
            how='left',
            on='shape_model_node_id',
            validate='many_to_one'
        ).rename(columns={'X':'shape_pt_lon', 'Y':'shape_pt_lat'}),
        crs=roadway_net.nodes_df.crs
    )
    WranglerLogger.debug(f"Final bus_node_sequence_gdf:\n{bus_node_sequence_gdf}")

    feed_tables['shapes'].to_crs(LAT_LON_CRS, inplace=True)
    # replace bus links in feed_tables['shapes'] with bus_node_sequence_gdf
    feed_tables['shapes'] = pd.concat([
        feed_tables['shapes'].loc[ ~feed_tables['shapes']['route_type'].isin([RouteType.BUS, RouteType.TROLLEYBUS]) ],
        bus_node_sequence_gdf
    ])

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['shapes'] for {trace_shape_id} at the end of create_bus_routes():\n"
                f"{feed_tables['shapes'].loc[ feed_tables['shapes']['shape_id']==trace_shape_id]}"
            )



def add_additional_data_to_stops(
    feed_tables: Dict[str, pd.DataFrame],
):
    """Updates feed_tables['stops'] with additional metadata about routes and agencies.
    
    Aggregates information from stop_times, trips, routes, and agencies tables to add
    comprehensive metadata about which routes and agencies serve each stop.
    
    Process Steps:
    1. Joins stop_times with trips to get route and shape information
    2. Joins with routes and agencies to get route types and agency names
    3. Groups by stop_id to aggregate all serving routes/agencies
    4. Identifies mixed-traffic stops based on route types
    5. Handles parent stations by checking child stop characteristics
    
    Modifies feed_tables['stops'] in place, adding columns:
    
    Route/Agency Information:
    - agency_ids (list of str): All agencies serving this stop
    - agency_names (list of str): Names of agencies serving this stop
    - route_ids (list of str): All routes serving this stop
    - route_names (list of str): Short names of routes serving this stop
    - route_types (list of int): Types of routes serving this stop
    - shape_ids (list of str): All shapes associated with this stop
    
    Stop Type Flags:
    - stop_in_mixed_traffic (bool): True if stop serves any MIXED_TRAFFIC_ONLY_ROUTE_TYPES
        (currently only BUS and TROLLEYBUS)
    - is_parent (bool): True if other stops reference this as parent_station
    - child_stop_in_mixed_traffic (bool): True if any child stop has stop_in_mixed_traffic=True
        (only set for parent stations)
    
    Args:
        feed_tables: Dictionary with required tables:
            - 'stop_times': Links stops to trips
            - 'trips': Links trips to routes and shapes
            - 'routes': Route information including type
            - 'agencies': Agency names
            - 'stops': Table to be updated
            
    Notes:
        - Parent stations may not appear in trips but are retained if referenced
        - Empty lists are used for stops with no associated routes
        - Handles missing parent_station column gracefully
    """

    # Add information about agencies, routes, directions and shapes to stops
    # Join stop_times with trips and routes
    stop_trips = pd.merge(
        feed_tables["stop_times"][["stop_id", "trip_id"]].drop_duplicates(),
        feed_tables["trips"][["trip_id", "direction_id", "route_id", "shape_id"]],
        on="trip_id",
        how="left",
    )
    WranglerLogger.debug(f"After joining stop_times with trips: {len(stop_trips):,} records")

    # Create stop to route, agency mapping with direction information
    stop_agencies = pd.merge(
        stop_trips, 
        feed_tables["routes"][["route_id", "agency_id", "route_short_name", "route_type"]], 
        on="route_id", 
        how="left"
    )[["stop_id", "agency_id", "route_id", "direction_id", "shape_id", "route_short_name","route_type"]].drop_duplicates()
    WranglerLogger.debug(f"stop_agencies.head():\n{stop_agencies.head()}")

    # pick up agency information
    stop_agencies = pd.merge(
        stop_agencies,
        feed_tables["agencies"][["agency_id", "agency_name"]],
        on="agency_id",
        how="left"
    )
    WranglerLogger.debug(f"stop_agencies.head():\n{stop_agencies.head()}")
    
    # Group by stop to get all agencies and routes serving each stop
    # Now including route_dir_ids as list of (route_id, direction_id) tuples
    stop_agency_info = stop_agencies.groupby("stop_id").agg({
        "agency_id": lambda x: list(x.dropna().unique()),
        "agency_name": lambda x: list(x.dropna().unique()) if x.notna().any() else [],
        "route_id": lambda x: list(x.dropna().unique()),
        "route_short_name": lambda x: list(x.dropna().unique()),
        "route_type": lambda x: list(x.dropna().unique()),
        "shape_id": lambda x: list(x.dropna().unique()),
    }).reset_index()
    
    stop_agency_info.columns = ["stop_id", "agency_ids", "agency_names", "route_ids", "route_names", "route_types", "shape_ids"]
    stop_agency_info["stop_in_mixed_traffic"] = stop_agency_info["route_types"].apply(
        lambda x: any(rt in MIXED_TRAFFIC_ONLY_ROUTE_TYPES for rt in x) if x else False
    )  # A stop is in mixed traffic if it serves MIXED_TRAFFIC_ONLY_ROUTE_TYPES

    # columns: stop_id (str), agency_ids (list of str), agency_names (list of str), 
    #   route_ids (list of str), route_names (list of str), route_types (list of int), 
    #   shape_ids (list of str), stop_in_mixed_traffic (bool)
    WranglerLogger.debug(f"stop_agency_info.head():\n{stop_agency_info.head()}")

    # Merge this information back to stops
    feed_tables["stops"] = pd.merge(
        feed_tables["stops"], stop_agency_info, on="stop_id", how="left"
    )

    # Handle parent stations that may not be included in trips
    # For these, set child_stop_in_mixed_traffic if any of the child stops has stop_in_mixed_traffic == True
    if "parent_station" in feed_tables["stops"].columns:
        # Create a subset of child stops with their parent_station and stop_in_mixed_traffic status
        child_stops = feed_tables["stops"][feed_tables["stops"]["parent_station"].notna() & (feed_tables["stops"]["parent_station"] != "")][
            ["parent_station", "stop_in_mixed_traffic"]
        ]
        
        if len(child_stops) > 0:
            # Group by parent_station and check if any child is in mixed traffic
            parent_mixed_traffic = child_stops.groupby("parent_station")["stop_in_mixed_traffic"].any().reset_index()
            parent_mixed_traffic.columns = ["stop_id", "child_stop_in_mixed_traffic"]
            parent_mixed_traffic["is_parent"] = True
            WranglerLogger.debug(f"parent_mixed_traffic:\n{parent_mixed_traffic}")
            
            # Merge back to the main dataframe
            feed_tables["stops"] = feed_tables["stops"].merge(
                parent_mixed_traffic,
                on="stop_id",
                how="left"
            )
            
            # Fill NaN values with False (stops that are not parent stations)
            # Suppress downcasting warning for fillna
            with pd.option_context('future.no_silent_downcasting', True):
                feed_tables["stops"]["is_parent"] = feed_tables["stops"]["is_parent"].fillna(value=False).astype(bool)
                feed_tables["stops"]["child_stop_in_mixed_traffic"] = feed_tables["stops"]["child_stop_in_mixed_traffic"].fillna(value=False).astype(bool)
            
            # Log parent stations with child stops in mixed traffic
            WranglerLogger.debug(
                f"Parent stations with mixed traffic children:\n" + \
                f"{feed_tables['stops'].loc[feed_tables['stops']['child_stop_in_mixed_traffic'] == True, ['stop_id', 'stop_name', 'child_stop_in_mixed_traffic']]}"
            )
    else:
        feed_tables["stops"]["is_parent"] = False
        feed_tables["stops"]["child_stop_in_mixed_traffic"] = False
    
    WranglerLogger.debug(f"add_additional_data_to_stops() completed. feed_tables['stops']:\n{feed_tables['stops']}")

def add_additional_data_to_shapes(
    feed_tables: Dict[str, pd.DataFrame],
    local_crs: str,
    crs_units: str,
    trace_shape_ids: Optional[List[str]] = None
):
    """Updates feed_tables['shapes'] with route/trip metadata and snaps shape points to stops.
    
    Enriches shape points with information from trips, routes, and agencies tables,
    then matches shape points to nearby stops and updates their locations.
    
    Process Steps:
    1. Converts shapes to GeoDataFrame if needed (using shape_pt_lon/lat)
    2. Joins with trips, routes, and agencies to add metadata
    3. Projects to local CRS for distance calculations
    4. For each shape, finds nearest shape point to each stop
    5. Updates matched shape points to stop locations
    6. Records match distance for quality assessment
    
    Assumes create_feed_frequencies() has already run, so each shape corresponds
    to one consolidated trip_id.
    
    Modifies feed_tables in place:
    
    feed_tables['shapes'] - Adds/modifies columns:
        Route/Trip Metadata:
        - trip_id (str): Associated trip ID
        - direction_id (int): Direction of travel (0 or 1)
        - route_id (str): Route identifier
        - agency_id (str): Agency identifier
        - agency_name (str): Agency name
        - route_short_name (str): Route short name
        - route_type (int): GTFS route type
        
        Stop Matching (for matched points only):
        - stop_id (str): Matched stop ID
        - stop_name (str): Matched stop name  
        - stop_sequence (int): Order of stop in trip
        - match_distance_{crs_units} (float): Distance from original to stop location
        - shape_pt_lon, shape_pt_lat: Updated to stop coordinates
        - geometry: Updated to stop location
    
    feed_tables['stop_times'] - Converted to GeoDataFrame with:
        - geometry: Stop location added from stops table
    
    Args:
        feed_tables: Dictionary with required tables:
            - 'shapes': Shape points to update
            - 'trips': Trip information
            - 'routes': Route information
            - 'agencies': Agency information
            - 'stops': Stop locations
            - 'stop_times': Stop sequences
        local_crs: Coordinate reference system for projections
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional shape IDs for debug logging
        
    Notes:
        - Uses BallTree for efficient nearest neighbor search
        - Only updates closest shape point to each stop
        - Preserves original locations for unmatched shape points
    """
    # Step 1: Convert shapes to GeoDataFrame if needed and add geometry from lat/lon coordinates
    # Create GeoDataFrame from shape points if not already one
    if not isinstance(feed_tables['shapes'], gpd.GeoDataFrame):
        shape_geometry = [
            shapely.geometry.Point(lon, lat) for lon, lat in \
                zip(feed_tables['shapes']["shape_pt_lon"], feed_tables['shapes']["shape_pt_lat"])
        ]
        feed_tables['shapes'] = gpd.GeoDataFrame(feed_tables['shapes'], geometry=shape_geometry, crs=LAT_LON_CRS)
        WranglerLogger.debug(f"Converted feed_tables['shapes'] to GeoDataFrame")
    else:
        WranglerLogger.debug(f"feed_tables['shapes'].crs={feed_tables['shapes'].crs}")

    # Step 2: Add agency, route, and trip information to shapes by joining with trips and routes tables
    # Get unique shape_ids to trips mapping
    trip_shapes_df = feed_tables['trips'][['shape_id', 'trip_id', 'direction_id', 'route_id']].drop_duplicates()
    WranglerLogger.debug(f"trip_shapes_df\n{trip_shapes_df}")
    # assumes trip_id and shape_id are equivalent due to add_additional_data_to_shapes
    assert(feed_tables['trips']['trip_id'].nunique() == feed_tables['trips']['shape_id'].nunique())
    
    # Get route information: Join routes with agencies to get agency names
    routes_with_agency_df = pd.merge(
        feed_tables['routes'][['route_id', 'agency_id', 'route_short_name', 'route_type']],
        feed_tables['agencies'][['agency_id', 'agency_name']],
        on='agency_id',
        how='left'
    )
    # Add agency information to trips_shapes_df
    trip_shapes_df = pd.merge(
        left=trip_shapes_df,
        right=routes_with_agency_df,
        how='left',
        on='route_id'
    )
    # Add this data to shapes table
    feed_tables['shapes'] = pd.merge(
        feed_tables['shapes'],
        trip_shapes_df,
        on='shape_id',
        how='left',
        validate='many_to_one'
    )

    WranglerLogger.debug(f"Added agency and route information to shapes table")
    WranglerLogger.debug(f"feed_tables['shapes'].head():\n{feed_tables['shapes'].head()}")
    # shapes columns: shape_id, shape_pt_lat, shape_pt_lon, shape_pt_sequence, shape_dist_traveled, geometry
    #                 trip_id, direction_id, route_id, agency_id, route_short_name, route_type, agency_name


    # For each trip and stop in that trip, find the nearest shape point and update it with stop info
    # This uses a vectorized approach for better performance
    WranglerLogger.info(f"Matching stops to shape points using vectorized approach")
    
    # Project both GeoDataFrames to specified CRS for distance calculations
    feed_tables['shapes'].to_crs(local_crs, inplace=True)
    feed_tables['stops'].to_crs(local_crs, inplace=True)
    
    # Initialize shape columns for stop information
    feed_tables['shapes'][f'match_distance_{crs_units}'] = np.inf
    feed_tables['shapes']['stop_id'] = None
    feed_tables['shapes']['stop_name'] = ''
    feed_tables['shapes']['stop_sequence'] = None
    
    # Add stop geometry to stop_times and convert it a GeoDataFrame
    WranglerLogger.debug(f"Before merge, {len(feed_tables['stop_times'])=:,}")
    feed_tables['stop_times'] = gpd.GeoDataFrame(
        pd.merge(
            left=feed_tables['stop_times'],
            right=feed_tables['stops'][['stop_id','stop_name','geometry']],
            how='left',
            on='stop_id',
            validate='many_to_one'
        ),
        geometry='geometry',
        crs=feed_tables['stops'].crs
    )

    WranglerLogger.debug(f"feed_tables['trips'] type={type(feed_tables['trips'])}:\n{feed_tables['trips']}")
    WranglerLogger.debug(f"feed_tables['stops'] type={type(feed_tables['stops'])}:\n{feed_tables['stops']}")
    WranglerLogger.debug(f"feed_tables['stop_times'] type={type(feed_tables['stop_times'])}:\n{feed_tables['stop_times']}")
    WranglerLogger.debug(f"trip_shapes_df\n{trip_shapes_df}")

    # Step 5: Match shape points to nearby stops
    unique_shape_ids = sorted(feed_tables['shapes']['shape_id'].unique())
    WranglerLogger.info(f"Finding stops for {len(unique_shape_ids):,} unique shape_ids")

    matched_count = 0

    # Log shape information for trace_shape_ids for debugging

    # Make sure this is sorted by trip and then stop sequence
    feed_tables['stop_times'].sort_values(by=['trip_id','stop_sequence'], inplace=True)

    # TODO: This is a little slow; modify to wholly vectorized?
    for shape_id in unique_shape_ids:
        # Get stop_times for this shape
        shape_stops_df = feed_tables['stop_times'][feed_tables['stop_times']['shape_id'] == shape_id]
        # Get unique stops for this shape (some stops may appear in multiple trips)
        unique_stop_info = shape_stops_df.groupby('stop_id').agg({
            'stop_sequence': 'first'  # Use first occurrence of stop_sequence
        }).reset_index()

        # Get shape points for this shape_id
        shape_df = feed_tables['shapes'][feed_tables['shapes']['shape_id'] == shape_id].sort_values(by='shape_pt_sequence')

        # WranglerLogger.debug(f"shape_stops_df:\n{shape_stops_df}")
        # WranglerLogger.debug(f"shape_df:\n{shape_df}")
        shape_indices = shape_df.index.values
        
        # Use spatial index for efficient nearest neighbor search
        from sklearn.neighbors import BallTree

        # Build BallTree for just this shape's points
        shape_coords = np.array([(geom.x, geom.y) for geom in shape_df.geometry])
        assert(len(shape_coords)==len(shape_df))
        shape_tree = BallTree(shape_coords)
        
        # Find nearest shape point for all stops at once
        stop_coords = np.array([(geom.x, geom.y) for geom in shape_stops_df.geometry])
        assert(len(stop_coords)==len(shape_stops_df))
        distances, indices = shape_tree.query(stop_coords, k=1)
        
        # Update shape points with stop information
        for i, stop_id in enumerate(shape_stops_df['stop_id'].tolist()):
            shape_local_idx = indices[i, 0]
            shape_global_idx = shape_indices[shape_local_idx]
            distance = distances[i, 0]
            
            # Only update if this is closer than any previous match
            if distance < feed_tables['shapes'].loc[shape_global_idx, f'match_distance_{crs_units}']:
                stop_info = feed_tables['stops'][feed_tables['stops']['stop_id'] == stop_id].iloc[0]
                stop_seq = unique_stop_info[unique_stop_info['stop_id'] == stop_id]['stop_sequence'].iloc[0]

                feed_tables['shapes'].loc[shape_global_idx, f'match_distance_{crs_units}'] = distance
                feed_tables['shapes'].loc[shape_global_idx, 'stop_id'] = stop_id
                feed_tables['shapes'].loc[shape_global_idx, 'stop_name'] = stop_info['stop_name']
                feed_tables['shapes'].loc[shape_global_idx, 'stop_sequence'] = stop_seq

                # update location to stop location
                feed_tables['shapes'].loc[shape_global_idx, 'shape_pt_lon'] = stop_info['stop_lon']
                feed_tables['shapes'].loc[shape_global_idx, 'shape_pt_lat'] = stop_info['stop_lat']
                feed_tables['shapes'].loc[shape_global_idx, 'geometry'] = stop_info['geometry']
                matched_count += 1

        # Log the result so far -- just the stop shapes
        if trace_shape_ids and (shape_id in trace_shape_ids):
            # refresh
            shape_df = feed_tables['shapes'].loc[feed_tables['shapes']['shape_id'] == shape_id].sort_values(by='shape_pt_sequence')
            # Get route and direction info from the first row
            first_row = shape_df.iloc[0]
            
            WranglerLogger.debug("")
            WranglerLogger.debug(f"  Shape ID: {shape_id}")
            WranglerLogger.debug(f"  Route ID: {first_row['route_id']}")
            WranglerLogger.debug(f"  Route Type: {first_row['route_type']}")
            WranglerLogger.debug(f"  Direction: {first_row['direction_id']}")

            debug_cols = ['shape_pt_sequence','shape_dist_traveled','stop_sequence','stop_id','stop_name',
                          f'match_distance_{crs_units}', 'shape_pt_lon', 'shape_pt_lat']
            WranglerLogger.debug(f"\n{shape_df[debug_cols]}")
    
    WranglerLogger.info("Finished adding stop information to shapes")


def add_stations_and_links_to_roadway_network(
    feed_tables: Dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    trace_shape_ids: Optional[List[str]] = None
) -> Tuple[Dict[str,int], gpd.GeoDataFrame]:
    """Add transit station nodes and dedicated transit links to the roadway network.

    Creates new roadway nodes for transit stations and adds dedicated transit links
    between stations for fixed-guideway transit (rail, subway, ferry, etc.). Bus stops
    use existing roadway nodes from match_bus_stops_to_roadway_nodes().
    
    Process Steps:
    1. Creates stop link pairs from consecutive stops in stop_times
    2. Aggregates intermediate shape points between stops into multi-point lines
    3. Filters links to STATION_ROUTE_TYPES for network addition
    4. Creates new roadway nodes for stations not already in network
    5. Creates dedicated transit links with appropriate access restrictions
    6. Updates feed_tables['stops'] with model_node_id for all stops
    7. Updates feed_tables['shapes'] with shape_model_node_id for all stations
    8. Returns bus stop links separately (not added to network)
    
    Modifies in place:
    
    roadway_net - Adds:
        - New nodes for transit stations with model_node_id
        - New links between stations with:
            - rail_only=True for rail types
            - ferry_only=True for ferry types  
            - drive/bike/walk/truck_access=False
            - Geometry following shape points if available
            
    feed_tables['stops'] - Adds/updates:
        - model_node_id (int): Roadway node ID for the stop
        - Updates existing bus stop model_node_ids
        - Adds new station model_node_ids
    
    feed_tables['shapes'] - Adds/updates:
        - shape_model_node_id (int): Roadway node ID for the shape point
    
    Args:
        feed_tables: Dictionary with required tables:
            - 'stops': Stop information with geometry
            - 'stop_times': Stop sequences for trips
            - 'shapes': Shape points between stops
            - 'routes': Route types
        roadway_net: RoadwayNetwork to modify with new nodes/links
        local_crs: Coordinate reference system for projections
        crs_units: Distance units ('feet' or 'meters')
        trace_shape_ids: Optional shape IDs for debug logging

    Returns:
        Tuple[Dict[str,int], gpd.GeoDataFrame]:
            - Dictionary mapping new station stop_ids to model_node_ids
            - GeoDataFrame of bus stop links (not added to network) with columns:
                shape_id, stop_sequence, stop_id, stop_name, next_stop_id,
                next_stop_name, A, B, geometry
                
    Notes:
        - Stations are new nodes; bus stops use existing road nodes
        - Self-loops (stop appearing twice consecutively) are filtered out
        - Links follow actual shape geometry when available
        - Parent stations without trips are handled correctly
    """
    WranglerLogger.info(f"Adding transit stations and station-based links to the roadway network")
    WranglerLogger.debug(f"feed_tables['shapes'] type={type(feed_tables['shapes'])}:\n{feed_tables['shapes']}")
    WranglerLogger.debug(f"feed_tables['stops'] type={type(feed_tables['stops'])}:\n{feed_tables['stops']}")
    WranglerLogger.debug(f"feed_tables['stop_times'] type={type(feed_tables['stop_times'])}:\n{feed_tables['stop_times']}")

    # Add route_type to stop_times
    if 'route_type' not in feed_tables['stop_times'].columns:
        feed_tables['stop_times'] = pd.merge(
            feed_tables['stop_times'],
            feed_tables['routes'][['route_id','route_type']],
            how='left',
            on='route_id',
            validate='many_to_one'
        )

    # Prepare new link list first
    # For all consecutive stop_ids in feed_table['stop_times'], create consecutive node pairs for each shape
    stop_links_df = feed_tables['stop_times'][['route_type','route_id','direction_id','shape_id','trip_id','stop_sequence','stop_id','stop_name','geometry']].copy()
    stop_links_df.rename(columns={'geometry':'stop_geometry'}, inplace=True)
    stop_links_df.sort_values(by=['trip_id','stop_sequence'], inplace=True)

    stop_links_df['next_stop_id']       = stop_links_df.groupby('trip_id')['stop_id'].shift(-1)
    stop_links_df['next_stop_name']     = stop_links_df.groupby('trip_id')['stop_name'].shift(-1)
    stop_links_df['next_stop_geometry'] = stop_links_df.groupby('trip_id')['stop_geometry'].shift(-1)
    
    # Filter to only rows that have a next node (excludes last point of each shape)
    # and filter out self-loops where the stop occurs twice in a row
    has_next = stop_links_df['next_stop_id'].notna()
    is_self_loop = stop_links_df['next_stop_id'] == stop_links_df['stop_id']
    
    num_self_loops = (has_next & is_self_loop).sum()
    if num_self_loops > 0:
        WranglerLogger.debug(f"Filtering out {num_self_loops:,} self-loop segments where stop_id == next_stop_id")
    
    stop_links_df = stop_links_df[has_next & ~is_self_loop]
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace stop_links_df for {trace_shape_id}:\n"
                f"{stop_links_df.loc[stop_links_df.shape_id==trace_shape_id]}"
            )

    WranglerLogger.debug(f"stop_links_df.dtypes:\n{stop_links_df.dtypes}")
    # route_type            category
    # route_id                object
    # direction_id          category
    # shape_id                object
    # trip_id                 object
    # stop_sequence            int64
    # stop_id                 object
    # stop_name               object
    # stop_geometry         geometry
    # next_stop_id            object
    # next_stop_name          object
    # next_stop_geometry    geometry
    # dtype: object

    # feed_tables['shapes'] is a GeoDataFrame of points, 3 the columns 'shape_id', 'stop_id' and 'stop_sequence'
    # Match these sequences with the stop_id/next_stop_id in stop_links_gdf based on shape_id and add intermediate points to the shape
    shape_links_df = feed_tables['shapes'][['shape_id','geometry','stop_sequence','stop_id']].copy()
    # set the stop_sequence to 1 for the first row of each shape_id if it's not set
    shape_links_df.loc[(~shape_links_df['shape_id'].duplicated()) & (shape_links_df['stop_sequence'].isna()), 'stop_sequence'] = -1
    shape_links_df.loc[(~shape_links_df['shape_id'].duplicated()) & (shape_links_df['stop_id'].isna()),       'stop_id'      ] = -1
    # fill forward
    # Suppress downcasting warning for ffill
    with pd.option_context('future.no_silent_downcasting', True):
        shape_links_df['shape_stop_sequence'] = shape_links_df['stop_sequence'].ffill()
        shape_links_df['shape_stop_id']       = shape_links_df['stop_id'].ffill()
    # drop the first one - that's already covered by the stop point
    shape_links_df.loc[ shape_links_df['stop_id'].notna(), 'shape_stop_sequence'] = np.nan
    shape_links_df.loc[ shape_links_df['stop_id'].notna(), 'shape_stop_id'] = np.nan

    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace shape_links_df for {trace_shape_id} after ffill():\n"
                f"{shape_links_df.loc[shape_links_df.shape_id==trace_shape_id]}"
            )
    # Example: this shape's first stop is stop_sequence=2, the shape points at the beginning are pre-trip points
    #                 shape_id                         geometry stop_sequence stop_id  shape_stop_sequence shape_stop_id
    # 909222  SF:9717:20230930  POINT (6013196.693 2109556.951)            -1      -1                  NaN           NaN
    # 909223  SF:9717:20230930  POINT (6013227.638 2109614.595)          None    None                -1.00            -1
    # 909224  SF:9717:20230930  POINT (6013265.791 2109685.568)          None    None                -1.00            -1
    # 909225  SF:9717:20230930  POINT (6013316.874 2109752.637)          None    None                -1.00            -1
    # 909226  SF:9717:20230930  POINT (6013604.053 2110029.431)          None    None                -1.00            -1
    # 909227  SF:9717:20230930  POINT (6013637.289 2110057.529)          None    None                -1.00            -1
    # 909228  SF:9717:20230930  POINT (6014283.122 2110658.483)             2   15240                  NaN           NaN
    # 909229  SF:9717:20230930  POINT (6014293.455 2110683.403)          None    None                 2.00         15240
    # 909230  SF:9717:20230930  POINT (6014940.906 2111322.944)          None    None                 2.00         15240
    # 909231  SF:9717:20230930  POINT (6015538.148 2111853.892)             3   15237                  NaN           NaN
    # 909232  SF:9717:20230930  POINT (6015603.806 2111941.794)          None    None                 3.00         15237
    # 909233  SF:9717:20230930  POINT (6015814.592 2112130.926)          None    None                 3.00         15237
    # 909234  SF:9717:20230930  POINT (6015897.484 2112242.153)          None    None                 3.00         15237
    # 909235  SF:9717:20230930   POINT (6015932.62 2112307.364)          None    None                 3.00         15237
    # 909236  SF:9717:20230930  POINT (6015955.806 2112367.716)          None    None                 3.00         15237
    # 909237  SF:9717:20230930  POINT (6015971.693 2112438.778)          None    None                 3.00         15237
    # 909238  SF:9717:20230930  POINT (6015985.888 2112526.263)          None    None                 3.00         15237
    # 909239  SF:9717:20230930   POINT (6015989.04 2112639.464)          None    None                 3.00         15237
    # 909240  SF:9717:20230930  POINT (6016045.522 2113160.945)          None    None                 3.00         15237
    # 909241  SF:9717:20230930  POINT (6016108.599 2113665.541)             4   17145                  NaN           NaN

    # aggregate and convert to list
    shape_links_agg_df = shape_links_df.groupby(by=['shape_id','shape_stop_sequence']).aggregate(
        point_list = pd.NamedAgg(column='geometry', aggfunc=list),
        num_points = pd.NamedAgg(column='geometry', aggfunc='nunique')
    ).reset_index(drop=False)
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace shape_links_agg_df for {trace_shape_id}:\n"
                f"{shape_links_agg_df.loc[shape_links_agg_df.shape_id==trace_shape_id]}"
            )
    # columns are shape_id, stop_sequence, point_list, num_points
    #                shape_id  shape_stop_sequence                                         point_list  num_points
    # 36235  SF:9717:20230930                -1.00  [POINT (6013227.6381617775 2109614.5949144145)...           5
    # 36236  SF:9717:20230930                 2.00  [POINT (6014293.454981883 2110683.4032201055),...           2
    # 36237  SF:9717:20230930                 3.00  [POINT (6015603.805947471 2111941.793745517), ...           9
    # 36238  SF:9717:20230930                 4.00  [POINT (6016092.661659231 2113720.8560314486),...          13

    # add intermediate stops to stop_links
    stop_links_df = pd.merge(
        left=stop_links_df,
        right=shape_links_agg_df.rename(columns={'shape_stop_sequence':'stop_sequence'}),
        on=['shape_id','stop_sequence'],
        how='left',
        indicator=True
    )
    WranglerLogger.debug(f"stop_links_df._merge.value_counts():\n{stop_links_df._merge.value_counts()}")
    # for links without intermediate shape points
    stop_links_df.loc[ stop_links_df['_merge'] == 'left_only', 'num_points'] = 0
    if trace_shape_ids:
        WranglerLogger.debug(f"trace stop_links_df:\n{stop_links_df.loc[stop_links_df.shape_id.isin(trace_shape_ids)]}")

    # turn them into multi-point lines
    stop_links_df['geometry'] = stop_links_df.apply(
        lambda row: 
            # if intermediate points
            shapely.geometry.LineString([row['stop_geometry']] + 
                                        row['point_list'] + 
                                        [row['next_stop_geometry']]) if row['num_points'] > 0 
            # no intermediate points
            else shapely.geometry.LineString([row['stop_geometry'], row['next_stop_geometry']]),
            axis=1)
    WranglerLogger.debug(f"stop_links_df including multi-point lines:\n{stop_links_df}")

    # create GeoDataFrame; this is in the local crs
    stop_links_df.drop(columns=['stop_geometry','next_stop_geometry','point_list','_merge'], inplace=True)
    stop_links_gdf = gpd.GeoDataFrame(stop_links_df, geometry='geometry', crs=feed_tables['stops'].crs)
    WranglerLogger.debug(f"stop_links_gdf.dtypes\n{stop_links_gdf.dtypes}")
    # route_type        category
    # route_id            object
    # direction_id      category
    # shape_id            object
    # trip_id             object
    # stop_sequence        int64
    # stop_id             object
    # stop_name           object
    # next_stop_id        object
    # next_stop_name      object
    # num_points           int64
    # geometry          geometry
    # dtype: object
    
    # Filter to STATION_ROUTE_TYPES for adding to the roadway network
    station_stop_links_gdf = stop_links_gdf.loc[ stop_links_gdf.route_type.isin(STATION_ROUTE_TYPES) ].copy()
    station_stop_id_set = set(station_stop_links_gdf['stop_id']) | set(station_stop_links_gdf['next_stop_id'])
    # Also add parent stations
    parent_station_id_set = set(feed_tables['stops'].loc[ feed_tables['stops']['is_parent'] == True, 'stop_id'].tolist())
    WranglerLogger.debug(f"parent_station_id_set len={len(parent_station_id_set)}:\n{parent_station_id_set}")

    if trace_shape_ids:
        WranglerLogger.debug(f"trace station_stop_links_gdf:\n{station_stop_links_gdf.loc[ station_stop_links_gdf.shape_id.isin(trace_shape_ids)]}")

    # Prepare nodes to add - station stops and parent stops
    station_stop_ids_gdf = feed_tables['stops'].loc[ 
        (feed_tables['stops']['stop_id'].isin(station_stop_id_set)) |
        (feed_tables['stops']['stop_id'].isin(parent_station_id_set)), 
        ['stop_id','stop_name','is_parent','stop_lat','stop_lon','model_node_id','geometry'] ].reset_index(drop=True).copy()
    station_stop_ids_gdf.rename(columns={'stop_lon':'X', 'stop_lat':'Y'}, inplace=True)
    station_stop_ids_gdf.to_crs(LAT_LON_CRS, inplace=True)
    WranglerLogger.debug(f"station_stop_ids_gdf:\n{station_stop_ids_gdf}")

    # Don't create new stations where one already exists! (stops that serve both bus and light rail)
    # => Filter to station ONLY  
    new_station_stop_ids_gdf = station_stop_ids_gdf.loc[ station_stop_ids_gdf['model_node_id'].isna() ].reset_index(drop=False)
    new_station_stop_ids_gdf.drop(columns={'model_node_id'}, inplace=True)

    # Assign model_node_id and add new stations to roadway network as roadway nodes
    max_node_num = roadway_net.nodes_df.model_node_id.max()
    new_station_stop_ids_gdf['model_node_id'] = new_station_stop_ids_gdf.index + max_node_num + 1
    WranglerLogger.info(f"Adding {len(new_station_stop_ids_gdf):,} nodes to roadway network")
    WranglerLogger.debug(f"new_station_stop_ids_gdf:\n{new_station_stop_ids_gdf}")
    WranglerLogger.debug(f"Before adding nodes, {len(roadway_net.nodes_df)=:,}")
    roadway_net.add_nodes(new_station_stop_ids_gdf)
    WranglerLogger.debug(f"After adding nodes, {len(roadway_net.nodes_df)=:,}")

    # put together new model_node_ids with bus model_node_ids
    station_stop_ids_gdf = pd.concat([
        station_stop_ids_gdf.loc[ station_stop_ids_gdf['model_node_id'].notna() ],
        new_station_stop_ids_gdf
    ])
    WranglerLogger.debug(f"station_stop_ids_gdf with bus and new station stops:\n{station_stop_ids_gdf}")

    stop_id_to_model_node_id_dict = new_station_stop_ids_gdf[['stop_id','model_node_id']].set_index('stop_id').to_dict()['model_node_id']
    WranglerLogger.debug(f"stop_id_to_model_node_id_dict:\n{stop_id_to_model_node_id_dict}")

    # Prepare links to add
    station_stop_links_gdf['A'] = station_stop_links_gdf['stop_id'].map(stop_id_to_model_node_id_dict)
    station_stop_links_gdf['B'] = station_stop_links_gdf['next_stop_id'].map(stop_id_to_model_node_id_dict)
    # Set rail/ferry only values
    station_stop_links_gdf = station_stop_links_gdf[['route_type','A','stop_id','stop_name','B','next_stop_id','next_stop_name','geometry']]
    station_stop_links_gdf['rail_only'] = False
    station_stop_links_gdf.loc[ station_stop_links_gdf.route_type.isin(RAIL_ROUTE_TYPES), 'rail_only'] = True
    station_stop_links_gdf['ferry_only'] = False
    station_stop_links_gdf.loc[ station_stop_links_gdf.route_type.isin(FERRY_ROUTE_TYPES), 'ferry_only'] = True
    # Aggregate by A,B, choosing first values, and convert back to GeoDataFrame
    station_road_links_gdf = gpd.GeoDataFrame(
        station_stop_links_gdf.groupby(by=['A','B']).aggregate(
            stop_id        = pd.NamedAgg(column='stop_id', aggfunc='first'),
            stop_name      = pd.NamedAgg(column='stop_name', aggfunc='first'),
            next_stop_id   = pd.NamedAgg(column='next_stop_id', aggfunc='first'),
            next_stop_name = pd.NamedAgg(column='next_stop_name', aggfunc='first'),
            geometry       = pd.NamedAgg(column='geometry', aggfunc='first'),
            rail_only      = pd.NamedAgg(column='rail_only', aggfunc=any),
            ferry_only     = pd.NamedAgg(column='ferry_only', aggfunc=any),
        ).reset_index(drop=False),
        crs=station_stop_links_gdf.crs
    )
    
    # Assign model_link_id, access for drive,walk,bike,truck,bus
    max_model_link_id = roadway_net.links_df.model_link_id.max()
    station_road_links_gdf['model_link_id'] = station_road_links_gdf.index + max_model_link_id + 1
    station_road_links_gdf['name'] = station_road_links_gdf['stop_name'] + " to " + station_road_links_gdf['next_stop_name']
    station_road_links_gdf['shape_id'] = station_road_links_gdf['stop_id'] + " to " + station_road_links_gdf['next_stop_id']
    station_road_links_gdf['drive_access'] = False
    station_road_links_gdf['bike_access'] = False
    station_road_links_gdf['walk_access'] = False
    station_road_links_gdf['truck_access'] = False
    station_road_links_gdf['bus_only'] = False
    station_road_links_gdf['lanes'] = 0
    # Set distance
    station_road_links_gdf.to_crs(local_crs, inplace=True)
    station_road_links_gdf['length'] = station_road_links_gdf.length
    if crs_units == "feet": 
        station_road_links_gdf['distance'] = station_road_links_gdf['length']/FEET_PER_MILE
    else:
        station_road_links_gdf['distance'] = station_road_links_gdf['length']/METERS_PER_KILOMETER
    station_road_links_gdf.to_crs(LAT_LON_CRS, inplace=True)

    # Add to roadway network
    WranglerLogger.info(f"Adding {len(station_road_links_gdf):,} links to roadway network")
    WranglerLogger.debug(f"station_road_links_gdf:\n{station_road_links_gdf}")
    WranglerLogger.debug(f"Before adding links, {len(roadway_net.links_df)=:,}")
    roadway_net.add_links(station_road_links_gdf)
    WranglerLogger.debug(f"After adding links, {len(roadway_net.links_df)=:,}")

    WranglerLogger.info(f"Adding {len(station_road_links_gdf):,} shapes to roadway network")
    roadway_net.add_shapes(station_road_links_gdf)

    # Update feed_table['stops']: set model_node_id for stations in feed_table['stops']
    WranglerLogger.debug(
        f"Before updating, feed_tables['stops'] with model_node_id set: "
        f"{feed_tables['stops']['model_node_id'].notna().sum():,}"
    )
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['stops'] for {trace_shape_id}:\n"
                f"{feed_tables['stops'].loc[ feed_tables['stops']['shape_ids'].apply(lambda x: isinstance(x, list) and trace_shape_id in x) ]}"
            )

    feed_tables['stops']['station_node_id'] = feed_tables['stops']['stop_id'].map(stop_id_to_model_node_id_dict)

    # Verify no stops table stops have a model_node_id already set (from match_bus_stops_to_roadway_nodes() *and* have a station_model_node_id
    have_both_df = feed_tables['stops'].loc[ 
        feed_tables['stops']['model_node_id'].notna() & 
        feed_tables['stops']['station_node_id'].notna()
    ]
    assert(len(have_both_df) == 0)

    feed_tables['stops'].loc[ feed_tables['stops']['station_node_id'].notna(), 'model_node_id'] = feed_tables['stops']['station_node_id']
    WranglerLogger.debug(
        f"After updating, feed_tables['stops'] with model_node_id set:\n"
        f"{feed_tables['stops']['model_node_id'].notna().sum():,}"
    )
    # Log feed_tables['stops'] for trace_shape_ids
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['stops'] for {trace_shape_id}:\n"
                f"{feed_tables['stops'].loc[ feed_tables['stops']['shape_ids'].apply(lambda x: isinstance(x, list) and trace_shape_id in x) ]}"
            )

    feed_tables['stops'].drop(columns=['station_node_id'], inplace=True)
    WranglerLogger.debug(f"feed_tables['stops']:\n{feed_tables['stops']}")
    # TODO: I think these are all parent nodes...
    WranglerLogger.debug(f"feed_tables['stops'] without model_node_id:\n{feed_tables['stops'].loc[ feed_tables['stops']['model_node_id'].isna() ]}")

    # Update feed_table['shapes']: set shape_model_node_id for stations in feed_table['shapes'] and delete the other nodes.
    # Those are now in the shape of the roadway network link
    WranglerLogger.debug(f"About to update feed['shapes'] in add_stations_and_links_to_roadway_network:\n{feed_tables['shapes']}")
    feed_tables['shapes']['station_node_id'] = feed_tables['shapes']['stop_id'].map(stop_id_to_model_node_id_dict)

    # Verify no shapes table stops have a shape_model_node_id already set *and* have a station_model_node_id
    have_both_df = feed_tables['shapes'].loc[ 
        feed_tables['shapes']['shape_model_node_id'].notna() & 
        feed_tables['shapes']['station_node_id'].notna()
    ]
    assert(len(have_both_df) == 0)
    feed_tables['shapes'].loc[ feed_tables['shapes']['station_node_id'].notna(), 'shape_model_node_id'] = feed_tables['shapes']['station_node_id']
    feed_tables['shapes'].drop(columns=['station_node_id'], inplace=True)
    # Delete other nodes
    feed_tables['shapes'] = feed_tables['shapes'].loc[ 
        # leave non station route types (bus) alone -- these will be handled elsewhere
        ~feed_tables['shapes']['route_type'].isin(STATION_ROUTE_TYPES) |
        # for station route types, only keep those with model_node_id set
        feed_tables['shapes']['shape_model_node_id'].notna()
    ]

    # Log feed_tables['shapes'] for trace_shape_ids
    if trace_shape_ids:
        for trace_shape_id in trace_shape_ids:
            WranglerLogger.debug(
                f"trace feed_tables['shapes'] for {trace_shape_id} at the end of add_stations_and_links_to_roadway_network():\n"
                f"{feed_tables['shapes'].loc[ feed_tables['shapes']['shape_id']==trace_shape_id]}"
            )

    # set A for stop_id's model_node_id
    stop_links_gdf = pd.merge(
        left=stop_links_gdf,
        right=feed_tables['stops'][['stop_id','model_node_id']],
        how='left',
        validate='many_to_one'
    ).rename(columns={'model_node_id':'A'})
    # set B for next_stop_id's model_node_id
    stop_links_gdf = pd.merge(
        left=stop_links_gdf,
        right=feed_tables['stops'][['stop_id','model_node_id']].rename(columns={'stop_id':'next_stop_id'}),
        how='left',
        validate='many_to_one'
    ).rename(columns={'model_node_id':'B'})
    WranglerLogger.debug(f"stop_links_gdf after setting A and B:\n{stop_links_gdf}")

    # Filter non-station stop links to return
    bus_stop_links_gdf = stop_links_gdf.loc[~stop_links_gdf.route_type.isin(STATION_ROUTE_TYPES)]

    return stop_id_to_model_node_id_dict, bus_stop_links_gdf
    
def create_feed_from_gtfs_model(
    gtfs_model: GtfsModel,
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    timeperiods: Dict[str, Tuple[str, str]],
    frequency_method: str,
    default_frequency_for_onetime_route: int = 10800,
    add_stations_and_links: bool = True,
    skip_stop_agencies: Optional[List[str]] = None,
    trace_shape_ids: Optional[List[str]] = None
) -> Feed:
    """Convert GTFS model to Wrangler Feed with stops mapped to roadway network.
    
    Comprehensive conversion that transforms GTFS schedule data into a frequency-based
    Feed representation compatible with travel modeling. Maps transit stops to roadway
    nodes and optionally adds station infrastructure to the network.
    
    Process Steps:
    1. Converts roadway nodes to GeoDataFrame if needed
    2. Copies and prepares GTFS tables (routes, trips, stops, etc.)
    3. Enriches stops with route/agency metadata (add_additional_data_to_stops)
    4. Creates frequency-based schedules from detailed timetables
    5. Enriches shapes with stop matching (add_additional_data_to_shapes)
    6. Matches bus stops to existing roadway nodes
    7. Adds station nodes and links for fixed-guideway transit
    8. Routes bus services through road network between stops
    9. Creates final Feed object with all processed tables
    
    Modifies in place:
    
    roadway_net.nodes_df - Converted to GeoDataFrame and adds:
        - geometry: Point geometry if not present
        - bus_access (bool): True if node is in bus modal graph
        - Additional modal access flags may be added
        
    roadway_net - If add_stations_and_links=True:
        - Adds new nodes for transit stations
        - Adds dedicated transit links between stations
        - Updates shapes and modal graphs
    
    Creates feed_tables dictionary with:
        - routes, trips, agencies: Copied from GTFS
        - stops: Enhanced with model_node_id mappings
        - stop_times: Converted to frequency-based
        - shapes: Updated with roadway routing
        - frequencies: Created from timetables
    
    Args:
        gtfs_model: Source GTFS data model
        roadway_net: Target roadway network for stop mapping
        local_crs: Coordinate system for distance calculations
        crs_units: Distance units ('feet' or 'meters')
        timeperiods: Time period definitions for frequencies
            Example: {'EA': ('03:00','06:00'), 'AM': ('06:00','10:00')}
        frequency_method: How to calculate headways
            ('uniform_headway', 'mean_headway', or 'median_headway')
        default_frequency_for_onetime_route: Default headway in seconds
            for routes with one trip per period (default: 10800)
        add_stations_and_links: If True, add stations to roadway network
            (recommended, False not implemented)
        skip_stop_agencies: Agency IDs using skip-stop patterns
        trace_shape_ids: Shape IDs for detailed debug logging

    Returns:
        Feed: Wrangler Feed object with:
            - Stops mapped to roadway nodes
            - Frequency-based trip representation
            - Routes following road network paths
    
    Raises:
        TransitValidationError: If bus stops can't be matched to roadway
        NodeNotFoundError: If required nodes aren't found
        Exception: Debug exception with intermediate data (temporary)
        
    Notes:
        - Currently raises a debug exception before completion (line 1698-1702)
        - Bus routes are re-routed through actual road network
        - Station routes keep original alignment with new nodes/links
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    # Convert roadway_net.nodes_df GeoDataFrame if needed (modifying in place)
    if not isinstance(roadway_net.nodes_df, gpd.GeoDataFrame):
        if "geometry" not in roadway_net.nodes_df.columns:
            node_geometry = [shapely.geometry.Point(x, y) for x, y in zip(roadway_net.nodes_df["X"], roadway_net.nodes_df["Y"])]
            roadway_net.nodes_df = gpd.GeoDataFrame(roadway_net.nodes_df, geometry=node_geometry, crs=LAT_LON_CRS)
        else:
            roadway_net.nodes_df = gpd.GeoDataFrame(roadway_net.nodes_df, crs=LAT_LON_CRS)
    else:
        if roadway_net.nodes_df.crs is None:
            roadway_net.nodes_df = roadway_net.nodes_df.set_crs(LAT_LON_CRS)
    
    # Start with the tables from the GTFS model
    feed_tables = {}

    # Copy over standard tables that don't need modification
    # GtfsModel guarantees routes and trips exist
    # create a copies of this which we'll manipulate
    feed_tables["routes"]     = gtfs_model.routes.copy()
    feed_tables["trips"]      = gtfs_model.trips.copy()
    feed_tables["agencies"]   = gtfs_model.agency.copy()
    feed_tables["stops"]      = gtfs_model.stops.copy()
    feed_tables["stop_times"] = gtfs_model.stop_times.copy()
    feed_tables["shapes"]     = gtfs_model.shapes.copy()

    # create mapping from gtfs_model stop to RoadwayNetwork nodes
    # GtfsModel guarantees stops exists
    if not isinstance(feed_tables["stops"], gpd.GeoDataFrame):
        stop_geometry = [
            shapely.geometry.Point(lon, lat) for lon, lat in zip(gtfs_model.stops["stop_lon"], gtfs_model.stops["stop_lat"])
        ]
        feed_tables["stops"] = gpd.GeoDataFrame(feed_tables["stops"], geometry=stop_geometry, crs=LAT_LON_CRS)

    # Add helpful extra data to stops table
    add_additional_data_to_stops(feed_tables)

    # create frequencies table from GTFS stop_times (if no frequencies table is specified)
    if hasattr(gtfs_model, "frequencies") and gtfs_model.frequencies is not None:
        feed_tables["frequencies"] = gtfs_model.frequencies
        # TODO: What if the frequencies are specified for the wrong time periods?
    else:
        # GtfsModel specifies every individual trip but Feed expects the trip to be
        # representative with frequencies. This makes that conversion
        create_feed_frequencies(
            feed_tables, timeperiods, frequency_method, default_frequency_for_onetime_route, trace_shape_ids
        )

    if not add_stations_and_links:
        raise NotImplementedError("create_feed_from_gtfs_feed() doesn't implement add_stations_and_links==False.")
    
    # Add helpful extra data to shapes table
    add_additional_data_to_shapes(feed_tables, local_crs, crs_units)

    match_bus_stops_to_roadway_nodes(
        feed_tables,
        roadway_net, 
        local_crs, 
        crs_units, 
        MAX_DISTANCE_STOP[crs_units],
        trace_shape_ids)
    
    # for fixed route transit, add the links and stops to the roadway network
    station_id_to_model_node_id_dict, bus_stop_links_gdf = add_stations_and_links_to_roadway_network(
        feed_tables, roadway_net, local_crs, crs_units, trace_shape_ids)
    
    WranglerLogger.debug(f"bus_stop_links_gdf:\n{bus_stop_links_gdf}")

    # finally, we need to find shortest paths through the bus network
    # between bus stops and update stops and shapes accordingly
    try:
        create_bus_routes(bus_stop_links_gdf, feed_tables, roadway_net, local_crs, crs_units, trace_shape_ids)
    except Exception as e:
        raise e
    
    # Getting ready to create Feed object
    # stop_id is now really the model_node_id -- set it
    feed_tables['stops'].rename(columns={'stop_id':'orig_stop_id', 'model_node_id':'stop_id'}, inplace=True)
    for table_name in feed_tables.keys():
        WranglerLogger.debug(f"Before creating Feed object, feed_tables[{table_name}]:\n{feed_tables[table_name]}")

    # create Feed object from results of the above
    try:
        feed = Feed(**feed_tables)
        WranglerLogger.info(f"Successfully created Feed with {len(feed_tables)} tables")
        return feed
    except Exception as e:
        WranglerLogger.error(f"Error creating Feed: {e}")
        raise e



def filter_transit_by_boundary(
    transit_data: Union[GtfsModel, Feed],
    boundary: Union[str, Path, gpd.GeoDataFrame],
    partially_include_route_type_action: Optional[Dict[RouteType, str]] = None,
) -> None:
    """Filter transit routes based on whether they have stops within a boundary.
    
    Removes routes that are entirely outside the boundary shapefile. Routes that are
    partially within the boundary are kept by default, but can be configured per 
    route type to be truncated at the boundary. Modifies transit_data in place.
    
    Args:
        transit_data: Either a GtfsModel or Feed object to filter. Modified in place.
        boundary: Path to boundary shapefile or a GeoDataFrame with boundary polygon(s)
        partially_include_route_type_action: Optional dictionary mapping RouteType enum to 
            action for routes partially within boundary:
            - "truncate": Truncate route to only include stops within boundary
            Route types not specified in this dictionary will be kept entirely (default).
    
    Example:
        >>> from network_wrangler.models.gtfs.types import RouteType
        >>> # Remove routes entirely outside the Bay Area
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp"
        ... )
        >>> 
        >>> # Truncate rail routes at boundary, keep other route types unchanged
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp",
        ...     partially_include_route_type_action={
        ...         RouteType.RAIL: "truncate",  # Rail - will be truncated at boundary
        ...         # Other route types not listed will be kept entirely
        ...     }
        ... )
    """
    WranglerLogger.info("Filtering transit routes by boundary")
    
    # Log input parameters
    WranglerLogger.debug(f"partially_include_route_type_action: {partially_include_route_type_action}")
    
    # Load boundary if it's a file path
    if isinstance(boundary, (str, Path)):
        WranglerLogger.debug(f"Loading boundary from file: {boundary}")
        boundary_gdf = gpd.read_file(boundary)
    else:
        WranglerLogger.debug("Using provided boundary GeoDataFrame")
        boundary_gdf = boundary
    
    WranglerLogger.debug(f"Boundary has {len(boundary_gdf)} polygon(s)")
    
    # Ensure boundary is in a geographic CRS for spatial operations
    if boundary_gdf.crs is None:
        WranglerLogger.warning("Boundary has no CRS, assuming EPSG:4326")
        boundary_gdf = boundary_gdf.set_crs(LAT_LON_CRS)
    else:
        WranglerLogger.debug(f"Boundary CRS: {boundary_gdf.crs}")
    
    # Get references to tables (not copies since we'll modify in place)
    is_gtfs = isinstance(transit_data, GtfsModel)
    stops_df = transit_data.stops
    routes_df = transit_data.routes
    trips_df = transit_data.trips
    stop_times_df = transit_data.stop_times
    
    if is_gtfs:
        WranglerLogger.debug("Processing GtfsModel data")
    else:
        WranglerLogger.debug("Processing Feed data")
    
    WranglerLogger.debug(f"Input data has {len(stops_df)} stops, {len(routes_df)} routes, {len(trips_df)} trips, {len(stop_times_df)} stop_times")
    
    # Create GeoDataFrame from stops
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs=LAT_LON_CRS
    )
    
    # Reproject to match boundary CRS if needed
    if stops_gdf.crs != boundary_gdf.crs:
        WranglerLogger.debug(f"Reprojecting stops from {stops_gdf.crs} to {boundary_gdf.crs}")
        stops_gdf = stops_gdf.to_crs(boundary_gdf.crs)
    
    # Spatial join to find stops within boundary
    WranglerLogger.debug("Performing spatial join to find stops within boundary")
    stops_in_boundary = gpd.sjoin(
        stops_gdf,
        boundary_gdf,
        how="inner",
        predicate="within"
    )
    stops_in_boundary_ids = set(stops_in_boundary.stop_id.unique())
    
    # Log some stops that are outside boundary for debugging
    stops_outside_boundary = set(stops_df.stop_id) - stops_in_boundary_ids
    if stops_outside_boundary:
        sample_outside = list(stops_outside_boundary)[:5]
        WranglerLogger.debug(f"Sample of stops outside boundary: {sample_outside}")
    
    WranglerLogger.info(
        f"Found {len(stops_in_boundary_ids):,} stops within boundary "
        f"out of {len(stops_df):,} total stops"
    )
    
    # Find which routes to keep
    # Get unique stop-route pairs from stop_times and trips
    stop_route_pairs = pd.merge(
        stop_times_df[['trip_id', 'stop_id']],
        trips_df[['trip_id', 'route_id']],
        on='trip_id'
    )[['stop_id', 'route_id']].drop_duplicates()
    
    # Group by route to find which stops each route serves
    route_stops = stop_route_pairs.groupby('route_id')['stop_id'].apply(set).reset_index()
    route_stops.columns = ['route_id', 'stop_ids']
    
    # Add route_type information
    route_stops = pd.merge(route_stops, routes_df[['route_id', 'route_type']], on='route_id', how='left')
    
    # Initialize with default filters
    if partially_include_route_type_action is None:
        partially_include_route_type_action = {}
    
    # Convert RouteType enum keys to int values for comparison with dataframe
    normalized_route_type_action = {}
    for key, value in partially_include_route_type_action.items():
        if not isinstance(key, RouteType):
            raise TypeError(f"Keys in partially_include_route_type_action must be RouteType enum, got {type(key)}")
        normalized_route_type_action[key.value] = value
    partially_include_route_type_action = normalized_route_type_action
    
    # Track routes to truncate
    routes_to_truncate = {}
    
    # Determine which routes to keep and how to handle them
    def determine_route_handling(row):
        route_id = row['route_id']
        route_type = row['route_type']
        stop_ids = row['stop_ids']
        
        # Check if route has stops both inside and outside boundary
        stops_inside = stop_ids.intersection(stops_in_boundary_ids)
        stops_outside = stop_ids - stops_in_boundary_ids
        
        # If all stops are outside, always remove
        if len(stops_inside) == 0:
            WranglerLogger.debug(f"Route {route_id} (type {route_type}): all {len(stop_ids)} stops outside boundary - REMOVE")
            return 'remove'
        
        # If all stops are inside, always keep
        if len(stops_outside) == 0:
            return 'keep'
        
        # Route has stops both inside and outside - check partially_include_route_type_action
        WranglerLogger.debug(
            f"Route {route_id} (type {route_type}): {len(stops_inside)} stops inside, "
            f"{len(stops_outside)} stops outside boundary"
        )
        
        if route_type in partially_include_route_type_action:
            action = partially_include_route_type_action[route_type]
            WranglerLogger.debug(f"  - Applying configured action for route_type {route_type}: {action}")
            if action == 'truncate':
                return 'truncate'
        
        # Default to keep if not specified
        WranglerLogger.debug(f"  - No action configured for route_type {route_type}, defaulting to KEEP")
        return 'keep'
    
    route_stops['handling'] = route_stops.apply(determine_route_handling, axis=1)
    WranglerLogger.debug(f"route_stops with handling set:\n{route_stops}")
    
    routes_to_keep = set(route_stops[route_stops['handling'].isin(['keep', 'truncate'])]['route_id'])
    routes_to_remove = set(route_stops[route_stops['handling'] == 'remove']['route_id'])
    routes_needing_truncation = set(route_stops[route_stops['handling'] == 'truncate']['route_id'])
    
    WranglerLogger.info(
        f"Keeping {len(routes_to_keep):,} routes "
        f"out of {len(routes_df):,} total routes"
    )
    
    if routes_to_remove:
        WranglerLogger.info(f"Removing {len(routes_to_remove):,} routes entirely outside boundary")
        WranglerLogger.debug(f"Routes being removed: {sorted(routes_to_remove)[:10]}...")
    
    if routes_needing_truncation:
        WranglerLogger.info(f"Truncating {len(routes_needing_truncation):,} routes at boundary")
        WranglerLogger.debug(f"Routes being truncated: {sorted(routes_needing_truncation)[:10]}...")
    
    # Filter data
    filtered_routes = routes_df[routes_df.route_id.isin(routes_to_keep)]
    filtered_trips = trips_df[trips_df.route_id.isin(routes_to_keep)]
    filtered_trip_ids = set(filtered_trips.trip_id)
    
    # Handle truncation by calling truncate_route_at_stop for each route needing truncation
    if routes_needing_truncation:
        WranglerLogger.debug(f"Processing truncation for {len(routes_needing_truncation)} routes")
        
        # Start with the current filtered data
        # Need to ensure stop_times only includes trips that are in filtered_trips
        filtered_stop_times_for_truncation = stop_times_df[stop_times_df.trip_id.isin(filtered_trip_ids)]
        
        # First update transit_data with filtered data before truncation (in order to maintain validation)
        transit_data.stop_times = filtered_stop_times_for_truncation
        transit_data.trips = filtered_trips
        transit_data.routes = filtered_routes
        
        # Process each route that needs truncation
        for route_id in routes_needing_truncation:
            WranglerLogger.debug(f"Processing truncation for route {route_id}")
            
            # Get trips for this route
            route_trips = trips_df[trips_df.route_id == route_id]
            
            # Group by direction_id
            for direction_id in route_trips.direction_id.unique():
                dir_trips = route_trips[route_trips.direction_id == direction_id]
                if len(dir_trips) == 0:
                    continue
                
                # Analyze stop patterns for this route/direction
                # Get a representative trip (first one)
                sample_trip_id = dir_trips.iloc[0].trip_id
                sample_stop_times = transit_data.stop_times[transit_data.stop_times.trip_id == sample_trip_id].sort_values('stop_sequence')
                
                # Find which stops are inside/outside boundary
                stop_boundary_status = sample_stop_times['stop_id'].isin(stops_in_boundary_ids)
                
                # Check if route exits and re-enters boundary (complex case)
                boundary_changes = stop_boundary_status.ne(stop_boundary_status.shift()).cumsum()
                num_segments = boundary_changes.nunique()
                
                if num_segments > 2:
                    # Complex case: route exits and re-enters boundary
                    route_info = routes_df[routes_df.route_id == route_id].iloc[0]
                    route_name = route_info.get('route_short_name', route_id)
                    msg = (
                        f"Route {route_name} ({route_id}) direction {direction_id} has a complex "
                        f"boundary crossing pattern (crosses boundary {num_segments - 1} times). "
                        f"Can only handle routes that exit boundary at beginning or end."
                    )
                    raise ValueError(msg)
                
                # Determine truncation type
                first_stop_inside = stop_boundary_status.iloc[0]
                last_stop_inside = stop_boundary_status.iloc[-1]
                
                if not first_stop_inside and not last_stop_inside:
                    # All stops outside - shouldn't happen as route would be removed
                    continue
                elif first_stop_inside and last_stop_inside:
                    # All stops inside - no truncation needed
                    continue
                elif not first_stop_inside and last_stop_inside:
                    # Starts outside, ends inside - truncate before first inside stop
                    # Find first True value (first stop inside boundary)
                    first_inside_pos = stop_boundary_status.tolist().index(True)
                    first_inside_stop = sample_stop_times.iloc[first_inside_pos]['stop_id']
                    
                    WranglerLogger.debug(
                        f"Route {route_id} dir {direction_id}: truncating before stop {first_inside_stop}"
                    )
                    truncate_route_at_stop(
                        transit_data, route_id, direction_id, first_inside_stop, "before"
                    )
                elif first_stop_inside and not last_stop_inside:
                    # Starts inside, ends outside - truncate after last inside stop
                    # Find last True value (last stop inside boundary) 
                    reversed_list = stop_boundary_status.tolist()[::-1]
                    last_inside_pos = len(reversed_list) - 1 - reversed_list.index(True)
                    last_inside_stop = sample_stop_times.iloc[last_inside_pos]['stop_id']
                    
                    WranglerLogger.debug(
                        f"Route {route_id} dir {direction_id}: truncating after stop {last_inside_stop}"
                    )
                    truncate_route_at_stop(
                        transit_data, route_id, direction_id, last_inside_stop, "after"
                    )
        
        # After truncation, transit_data has been modified in place
        # Update references to current state (in order to maintain validation)
        filtered_stop_times = transit_data.stop_times
        filtered_trips = transit_data.trips
        filtered_routes = transit_data.routes
        filtered_stops = transit_data.stops
    else:
        # No truncation needed - update transit_data with filtered data
        filtered_stop_times = stop_times_df[stop_times_df.trip_id.isin(filtered_trip_ids)]
        filtered_stops = stops_df[stops_df.stop_id.isin(filtered_stop_times.stop_id.unique())]
        
        # Check if any of the filtered stops reference parent stations
        if "parent_station" in filtered_stops.columns:
            # Get parent stations that are referenced by kept stops
            parent_stations = filtered_stops["parent_station"].dropna().unique()
            parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings
            
            if len(parent_stations) > 0:
                # Find parent stations that aren't already in our filtered stops
                existing_stop_ids = set(filtered_stops.stop_id)
                missing_parent_stations = [ps for ps in parent_stations if ps not in existing_stop_ids]
                
                if len(missing_parent_stations) > 0:
                    WranglerLogger.debug(
                        f"Adding back {len(missing_parent_stations)} parent stations referenced by kept stops"
                    )
                    
                    # Get the parent station records
                    parent_station_records = stops_df[
                        stops_df.stop_id.isin(missing_parent_stations)
                    ]
                    
                    # Append parent stations to filtered stops
                    filtered_stops = pd.concat(
                        [filtered_stops, parent_station_records], 
                        ignore_index=True
                    )
        
        transit_data.stop_times = filtered_stop_times
        transit_data.trips = filtered_trips
        transit_data.routes = filtered_routes
        transit_data.stops = filtered_stops
    
    # Log details about removed stops
    stops_still_used = set(filtered_stops.stop_id.unique())
    removed_stops = set(stops_df.stop_id) - stops_still_used
    if removed_stops:
        WranglerLogger.debug(f"Removed {len(removed_stops)} stops that are no longer referenced")
        
        # Get details of removed stops
        removed_stops_df = stops_df[stops_df['stop_id'].isin(removed_stops)][['stop_id', 'stop_name']]
        
        # Log up to 20 removed stops with their names
        sample_size = min(20, len(removed_stops_df))
        for _, stop in removed_stops_df.head(sample_size).iterrows():
            WranglerLogger.debug(f"  - Removed stop: {stop['stop_id']} ({stop['stop_name']})")
        
        if len(removed_stops) > sample_size:
            WranglerLogger.debug(f"  ... and {len(removed_stops) - sample_size} more stops")
    
    WranglerLogger.info(
        f"After filtering: {len(filtered_routes):,} routes, "
        f"{len(filtered_trips):,} trips, {len(filtered_stops):,} stops"
    )
    
    # Log summary of filtering by action type
    route_handling_summary = route_stops.groupby('handling').size()
    WranglerLogger.debug(f"Route handling summary:\n{route_handling_summary}")
    
    # Log route type distribution for routes with mixed stops
    mixed_routes = route_stops[
        (route_stops['handling'].isin(['keep', 'truncate'])) & 
        (route_stops['route_id'].isin(routes_needing_truncation) | 
         route_stops['handling'] == 'keep')
    ]
    if len(mixed_routes) > 0:
        route_type_summary = mixed_routes.groupby('route_type')['handling'].value_counts()
        WranglerLogger.debug(f"Route types with partial stops:\n{route_type_summary}")
    
    # Update other tables in transit_data in place
    if is_gtfs:
        # For GtfsModel, also filter shapes and other tables if they exist
        if hasattr(transit_data, 'agency') and transit_data.agency is not None:
            # Keep only agencies referenced by remaining routes
            if 'agency_id' in filtered_routes.columns:
                agency_ids = set(filtered_routes.agency_id.dropna().unique())
                transit_data.agency = transit_data.agency[
                    transit_data.agency.agency_id.isin(agency_ids)
                ]
        
        if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
            # Keep only shapes referenced by remaining trips
            if 'shape_id' in filtered_trips.columns:
                shape_ids = set(filtered_trips.shape_id.dropna().unique())
                transit_data.shapes = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]
        
        if hasattr(transit_data, 'calendar') and transit_data.calendar is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(filtered_trips.service_id.unique())
            transit_data.calendar = transit_data.calendar[
                transit_data.calendar.service_id.isin(service_ids)
            ]
        
        if hasattr(transit_data, 'calendar_dates') and transit_data.calendar_dates is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(filtered_trips.service_id.unique())
            transit_data.calendar_dates = transit_data.calendar_dates[
                transit_data.calendar_dates.service_id.isin(service_ids)
            ]
        
        if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            transit_data.frequencies = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(filtered_trip_ids)
            ]
    
    else:  # Feed
        # For Feed, also handle frequencies and shapes
        if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
            # Keep only shapes referenced by remaining trips
            if 'shape_id' in filtered_trips.columns:
                shape_ids = set(filtered_trips.shape_id.dropna().unique())
                transit_data.shapes = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]
        
        if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            transit_data.frequencies = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(filtered_trip_ids)
            ]


def drop_transit_agency(
    transit_data: Union[GtfsModel, Feed],
    agency_id: Union[str, List[str]],
) -> None:
    """Remove all routes, trips, stops, etc. for a specific agency or agencies.
    
    Filters out all data associated with the specified agency_id(s), ensuring
    the resulting transit data remains valid by removing orphaned stops and
    maintaining referential integrity. Modifies transit_data in place.
    
    Args:
        transit_data: Either a GtfsModel or Feed object to filter. Modified in place.
        agency_id: Single agency_id string or list of agency_ids to remove
    
    Example:
        >>> # Remove a single agency
        >>> drop_transit_agency(gtfs_model, "SFMTA")
        >>> 
        >>> # Remove multiple agencies
        >>> drop_transit_agency(
        ...     gtfs_model,
        ...     ["SFMTA", "AC"]
        ... )
    """
    # Convert single agency_id to list for uniform handling
    if isinstance(agency_id, str):
        agency_ids_to_remove = [agency_id]
    else:
        agency_ids_to_remove = agency_id
    
    WranglerLogger.info(f"Removing transit data for agency/agencies: {agency_ids_to_remove}")
    
    # Get data tables (references, not copies)
    routes_df = transit_data.routes
    trips_df = transit_data.trips
    stop_times_df = transit_data.stop_times
    stops_df = transit_data.stops
    is_gtfs = isinstance(transit_data, GtfsModel)
    
    # Find routes to keep (those NOT belonging to agencies being removed)
    if 'agency_id' in routes_df.columns:
        routes_to_keep = routes_df[~routes_df.agency_id.isin(agency_ids_to_remove)]
        routes_removed = len(routes_df) - len(routes_to_keep)
    else:
        # If no agency_id column in routes, log warning and keep all routes
        WranglerLogger.warning("No agency_id column found in routes table - cannot filter by agency")
        routes_to_keep = routes_df
        routes_removed = 0
    
    route_ids_to_keep = set(routes_to_keep.route_id)
    
    # Filter trips based on remaining routes
    trips_to_keep = trips_df[trips_df.route_id.isin(route_ids_to_keep)]
    trips_removed = len(trips_df) - len(trips_to_keep)
    trip_ids_to_keep = set(trips_to_keep.trip_id)
    
    # Filter stop_times based on remaining trips
    stop_times_to_keep = stop_times_df[stop_times_df.trip_id.isin(trip_ids_to_keep)]
    stop_times_removed = len(stop_times_df) - len(stop_times_to_keep)
    
    # Find stops that are still referenced
    stops_still_used = set(stop_times_to_keep.stop_id.unique())
    stops_to_keep = stops_df[stops_df.stop_id.isin(stops_still_used)]
    
    # Check if any of these stops reference parent stations
    if "parent_station" in stops_to_keep.columns:
        # Get parent stations that are referenced by kept stops
        parent_stations = stops_to_keep["parent_station"].dropna().unique()
        parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings
        
        if len(parent_stations) > 0:
            # Find parent stations that aren't already in our filtered stops
            existing_stop_ids = set(stops_to_keep.stop_id)
            missing_parent_stations = [ps for ps in parent_stations if ps not in existing_stop_ids]
            
            if len(missing_parent_stations) > 0:
                WranglerLogger.debug(
                    f"Adding back {len(missing_parent_stations)} parent stations referenced by kept stops"
                )
                
                # Get the parent station records
                parent_station_records = stops_df[
                    stops_df.stop_id.isin(missing_parent_stations)
                ]
                
                # Append parent stations to filtered stops
                stops_to_keep = pd.concat(
                    [stops_to_keep, parent_station_records], 
                    ignore_index=True
                )
    
    stops_removed = len(stops_df) - len(stops_to_keep)
    
    WranglerLogger.info(
        f"Removed: {routes_removed:,} routes, {trips_removed:,} trips, "
        f"{stop_times_removed:,} stop_times, {stops_removed:,} stops"
    )
    
    WranglerLogger.info(
        f"Remaining: {len(routes_to_keep):,} routes, {len(trips_to_keep):,} trips, "
        f"{len(stops_to_keep):,} stops"
    )
    WranglerLogger.debug(f"Stops removed:\n{stops_df.loc[~stops_df['stop_id'].isin(stops_still_used)]}")
    
    # Update tables in place, in order so that validation is ok
    transit_data.stop_times = stop_times_to_keep
    transit_data.trips = trips_to_keep
    transit_data.routes = routes_to_keep
    transit_data.stops = stops_to_keep
    
    # Handle agency table
    if hasattr(transit_data, 'agency') and transit_data.agency is not None:
        # Keep agencies that are NOT being removed
        filtered_agency = transit_data.agency[
            ~transit_data.agency.agency_id.isin(agency_ids_to_remove)
        ]
        WranglerLogger.info(
            f"Removed {len(transit_data.agency) - len(filtered_agency):,} agencies"
        )
        transit_data.agency = filtered_agency
    
    # Handle shapes table
    if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
        # Keep only shapes referenced by remaining trips
        if 'shape_id' in trips_to_keep.columns:
            shape_ids = set(trips_to_keep.shape_id.dropna().unique())
            filtered_shapes = transit_data.shapes[
                transit_data.shapes.shape_id.isin(shape_ids)
            ]
            WranglerLogger.info(
                f"Removed {len(transit_data.shapes) - len(filtered_shapes):,} shape points"
            )
            transit_data.shapes = filtered_shapes
    
    # Handle calendar table
    if hasattr(transit_data, 'calendar') and transit_data.calendar is not None:
        # Keep only service_ids referenced by remaining trips
        service_ids = set(trips_to_keep.service_id.unique())
        transit_data.calendar = transit_data.calendar[
            transit_data.calendar.service_id.isin(service_ids)
        ]
    
    # Handle calendar_dates table
    if hasattr(transit_data, 'calendar_dates') and transit_data.calendar_dates is not None:
        # Keep only service_ids referenced by remaining trips
        service_ids = set(trips_to_keep.service_id.unique())
        transit_data.calendar_dates = transit_data.calendar_dates[
            transit_data.calendar_dates.service_id.isin(service_ids)
        ]
    
    # Handle frequencies table
    if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
        # Keep only frequencies for remaining trips
        transit_data.frequencies = transit_data.frequencies[
            transit_data.frequencies.trip_id.isin(trip_ids_to_keep)
        ]


def truncate_route_at_stop(
    transit_data: Union[GtfsModel, Feed],
    route_id: str,
    direction_id: int,
    stop_id: Union[str, int],
    truncate: Literal["before", "after"]
) -> None:
    """Truncate all trips of a route at a specific stop.
    
    Removes stops before or after the specified stop for all trips matching
    the given route_id and direction_id. This is useful for shortening routes
    at terminal stations or service boundaries. Modifies transit_data in place.
    
    Args:
        transit_data: Either a GtfsModel or Feed object to modify. Modified in place.
        route_id: The route_id to truncate
        direction_id: The direction_id of trips to truncate (0 or 1)
        stop_id: The stop where truncation occurs. For GtfsModel, this should be
                a string stop_id. For Feed, this should be an integer model_node_id.
        truncate: Either "before" to remove stops before stop_id, or
                 "after" to remove stops after stop_id
        
    Raises:
        ValueError: If truncate is not "before" or "after"
        ValueError: If stop_id is not found in any trips of the route/direction
        
    Example:
        >>> # Truncate outbound BART trips to end at Embarcadero (GtfsModel)
        >>> truncate_route_at_stop(
        ...     gtfs_model,
        ...     route_id="BART-01",
        ...     direction_id=0,
        ...     stop_id="EMBR",  # string stop_id
        ...     truncate="after"
        ... )
        >>> 
        >>> # Truncate outbound BART trips to end at node 12345 (Feed)
        >>> truncate_route_at_stop(
        ...     feed,
        ...     route_id="BART-01",
        ...     direction_id=0,
        ...     stop_id=12345,  # integer model_node_id
        ...     truncate="after"
        ... )
    """
    if truncate not in ["before", "after"]:
        msg = f"truncate must be 'before' or 'after', got '{truncate}'"
        raise ValueError(msg)
        
    WranglerLogger.info(
        f"Truncating route {route_id} direction {direction_id} {truncate} stop {stop_id}"
    )
    
    # Get data tables (references, not copies)
    routes_df = transit_data.routes
    trips_df = transit_data.trips
    stop_times_df = transit_data.stop_times
    stops_df = transit_data.stops
    is_gtfs = isinstance(transit_data, GtfsModel)
    
    # Find trips to truncate
    trips_to_truncate = trips_df[
        (trips_df.route_id == route_id) & 
        (trips_df.direction_id == direction_id)
    ]
    
    if len(trips_to_truncate) == 0:
        WranglerLogger.warning(
            f"No trips found for route {route_id} direction {direction_id}"
        )
        return  # No changes needed
    
    trip_ids_to_truncate = set(trips_to_truncate.trip_id)
    WranglerLogger.debug(f"Found {len(trip_ids_to_truncate)} trips to truncate")
    
    # Check if stop_id exists in any of these trips
    stop_times_for_route = stop_times_df[
        (stop_times_df.trip_id.isin(trip_ids_to_truncate)) &
        (stop_times_df.stop_id == stop_id)
    ]
    
    if len(stop_times_for_route) == 0:
        msg = f"Stop {stop_id} not found in any trips of route {route_id} direction {direction_id}"
        raise ValueError(msg)
    
    # Process stop_times to truncate trips
    truncated_stop_times = []
    trips_truncated = 0
    
    for trip_id in trip_ids_to_truncate:
        trip_stop_times = stop_times_df[stop_times_df.trip_id == trip_id]
        trip_stop_times = trip_stop_times.sort_values('stop_sequence')
        
        # Find the stop_sequence for the truncation stop
        stop_mask = trip_stop_times.stop_id == stop_id
        if not stop_mask.any():
            # This trip doesn't have the stop, keep all stops
            truncated_stop_times.append(trip_stop_times)
            continue
            
        stop_sequence_at_stop = trip_stop_times.loc[stop_mask, 'stop_sequence'].iloc[0]
        
        # Truncate based on direction
        if truncate == "before":
            # Keep stops from stop_id onwards
            truncated_stops = trip_stop_times[
                trip_stop_times.stop_sequence >= stop_sequence_at_stop
            ].copy()  # Need copy here since we'll modify stop_sequence
        else:  # truncate == "after"
            # Keep stops up to and including stop_id
            truncated_stops = trip_stop_times[
                trip_stop_times.stop_sequence <= stop_sequence_at_stop
            ].copy()  # Need copy here since we'll modify stop_sequence
        
        # Renumber stop_sequence to be consecutive starting from 0
        if len(truncated_stops) > 0:
            truncated_stops['stop_sequence'] = range(len(truncated_stops))
        
        # Log truncation details
        original_count = len(trip_stop_times)
        truncated_count = len(truncated_stops)
        if truncated_count < original_count:
            trips_truncated += 1
            
            # Get removed stops details
            removed_stop_ids = set(trip_stop_times.stop_id) - set(truncated_stops.stop_id)
            if removed_stop_ids and len(removed_stop_ids) <= 10:
                # Get stop names for removed stops
                removed_stops_info = stops_df[stops_df.stop_id.isin(removed_stop_ids)][['stop_id', 'stop_name']]
                removed_stops_list = [f"{row['stop_id']} ({row['stop_name']})" 
                                     for _, row in removed_stops_info.iterrows()]
                
                WranglerLogger.debug(
                    f"Trip {trip_id}: truncated from {original_count} to {truncated_count} stops. "
                    f"Removed: {', '.join(removed_stops_list)}"
                )
            else:
                WranglerLogger.debug(
                    f"Trip {trip_id}: truncated from {original_count} to {truncated_count} stops"
                )
        
        truncated_stop_times.append(truncated_stops)
    
    WranglerLogger.info(f"Truncated {trips_truncated} trips")
    
    # Combine all stop times (truncated and non-truncated)
    other_stop_times = stop_times_df[~stop_times_df.trip_id.isin(trip_ids_to_truncate)]
    all_stop_times = pd.concat([other_stop_times] + truncated_stop_times, ignore_index=True)
    
    # Find stops that are still referenced
    stops_still_used = set(all_stop_times.stop_id.unique())
    filtered_stops = stops_df[stops_df.stop_id.isin(stops_still_used)]
    
    # Check if any of these stops reference parent stations
    if "parent_station" in filtered_stops.columns:
        # Get parent stations that are referenced by kept stops
        parent_stations = filtered_stops["parent_station"].dropna().unique()
        parent_stations = [ps for ps in parent_stations if ps != ""]  # Remove empty strings
        
        if len(parent_stations) > 0:
            # Find parent stations that aren't already in our filtered stops
            existing_stop_ids = set(filtered_stops.stop_id)
            missing_parent_stations = [ps for ps in parent_stations if ps not in existing_stop_ids]
            
            if len(missing_parent_stations) > 0:
                WranglerLogger.debug(
                    f"Adding back {len(missing_parent_stations)} parent stations referenced by kept stops"
                )
                
                # Get the parent station records
                parent_station_records = stops_df[
                    stops_df.stop_id.isin(missing_parent_stations)
                ]
                
                # Append parent stations to filtered stops
                filtered_stops = pd.concat(
                    [filtered_stops, parent_station_records], 
                    ignore_index=True
                )
    
    # Log removed stops
    removed_stops = set(stops_df.stop_id) - set(filtered_stops.stop_id)
    if removed_stops:
        WranglerLogger.debug(f"Removed {len(removed_stops)} stops that are no longer referenced")
        
        # Get details of removed stops
        removed_stops_df = stops_df[stops_df.stop_id.isin(removed_stops)][['stop_id', 'stop_name']]
        
        # Log up to 20 removed stops with their names
        sample_size = min(20, len(removed_stops_df))
        for _, stop in removed_stops_df.head(sample_size).iterrows():
            WranglerLogger.debug(f"  - Removed stop: {stop['stop_id']} ({stop['stop_name']})")
        
        if len(removed_stops) > sample_size:
            WranglerLogger.debug(f"  ... and {len(removed_stops) - sample_size} more stops")
    
    # Update transit_data in place
    transit_data.stop_times = all_stop_times
    transit_data.trips = trips_df
    transit_data.routes = routes_df
    transit_data.stops = filtered_stops
    
    # Note: shapes would need to be truncated to match truncated trips
    # TODO: truncate shapes to match truncated trips
