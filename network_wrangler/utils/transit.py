"""Utilities for getting GTFS into wrangler"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..logger import WranglerLogger
from ..models.gtfs.converters import (
    convert_stop_times_to_wrangler_stop_times,
    convert_stops_to_wrangler_stops,
)
from ..models.gtfs.gtfs import GtfsModel
from ..roadway.network import RoadwayNetwork
from ..transit.feed.feed import Feed
from .time import seconds_to_time, time_to_seconds

# GTFS route types that operate at street level and need drive-accessible stops
# Route type 0: Tram, Streetcar, Light rail - operates in mixed traffic
# Route type 3: Bus - operates on streets
# Route type 5: Cable tram/Cable car - street-level rail with underground cable
# Route type 11: Trolleybus - electric buses with overhead wires
STREET_LEVEL_ROUTE_TYPES = [0, 3, 5, 11]

# GTFS route types that should use connectivity-aware matching
# Route type 1: Subway, Metro
# Route type 2: Rail
# Route type 4: Ferry
CONNECTIVITY_MATCH_ROUTE_TYPES = [1, 2, 4]

# Maximum distance in miles for a stop to match to a node
MAX_STOP_DISTANCE_MILES = 0.25

# Feet to miles conversion
FEET_PER_MILE = 5280.0
def create_frequencies_from_stop_times(
    feed_tables: Dict[str, pd.DataFrame],
    time_periods: List[Dict[str, str]],
    default_frequency_for_onetime_route: int = 10800,
) -> pd.DataFrame:
    """Create a frequencies table from feed stop_times data.

    This function analyzes actual trip departure times in the feed data and calculates
    average headways for each route pattern within specified time periods.

    Args:
        feed_tables (Dict[str, pd.DataFrame]): Dictionary of feed tables including:
            - 'stop_times': Wrangler-formatted stop times table
            - 'trips': Trips table with route information
        time_periods (List[Dict[str, str]]): List of time period definitions.
            Each dict should have 'start_time' and 'end_time' keys.
            Example: [{'start_time': '06:00:00', 'end_time': '10:00:00'}, ...]
        default_frequency_for_onetime_route (int, optional): Default headway in seconds
            for routes with only one trip in a period. Defaults to 10800 (3 hours).

    Returns:
        pd.DataFrame: Frequencies table with columns:
            - trip_id: Template trip ID for this frequency entry
            - start_time: Start time of the period
            - end_time: End time of the period
            - headway_secs: Average headway in seconds
            - exact_times: Always 0 (frequency-based service)

    Notes:
        - Groups trips by route and stop pattern to handle different service patterns
        - For routes with only one trip in a period, uses the default_frequency_for_onetime_route
        - Handles time periods that cross midnight (e.g., '19:00:00' to '03:00:00')
        - Returns empty DataFrame if no frequency data can be calculated
    """
    WranglerLogger.info("Creating frequencies table from feed stop_times data")

    stop_times_df = feed_tables["stop_times"].copy()
    trips_df = feed_tables["trips"].copy()

    stop_times_df["departure_seconds"] = stop_times_df["departure_time"].apply(time_to_seconds)

    # Get first stop for each trip to use as the reference time
    first_stops = stop_times_df.groupby("trip_id")["departure_seconds"].min().reset_index()
    first_stops.columns = ["trip_id", "first_departure_seconds"]

    # Join with trips to get route info
    trips_with_times = pd.merge(
        first_stops, trips_df[["trip_id", "route_id"]], on="trip_id", how="left"
    )

    frequencies_data = []

    # Get stop patterns for all trips to identify distinct patterns
    # Note: In wrangler format, model_node_id is renamed to stop_id
    stop_sequences = stop_times_df.groupby("trip_id")["stop_id"].apply(list).reset_index()
    stop_sequences.columns = ["trip_id", "stop_sequence"]
    stop_sequences["stop_pattern"] = stop_sequences["stop_sequence"].apply(
        lambda x: ",".join(map(str, x))
    )

    # Join with trip times
    trips_with_patterns = pd.merge(
        trips_with_times, stop_sequences[["trip_id", "stop_pattern"]], on="trip_id", how="left"
    )
    WranglerLogger.debug(
        f"create_frequencies_from_stop_times(): trips_with_patterns=\n{trips_with_patterns}"
    )

    # Process each route
    for route_id, route_trips in trips_with_patterns.groupby("route_id"):
        if len(route_trips) == 0:
            WranglerLogger.debug(f"Route {route_id}: no trips found, skipping")
            continue

        # Find distinct stop patterns for this route
        pattern_groups = route_trips.groupby("stop_pattern")
        num_patterns = len(pattern_groups)

        if num_patterns > 1:
            WranglerLogger.info(f"Route {route_id}: Found {num_patterns} distinct stop patterns")

        # Process each pattern separately
        for pattern_idx, (stop_pattern, pattern_trips) in enumerate(pattern_groups):
            if len(pattern_trips) == 0:
                continue

            # Sort trips by departure time
            pattern_trips = pattern_trips.sort_values("first_departure_seconds")
            departures = pattern_trips["first_departure_seconds"].values

            # Use first trip of this pattern as template
            template_trip_id = pattern_trips.iloc[0]["trip_id"]

            # Log pattern info
            num_stops = len(stop_pattern.split(","))
            WranglerLogger.debug(
                f"Route {route_id} pattern {pattern_idx + 1}/{num_patterns}: {len(pattern_trips)} trips, {num_stops} stops, template trip {template_trip_id}"
            )

            # Calculate headways for each time period
            for period in time_periods:
                start_seconds = time_to_seconds(period["start_time"])
                end_seconds = time_to_seconds(period["end_time"])

                # Handle periods that cross midnight
                if end_seconds <= start_seconds:
                    # Period crosses midnight
                    period_departures = departures[
                        (departures >= start_seconds) | (departures < end_seconds)
                    ]
                else:
                    # Normal period
                    period_departures = departures[
                        (departures >= start_seconds) & (departures < end_seconds)
                    ]

                if len(period_departures) == 0:
                    WranglerLogger.debug(
                        f"Route {route_id} pattern {pattern_idx + 1}: no departures in period {period['start_time']}-{period['end_time']}, skipping"
                    )
                    continue
                if len(period_departures) == 1:
                    # For single trip, use default headway
                    avg_headway = default_frequency_for_onetime_route
                    WranglerLogger.debug(
                        f"Route {route_id} pattern {pattern_idx + 1}: only 1 departure in period {period['start_time']}-{period['end_time']}, using default {default_frequency_for_onetime_route / 3600:.1f}-hour headway"
                    )
                else:
                    # Calculate average headway for multiple trips
                    headways = np.diff(np.sort(period_departures))
                    avg_headway = int(np.mean(headways))
                    WranglerLogger.debug(
                        f"Route {route_id} pattern {pattern_idx + 1}: {len(period_departures)} departure in period {period['start_time']}-{period['end_time']}, period_departures={period_departures} headways={headways} avg_headway={avg_headway}"
                    )
                    # this can happen if there are two trips and they leave at the same times
                    if avg_headway == 0:
                        avg_headway = default_frequency_for_onetime_route

                frequencies_data.append(
                    {
                        "trip_id": template_trip_id,
                        "start_time": period["start_time"],
                        "end_time": period["end_time"],
                        "headway_secs": avg_headway,
                        "exact_times": 0,
                    }
                )

                WranglerLogger.debug(
                    f"Route {route_id} pattern {pattern_idx + 1} period {period['start_time']}-{period['end_time']}: {len(period_departures)} trips, avg headway {avg_headway // 60:.1f} min"
                )

    if frequencies_data:
        frequencies_df = pd.DataFrame(frequencies_data)
        WranglerLogger.info(
            f"Created frequencies table with {len(frequencies_df)} entries from stop_times data"
        )
        WranglerLogger.debug(
            f"frequencies_df.loc[frequencies_df.trip_id=='AC:11539040:20231031']:\n{frequencies_df.loc[frequencies_df.trip_id == 'AC:11539040:20231031']}"
        )
        WranglerLogger.debug(
            f"feed_tables['trips'].loc[ feed_tables['trips'].trip_id=='AC:11539040:20231031']:\n{feed_tables['trips'].loc[feed_tables['trips'].trip_id == 'AC:11539040:20231031']}"
        )
        WranglerLogger.debug(f"feed_tables['stop_times']:\n{feed_tables['stop_times']}")
        return frequencies_df
    WranglerLogger.warning("No frequency data could be calculated from stop_times")
    return pd.DataFrame()


def analyze_frequency_coverage(
    gtfs_model: GtfsModel, frequencies_df: pd.DataFrame
) -> pd.DataFrame:
    """Analyze how well template trips in frequencies table cover all trips.

    Args:
        gtfs_model (GtfsModel): GTFS model with trips and stop_times data
        frequencies_df (pd.DataFrame): Frequencies table with template trip_ids

    Returns:
        pd.DataFrame: Coverage statistics by route with columns:
            - route_id: Route identifier
            - total_trips: Total number of trips for the route
            - total_patterns: Total number of distinct stop patterns
            - covered_patterns: Number of patterns covered by template trips
            - covered_trips: Number of trips covered by template patterns
            - coverage_pct: Percentage of trips covered
    """
    # Get stop patterns for all trips
    stop_sequences = gtfs_model.stop_times.groupby("trip_id")["stop_id"].apply(list).reset_index()
    stop_sequences.columns = ["trip_id", "stop_sequence"]
    stop_sequences["stop_pattern"] = stop_sequences["stop_sequence"].apply(lambda x: ",".join(x))

    # Get all trips and their patterns
    all_trip_patterns = pd.merge(
        gtfs_model.trips[["trip_id", "route_id"]],
        stop_sequences[["trip_id", "stop_pattern"]],
        on="trip_id",
        how="left",
    )

    # Get template trips and their patterns
    template_trip_patterns = all_trip_patterns[
        all_trip_patterns["trip_id"].isin(frequencies_df["trip_id"].unique())
    ]

    # Analyze coverage by route
    coverage_stats = []
    for route_id in all_trip_patterns["route_id"].unique():
        route_all_trips = all_trip_patterns[all_trip_patterns["route_id"] == route_id]
        route_template_trips = template_trip_patterns[
            template_trip_patterns["route_id"] == route_id
        ]

        total_trips = len(route_all_trips)
        total_patterns = route_all_trips["stop_pattern"].nunique()
        covered_patterns = route_template_trips["stop_pattern"].nunique()

        # Count trips covered by template patterns
        covered_trip_count = route_all_trips[
            route_all_trips["stop_pattern"].isin(route_template_trips["stop_pattern"])
        ]["trip_id"].nunique()

        coverage_pct = (covered_trip_count / total_trips * 100) if total_trips > 0 else 0

        coverage_stats.append(
            {
                "route_id": route_id,
                "total_trips": total_trips,
                "total_patterns": total_patterns,
                "covered_patterns": covered_patterns,
                "covered_trips": covered_trip_count,
                "coverage_pct": coverage_pct,
            }
        )

        if covered_patterns < total_patterns:
            WranglerLogger.warning(
                f"Route {route_id}: {covered_patterns}/{total_patterns} stop patterns covered by template trips ({coverage_pct:.1f}% of trips)"
            )

    # Summary statistics
    coverage_df = pd.DataFrame(coverage_stats)
    avg_coverage = coverage_df["coverage_pct"].mean()
    routes_with_incomplete_coverage = len(coverage_df[coverage_df["coverage_pct"] < 100])

    WranglerLogger.info(
        f"Pattern coverage summary: {avg_coverage:.1f}% average trip coverage across all routes"
    )
    if routes_with_incomplete_coverage > 0:
        WranglerLogger.warning(
            f"{routes_with_incomplete_coverage} routes have incomplete pattern coverage"
        )

    return coverage_df


def build_transit_graph(transit_links_df: pd.DataFrame) -> Dict[int, set]:
    """Build directed adjacency graph from transit links.
    
    Args:
        transit_links_df: DataFrame of links with transit=True, must have columns A and B
        
    Returns:
        Dict mapping node_id to set of reachable node_ids (following link direction)
    """
    graph = {}
    for _, link in transit_links_df.iterrows():
        node_a, node_b = link['A'], link['B']
        if node_a not in graph:
            graph[node_a] = set()
        if node_b not in graph:
            graph[node_b] = set()
        # Only add edge from A to B (directed)
        graph[node_a].add(node_b)
    return graph


def match_route_stops_with_connectivity(
    route_stops_df: pd.DataFrame,
    candidate_nodes_gdf: gpd.GeoDataFrame,
    transit_graph: Dict[int, set],
    stops_gdf_proj: gpd.GeoDataFrame,
    max_distance_ft: float = 5000
) -> Tuple[Dict[str, int], bool, Dict[str, Any]]:
    """Match stops for a route considering transit link connectivity.
    
    Args:
        route_stops_df: DataFrame with stops for one route, ordered by stop_sequence
        candidate_nodes_gdf: GeoDataFrame of candidate nodes (projected coordinates)
        transit_graph: Adjacency graph of transit links
        stops_gdf_proj: GeoDataFrame of all stops with projected coordinates and all stop attributes
        max_distance_ft: Maximum allowed distance from stop to node
        
    Returns:
        Tuple of:
        - Dict mapping stop_id to model_node_id
        - bool indicating if connectivity matching succeeded
        - Dict with failure details (if failed)
    """
    WranglerLogger.debug(f"==== match_route_stops_with_connectivity() ===")
    WranglerLogger.debug(f"route_stops_df:\n{route_stops_df}")

    stop_matches = {}
    failure_info = {}
    
    # Get ordered list of stops with names
    ordered_stops_df = route_stops_df.sort_values('stop_sequence')
    ordered_stops = ordered_stops_df['stop_id'].tolist()
    
    if len(ordered_stops) == 0:
        return stop_matches, True, failure_info
    
    # Get stop names for logging
    stop_info = {}
    for stop_id in ordered_stops:
        stop_data = stops_gdf_proj[stops_gdf_proj['stop_id'] == stop_id]
        if len(stop_data) > 0:
            stop_info[stop_id] = stop_data.iloc[0]['stop_name']
        else:
            stop_info[stop_id] = f"Unknown ({stop_id})"
    
    # For each stop, find candidate nodes within max_distance
    stop_candidates = {}
    for stop_id in ordered_stops:
        stop_geom = stops_gdf_proj.loc[stops_gdf_proj['stop_id'] == stop_id, 'geometry'].iloc[0]
        distances = candidate_nodes_gdf.geometry.distance(stop_geom)
        nearby_mask = distances <= max_distance_ft
        nearby_nodes = candidate_nodes_gdf[nearby_mask]['model_node_id'].tolist()
        stop_candidates[stop_id] = nearby_nodes
    
    # Try to find a connected path through the candidates
    # Start with the first stop and try each candidate
    first_stop = ordered_stops[0]
    best_failure_info = None
    
    for first_node in stop_candidates.get(first_stop, []):
        matches = {first_stop: first_node}
        current_node = first_node
        route_progress = [(0, first_stop, first_node, True)]
        
        # Try to match subsequent stops
        success = True
        failure_idx = -1
        
        for i in range(1, len(ordered_stops)):
            stop_id = ordered_stops[i]
            candidates = stop_candidates.get(stop_id, [])
            
            # Find candidates that are connected to current_node
            connected_candidates = [
                node for node in candidates 
                if node in transit_graph.get(current_node, set())
            ]
            
            if connected_candidates:
                # Choose the closest connected candidate
                stop_geom = stops_gdf_proj.loc[stops_gdf_proj['stop_id'] == stop_id, 'geometry'].iloc[0]
                best_node = min(connected_candidates, 
                              key=lambda n: stop_geom.distance(
                                  candidate_nodes_gdf[candidate_nodes_gdf['model_node_id'] == n].geometry.iloc[0]
                              ))
                matches[stop_id] = best_node
                route_progress.append((i, stop_id, best_node, True))
                current_node = best_node
            else:
                # No connected candidate found
                success = False
                failure_idx = i
                route_progress.append((i, stop_id, None, False))
                
                # Track failure details
                this_failure_info = {
                    'failed_at_stop': i + 1,  # 1-based for user display
                    'total_stops': len(ordered_stops),
                    'last_matched_stop': ordered_stops[i-1] if i > 0 else None,
                    'last_matched_node': current_node,
                    'failed_stop': stop_id,
                    'failed_stop_name': stop_info[stop_id],
                    'candidates_at_failure': len(candidates),
                    'route_progress': route_progress,
                    'stop_info': stop_info
                }
                
                # Keep the failure that got furthest
                if best_failure_info is None or i > best_failure_info['failed_at_stop'] - 1:
                    best_failure_info = this_failure_info
                break
        
        if success:
            return matches, True, {}  # Return True to indicate connectivity matching succeeded
    
    # Use the best failure info if we have one
    if best_failure_info:
        failure_info = best_failure_info
    else:
        # If we didn't even try (no candidates for first stop), create a failure info
        failure_info = {
            'failed_at_stop': 1,
            'total_stops': len(ordered_stops),
            'last_matched_stop': None,
            'last_matched_node': None,
            'failed_stop': first_stop,
            'failed_stop_name': stop_info[first_stop],
            'candidates_at_failure': len(stop_candidates.get(first_stop, [])),
            'route_progress': [(0, first_stop, None, False)],
            'stop_info': stop_info
        }
    
    # If no connected path found, fall back to nearest node matching with distance threshold
    # Also build complete route info showing what would have been matched
    complete_route_progress = []
    stop_distances = {}
    for i, stop_id in enumerate(ordered_stops):
        stop_geom = stops_gdf_proj.loc[stops_gdf_proj['stop_id'] == stop_id, 'geometry'].iloc[0]
        distances = candidate_nodes_gdf.geometry.distance(stop_geom)
        nearest_idx = distances.idxmin()
        nearest_node = candidate_nodes_gdf.loc[nearest_idx, 'model_node_id']
        nearest_distance = distances[nearest_idx]
        
        # Only match if within threshold
        if nearest_distance <= max_distance_ft:
            stop_matches[stop_id] = nearest_node
        else:
            # Log that this stop is too far
            WranglerLogger.debug(
                f"Stop {stop_id} is {nearest_distance / FEET_PER_MILE:.2f} mi from nearest node - not matching"
            )
            nearest_node = None  # Mark as unmatched in the route progress
            
        stop_distances[stop_id] = nearest_distance
        
        # Mark whether this stop was part of the connected path attempt
        was_connected = i < len(failure_info.get('route_progress', []))
        complete_route_progress.append((i, stop_id, nearest_node, was_connected))
    
    # Update failure info with complete route and distances
    failure_info['complete_route'] = complete_route_progress
    failure_info['stop_distances'] = stop_distances
    
    return stop_matches, False, failure_info  # Return failure info


def create_feed_from_gtfs_model(
    gtfs_model: GtfsModel,
    roadway_net: RoadwayNetwork,
    time_periods: Optional[List[Dict[str, str]]] = None,
    default_frequency_for_onetime_route: int = 10800,
) -> Feed:
    """Converts a GTFS feed to a Wrangler Feed object compatible with the given RoadwayNetwork.

    Args:
        gtfs_model (GtfsModel): Standard GTFS model to convert
        roadway_net (RoadwayNetwork): RoadwayNetwork to map stops to
        time_periods (Optional[List[Dict[str, str]]]): List of time period definitions for frequencies.
            Each dict should have 'start_time' and 'end_time' keys.
            Example: [{'start_time': '03:00:00', 'end_time': '06:00:00'}, ...]
            If None, frequencies table will not be created from stop_times.
        default_frequency_for_onetime_route (int, optional): Default headway in seconds
            for routes with only one trip in a period. Defaults to 10800 (3 hours).

    Returns:
        Feed: Wrangler Feed object with stops mapped to roadway network nodes
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")

    # Start with the tables from the GTFS model
    feed_tables = {}

    # Copy over standard tables that don't need modification
    # GtfsModel guarantees routes and trips exist
    feed_tables["routes"] = gtfs_model.routes.copy()
    feed_tables["trips"] = gtfs_model.trips.copy()

    if hasattr(gtfs_model, "agencies") and gtfs_model.agencies is not None:
        feed_tables["agencies"] = gtfs_model.agencies.copy()

    # Get all nodes and links for spatial matching
    all_nodes_df = roadway_net.nodes_df.copy()
    links_df = roadway_net.links_df

    # Prepare different node sets for different route types
    drive_accessible_nodes = None
    transit_accessible_nodes = None
    
    if "drive_access" in links_df.columns:
        # Get nodes that are connected by drive-accessible links (for street-level transit)
        drive_links = links_df[links_df["drive_access"] == True]
        drive_accessible_node_ids = set(drive_links["A"].unique()) | set(drive_links["B"].unique())
        drive_accessible_nodes = all_nodes_df[
            all_nodes_df["model_node_id"].isin(drive_accessible_node_ids)
        ].copy()
        WranglerLogger.info(
            f"Found {len(drive_accessible_nodes):,} drive-accessible nodes (for street-level transit) out of {len(all_nodes_df):,} total"
        )
    else:
        WranglerLogger.warning(
            "No drive_access column found in links, all nodes will be used for all route types"
        )
        drive_accessible_nodes = all_nodes_df.copy()
    
    # Get nodes connected by transit links (for rail/subway)
    if "transit" in links_df.columns:
        transit_links = links_df[links_df["transit"] == True]
        transit_accessible_node_ids = set(transit_links["A"].unique()) | set(transit_links["B"].unique())
        transit_accessible_nodes = all_nodes_df[
            all_nodes_df["model_node_id"].isin(transit_accessible_node_ids)
        ].copy()
        WranglerLogger.info(
            f"Found {len(transit_accessible_nodes):,} transit-accessible nodes (for rail/subway) out of {len(all_nodes_df):,} total"
        )
    else:
        WranglerLogger.info(
            "No transit column found in links, all nodes will be used for non-street transit"
        )
        transit_accessible_nodes = all_nodes_df.copy()

    # create mapping from gtfs_model stop to RoadwayNetwork nodes
    # GtfsModel guarantees stops exists
    stops_df = gtfs_model.stops.copy()

    # Determine which stops are used by street-level transit vs other route types
    # Need to join stops -> stop_times -> trips -> routes to get route types
    WranglerLogger.info("Determining route types that serve each stop")

    # GtfsModel guarantees stops, stop_times, trips and routes exist
    WranglerLogger.debug(
        f"Processing {len(gtfs_model.stops):,} stops, {len(gtfs_model.stop_times):,} stop_times, {len(gtfs_model.trips):,} trips, {len(gtfs_model.routes):,} routes"
    )

    # Join stop_times with trips and routes
    stop_trips = pd.merge(
        gtfs_model.stop_times[["stop_id", "trip_id"]].drop_duplicates(),
        gtfs_model.trips[["trip_id", "route_id"]],
        on="trip_id",
        how="left",
    )
    WranglerLogger.debug(f"After joining stop_times with trips: {len(stop_trips):,} records")

    stop_route_types = pd.merge(
        stop_trips, gtfs_model.routes[["route_id", "route_type"]], on="route_id", how="left"
    )[["stop_id", "route_type"]].drop_duplicates()
    WranglerLogger.debug(
        f"After joining with routes: {len(stop_route_types):,} unique stop-route_type combinations"
    )
    WranglerLogger.debug(f"stop_route_types:\n{stop_route_types}")

    # Log route type distribution
    route_type_counts = stop_route_types["route_type"].value_counts()
    WranglerLogger.debug("Route type distribution in stop_route_types:")
    for rt, count in route_type_counts.items():
        WranglerLogger.debug(f"  Route type {rt}: {count:,} stop-route combinations")

    # Also create stop to agency mapping
    stop_agencies = pd.merge(
        stop_trips, 
        gtfs_model.routes[["route_id", "agency_id", "route_short_name"]], 
        on="route_id", 
        how="left"
    )[["stop_id", "agency_id", "route_short_name"]].drop_duplicates()
    
    # If agency table exists, merge in agency names
    if hasattr(gtfs_model, "agency") and gtfs_model.agency is not None:
        stop_agencies = pd.merge(
            stop_agencies,
            gtfs_model.agency[["agency_id", "agency_name"]],
            on="agency_id",
            how="left"
        )
    else:
        stop_agencies["agency_name"] = None
    
    # Group by stop to get all agencies and routes serving each stop
    stop_agency_info = stop_agencies.groupby("stop_id").agg({
        "agency_id": lambda x: list(x.dropna().unique()),
        "agency_name": lambda x: list(x.dropna().unique()) if x.notna().any() else [],
        "route_short_name": lambda x: list(x.dropna().unique())
    }).reset_index()
    stop_agency_info.columns = ["stop_id", "agency_ids", "agency_names", "route_names"]

    # Group by stop to find which route types serve each stop
    stop_route_types_agg = (
        stop_route_types.groupby("stop_id")["route_type"].apply(list).reset_index()
    )
    stop_route_types_agg["has_street_transit"] = stop_route_types_agg["route_type"].apply(
        lambda x: any(rt in x for rt in STREET_LEVEL_ROUTE_TYPES)
    )  # Check for street-level route types

    # Merge back to stops
    stops_df = pd.merge(
        stops_df, stop_route_types_agg[["stop_id", "has_street_transit"]], on="stop_id", how="left"
    )
    # Also merge agency information
    stops_df = pd.merge(
        stops_df, stop_agency_info, on="stop_id", how="left"
    )

    # Check for stops without route type info
    unmatched_stops = stops_df[stops_df["has_street_transit"].isna()]
    if len(unmatched_stops) > 0:
        WranglerLogger.warning(
            f"{len(unmatched_stops)} stops have no route type information (not found in stop_times)"
        )
        WranglerLogger.debug(
            f"Example unmatched stops: {unmatched_stops['stop_id'].head().tolist()}"
        )

    stops_df["has_street_transit"] = stops_df["has_street_transit"].fillna(False)

    street_transit_stops = stops_df[stops_df["has_street_transit"]]
    non_street_transit_stops = stops_df[~stops_df["has_street_transit"]]
    WranglerLogger.info(
        f"Found {len(street_transit_stops):,} stops served by street-level transit, {len(non_street_transit_stops):,} stops served by other modes"
    )

    # Log some examples
    if len(street_transit_stops) > 0:
        WranglerLogger.debug(f"Example street-level transit stops: {street_transit_stops['stop_id'].head().tolist()}")
    if len(non_street_transit_stops) > 0:
        WranglerLogger.debug(f"Example non-street transit stops: {non_street_transit_stops['stop_id'].head().tolist()}")

    # Create GeoDataFrames for spatial matching
    stop_geometry = [
        Point(lon, lat) for lon, lat in zip(stops_df["stop_lon"], stops_df["stop_lat"])
    ]
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=stop_geometry, crs="EPSG:4326")

    # Project to local coordinate system
    stops_gdf_proj = stops_gdf.to_crs("EPSG:2227")

    # Prepare node GeoDataFrames
    if "geometry" not in all_nodes_df.columns:
        node_geometry = [Point(x, y) for x, y in zip(all_nodes_df["X"], all_nodes_df["Y"])]
        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, geometry=node_geometry, crs="EPSG:4326")
    else:
        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, crs="EPSG:4326")

    if "geometry" not in drive_accessible_nodes.columns:
        node_geometry = [
            Point(x, y) for x, y in zip(drive_accessible_nodes["X"], drive_accessible_nodes["Y"])
        ]
        drive_nodes_gdf = gpd.GeoDataFrame(
            drive_accessible_nodes, geometry=node_geometry, crs="EPSG:4326"
        )
    else:
        drive_nodes_gdf = gpd.GeoDataFrame(drive_accessible_nodes, crs="EPSG:4326")
        
    if "geometry" not in transit_accessible_nodes.columns:
        node_geometry = [
            Point(x, y) for x, y in zip(transit_accessible_nodes["X"], transit_accessible_nodes["Y"])
        ]
        transit_nodes_gdf = gpd.GeoDataFrame(
            transit_accessible_nodes, geometry=node_geometry, crs="EPSG:4326"
        )
    else:
        transit_nodes_gdf = gpd.GeoDataFrame(transit_accessible_nodes, crs="EPSG:4326")

    all_nodes_gdf_proj = all_nodes_gdf.to_crs("EPSG:2227")
    drive_nodes_gdf_proj = drive_nodes_gdf.to_crs("EPSG:2227")
    transit_nodes_gdf_proj = transit_nodes_gdf.to_crs("EPSG:2227")

    # Use spatial index for efficient nearest neighbor search
    import numpy as np
    from sklearn.neighbors import BallTree

    WranglerLogger.info("Building spatial indices for stop-to-node matching")

    # Build spatial indices
    all_node_coords = np.array([(geom.x, geom.y) for geom in all_nodes_gdf_proj.geometry])
    drive_node_coords = np.array([(geom.x, geom.y) for geom in drive_nodes_gdf_proj.geometry])
    transit_node_coords = np.array([(geom.x, geom.y) for geom in transit_nodes_gdf_proj.geometry])

    all_nodes_tree = BallTree(all_node_coords)
    drive_nodes_tree = BallTree(drive_node_coords)
    transit_nodes_tree = BallTree(transit_node_coords)

    # Initialize results
    stops_gdf_proj["model_node_id"] = None
    stops_gdf_proj["match_distance_ft"] = None
    
    # Build transit graph for connectivity-aware matching
    transit_graph = None
    connectivity_routes = []
    route_stop_patterns = {}
    
    if "transit" in links_df.columns:
        transit_links = links_df[links_df["transit"] == True]
        if len(transit_links) > 0:
            transit_graph = build_transit_graph(transit_links)
            WranglerLogger.info(f"Built transit graph with {len(transit_graph)} nodes for connectivity matching")
    
    # Handle connectivity-based matching for rail and ferry routes
    if transit_graph:
        # Get routes that need connectivity matching
        connectivity_routes = gtfs_model.routes[
            gtfs_model.routes['route_type'].isin(CONNECTIVITY_MATCH_ROUTE_TYPES)
        ]['route_id'].tolist()
        
        if connectivity_routes:
            WranglerLogger.info(f"Processing {len(connectivity_routes)} routes with connectivity-aware matching")
            
            # Get trips for these routes
            connectivity_trips = gtfs_model.trips[
                gtfs_model.trips['route_id'].isin(connectivity_routes)
            ]['trip_id'].tolist()
            
            # Get stop patterns for these trips
            connectivity_stop_times = gtfs_model.stop_times[
                gtfs_model.stop_times['trip_id'].isin(connectivity_trips)
            ]
            
            # Group by route and get unique stop patterns
            failed_connectivity_routes = []
            
            for route_id in connectivity_routes:
                route_trips = gtfs_model.trips[gtfs_model.trips['route_id'] == route_id]['trip_id']
                route_stop_times = connectivity_stop_times[
                    connectivity_stop_times['trip_id'].isin(route_trips)
                ]
                
                # Get the most common stop pattern for this route
                pattern_groups = route_stop_times.groupby('trip_id')[['stop_id', 'stop_sequence']].apply(
                    lambda x: x.sort_values('stop_sequence')
                ).reset_index(level=0, drop=True)
                
                # Use the first trip's pattern as representative
                if len(pattern_groups) > 0:
                    first_trip = route_trips.iloc[0]
                    route_stops = route_stop_times[route_stop_times['trip_id'] == first_trip][
                        ['stop_id', 'stop_sequence']
                    ].sort_values('stop_sequence')
                    route_stop_patterns[route_id] = route_stops
            
            # Match stops for each route
            for route_id, route_stops in route_stop_patterns.items():
                if len(route_stops) > 0:
                    WranglerLogger.debug(f"Matching {len(route_stops)} stops for route {route_id}")
                    
                    # Get route info
                    route_info = gtfs_model.routes[gtfs_model.routes['route_id'] == route_id].iloc[0]
                    route_name = route_info.get('route_short_name', route_id)
                    
                    # Match stops with connectivity
                    stop_matches, connectivity_success, failure_details = match_route_stops_with_connectivity(
                        route_stops,
                        transit_nodes_gdf_proj,
                        transit_graph,
                        stops_gdf_proj,
                        max_distance_ft=MAX_STOP_DISTANCE_MILES * FEET_PER_MILE
                    )
                    
                    # Track failed routes
                    if not connectivity_success:
                        # Get all trips for this route to get direction_ids
                        route_trip_info = gtfs_model.trips[gtfs_model.trips['route_id'] == route_id][
                            ['trip_id', 'direction_id']
                        ].drop_duplicates('direction_id')
                        
                        for _, trip_row in route_trip_info.iterrows():
                            failed_connectivity_routes.append({
                                'agency_id': route_info.get('agency_id', 'N/A'),
                                'route_id': route_id,
                                'route_short_name': route_info.get('route_short_name', 'N/A'),
                                'direction_id': trip_row['direction_id'],
                                'failure_details': failure_details
                            })
                    
                    # Apply matches
                    for stop_id, node_id in stop_matches.items():
                        if stop_id in stops_gdf_proj['stop_id'].values:
                            stop_idx = stops_gdf_proj[stops_gdf_proj['stop_id'] == stop_id].index[0]
                            stops_gdf_proj.loc[stop_idx, 'model_node_id'] = node_id
                            
                            # Calculate distance
                            stop_geom = stops_gdf_proj.loc[stop_idx, 'geometry']
                            node_geom = transit_nodes_gdf_proj[
                                transit_nodes_gdf_proj['model_node_id'] == node_id
                            ].geometry.iloc[0]
                            distance = stop_geom.distance(node_geom)
                            stops_gdf_proj.loc[stop_idx, 'match_distance_ft'] = distance
                    
                    WranglerLogger.info(f"Matched {len(stop_matches)} of {len(route_stops)} stops for route {route_name}")
            
            # Log failed connectivity routes
            if failed_connectivity_routes:
                WranglerLogger.debug("\n=== Routes that failed connectivity matching ===")
                WranglerLogger.debug("These routes fell back to nearest-node matching:")
                
                # Group by route for cleaner output
                routes_by_id = {}
                for route in failed_connectivity_routes:
                    route_key = (route['agency_id'], route['route_id'], route['route_short_name'])
                    if route_key not in routes_by_id:
                        routes_by_id[route_key] = []
                    routes_by_id[route_key].append(route)
                
                for (agency_id, route_id, route_name), route_dirs in routes_by_id.items():
                    WranglerLogger.debug(f"\nAgency: {agency_id}, Route ID: {route_id}, Route Name: {route_name}")
                    
                    # Use the first direction's failure details (they should be the same for both directions)
                    failure = route_dirs[0]['failure_details']
                    if failure:
                        WranglerLogger.debug(f"  Failed at stop {failure['failed_at_stop']} of {failure['total_stops']}")
                        WranglerLogger.debug("  Route portion:")
                        
                        # Show complete route with connectivity status
                        route_to_show = failure.get('complete_route', failure.get('route_progress', []))
                        if route_to_show:
                            for idx, stop_id, node_id, was_in_connected_attempt in route_to_show:
                                stop_name = failure['stop_info'].get(stop_id, stop_id)
                                if idx < failure['failed_at_stop'] - 1:
                                    # This stop was successfully connected
                                    WranglerLogger.debug(f"    [OK] Stop {idx + 1}: {stop_name} ({stop_id}) -> Node {node_id}")
                                elif idx == failure['failed_at_stop'] - 1:
                                    # This is where connectivity failed
                                    candidates = failure['candidates_at_failure']
                                    if candidates == 0:
                                        reason = f"No nodes within {MAX_STOP_DISTANCE_MILES} mi"
                                    else:
                                        reason = f"{candidates} candidates found but none connected from node {failure['last_matched_node']}"
                                    WranglerLogger.debug(f"    [FAIL] Stop {idx + 1}: {stop_name} ({stop_id}) - {reason}")
                                else:
                                    # Subsequent stops that weren't attempted for connectivity
                                    distance_ft = failure.get('stop_distances', {}).get(stop_id, 0)
                                    distance_miles = distance_ft / FEET_PER_MILE
                                    if node_id is not None:
                                        WranglerLogger.debug(f"    [-] Stop {idx + 1}: {stop_name} ({stop_id}) -> Node {node_id} (nearest match, {distance_miles:.2f} mi)")
                                    else:
                                        WranglerLogger.debug(f"    [X] Stop {idx + 1}: {stop_name} ({stop_id}) - No match (nearest node is {distance_miles:.2f} mi away)")
                        
                        # Show directions affected
                        directions = [r['direction_id'] for r in route_dirs]
                        WranglerLogger.debug(f"  Directions affected: {', '.join(map(str, directions))}")
                
                WranglerLogger.debug(f"\nTotal: {len(failed_connectivity_routes)} route-directions failed")
                WranglerLogger.debug("===============================================\n")
    
    # Identify station stops (stops with "station" in the name, case-insensitive)
    stops_gdf_proj["is_station"] = stops_gdf_proj["stop_name"].str.lower().str.contains("station", na=False)

    # Match remaining street-level transit stops (not already matched by connectivity)
    unmatched_mask = stops_gdf_proj["model_node_id"].isna()
    street_transit_stops_unmatched = street_transit_stops[street_transit_stops.index.isin(stops_gdf_proj[unmatched_mask].index)]
    
    if len(street_transit_stops_unmatched) > 0:
        # Split street transit stops into station and non-station stops
        street_station_mask = stops_gdf_proj["has_street_transit"] & stops_gdf_proj["is_station"] & unmatched_mask
        street_non_station_mask = stops_gdf_proj["has_street_transit"] & ~stops_gdf_proj["is_station"] & unmatched_mask
        
        street_station_indices = stops_gdf_proj[street_station_mask].index
        street_non_station_indices = stops_gdf_proj[street_non_station_mask].index
        
        # Log counts
        WranglerLogger.info(
            f"Street-level transit stops: {len(street_station_indices):,} stations, {len(street_non_station_indices):,} non-stations"
        )
        
        # Match street-level stations to transit nodes
        if len(street_station_indices) > 0:
            street_station_coords = np.array(
                [(geom.x, geom.y) for geom in stops_gdf_proj.loc[street_station_indices].geometry]
            )
            WranglerLogger.debug(
                f"Matching {len(street_station_coords):,} street-level station stops to {len(transit_nodes_gdf):,} transit-accessible nodes"
            )
            distances, indices = transit_nodes_tree.query(street_station_coords, k=1)
            
            for i, stop_idx in enumerate(street_station_indices):
                if distances[i][0] <= MAX_STOP_DISTANCE_MILES * FEET_PER_MILE:
                    stops_gdf_proj.loc[stop_idx, "model_node_id"] = transit_nodes_gdf.iloc[indices[i][0]][
                        "model_node_id"
                    ]
                    stops_gdf_proj.loc[stop_idx, "match_distance_ft"] = distances[i][0]
                else:
                    # Get agency and route info for this stop
                    agency_ids = stops_gdf_proj.loc[stop_idx, 'agency_ids'] if isinstance(stops_gdf_proj.loc[stop_idx, 'agency_ids'], list) else []
                    agency_names = stops_gdf_proj.loc[stop_idx, 'agency_names'] if isinstance(stops_gdf_proj.loc[stop_idx, 'agency_names'], list) else []
                    route_names = stops_gdf_proj.loc[stop_idx, 'route_names'] if isinstance(stops_gdf_proj.loc[stop_idx, 'route_names'], list) else []
                    
                    # Format agency and route info for logging
                    if agency_names:
                        # Use agency names with IDs in parentheses
                        agency_list = []
                        for j, agency_id in enumerate(agency_ids):
                            if j < len(agency_names) and agency_names[j]:
                                agency_list.append(f"{agency_names[j]} ({agency_id})")
                            else:
                                agency_list.append(str(agency_id))
                        agency_info = f"Agencies: {', '.join(agency_list)}"
                    elif agency_ids:
                        agency_info = f"Agencies: {', '.join(map(str, agency_ids))}"
                    else:
                        agency_info = "No agency info"
                    
                    route_info = f"Routes: {', '.join(map(str, route_names[:5]))}" + ("..." if len(route_names) > 5 else "") if route_names else ""
                    
                    distance_ft = distances[i][0]
                    distance_miles = distance_ft / FEET_PER_MILE
                    WranglerLogger.warning(
                        f"Stop {stops_gdf_proj.loc[stop_idx, 'stop_id']} ({stops_gdf_proj.loc[stop_idx, 'stop_name']}) "
                        f"is {distance_miles:.2f} mi ({distance_ft:.1f} ft) from nearest transit node - exceeds {MAX_STOP_DISTANCE_MILES} mi threshold. "
                        f"{agency_info}. {route_info}"
                    )
            
            if len(distances) > 0:
                station_avg_dist = np.mean(distances)
                station_max_dist = np.max(distances)
                WranglerLogger.debug(
                    f"  Street-level stations - Average distance: {station_avg_dist:.1f} ft, Max distance: {station_max_dist:.1f} ft"
                )
        
        # Match street-level non-station stops to drive nodes
        if len(street_non_station_indices) > 0:
            street_non_station_coords = np.array(
                [(geom.x, geom.y) for geom in stops_gdf_proj.loc[street_non_station_indices].geometry]
            )
            WranglerLogger.debug(
                f"Matching {len(street_non_station_coords):,} street-level non-station stops to {len(drive_nodes_gdf):,} drive-accessible nodes"
            )
            distances, indices = drive_nodes_tree.query(street_non_station_coords, k=1)
            
            for i, stop_idx in enumerate(street_non_station_indices):
                if distances[i][0] <= MAX_STOP_DISTANCE_MILES * FEET_PER_MILE:
                    stops_gdf_proj.loc[stop_idx, "model_node_id"] = drive_nodes_gdf.iloc[indices[i][0]][
                        "model_node_id"
                    ]
                    stops_gdf_proj.loc[stop_idx, "match_distance_ft"] = distances[i][0]
                else:
                    # Get agency and route info for this stop
                    agency_ids = stops_gdf_proj.loc[stop_idx, 'agency_ids'] if isinstance(stops_gdf_proj.loc[stop_idx, 'agency_ids'], list) else []
                    agency_names = stops_gdf_proj.loc[stop_idx, 'agency_names'] if isinstance(stops_gdf_proj.loc[stop_idx, 'agency_names'], list) else []
                    route_names = stops_gdf_proj.loc[stop_idx, 'route_names'] if isinstance(stops_gdf_proj.loc[stop_idx, 'route_names'], list) else []
                    
                    # Format agency and route info for logging
                    if agency_names:
                        # Use agency names with IDs in parentheses
                        agency_list = []
                        for j, agency_id in enumerate(agency_ids):
                            if j < len(agency_names) and agency_names[j]:
                                agency_list.append(f"{agency_names[j]} ({agency_id})")
                            else:
                                agency_list.append(str(agency_id))
                        agency_info = f"Agencies: {', '.join(agency_list)}"
                    elif agency_ids:
                        agency_info = f"Agencies: {', '.join(map(str, agency_ids))}"
                    else:
                        agency_info = "No agency info"
                    
                    route_info = f"Routes: {', '.join(map(str, route_names[:5]))}" + ("..." if len(route_names) > 5 else "") if route_names else ""
                    
                    distance_ft = distances[i][0]
                    distance_miles = distance_ft / FEET_PER_MILE
                    WranglerLogger.warning(
                        f"Stop {stops_gdf_proj.loc[stop_idx, 'stop_id']} ({stops_gdf_proj.loc[stop_idx, 'stop_name']}) "
                        f"is {distance_miles:.2f} mi ({distance_ft:.1f} ft) from nearest drive node - exceeds {MAX_STOP_DISTANCE_MILES} mi threshold. "
                        f"{agency_info}. {route_info}"
                    )
            
            if len(distances) > 0:
                non_station_avg_dist = np.mean(distances)
                non_station_max_dist = np.max(distances)
                WranglerLogger.debug(
                    f"  Street-level non-stations - Average distance: {non_station_avg_dist:.1f} ft, Max distance: {non_station_max_dist:.1f} ft"
                )
        
        WranglerLogger.info(f"Matched {len(street_transit_stops_unmatched):,} street-level transit stops (excluding connectivity-matched)")

    # Match remaining non-street transit stops to transit-accessible nodes (not already matched by connectivity)
    non_street_transit_stops_unmatched = non_street_transit_stops[non_street_transit_stops.index.isin(stops_gdf_proj[unmatched_mask].index)]
    
    if len(non_street_transit_stops_unmatched) > 0:
        non_street_stop_indices = stops_gdf_proj[~stops_gdf_proj["has_street_transit"] & unmatched_mask].index
        non_street_stop_coords = np.array(
            [(geom.x, geom.y) for geom in stops_gdf_proj.loc[non_street_stop_indices].geometry]
        )

        WranglerLogger.debug(
            f"Matching {len(non_street_stop_coords)} non-street transit stops to {len(transit_nodes_gdf)} transit-accessible nodes"
        )
        distances, indices = transit_nodes_tree.query(non_street_stop_coords, k=1)

        for i, stop_idx in enumerate(non_street_stop_indices):
            if distances[i][0] <= MAX_STOP_DISTANCE_MILES * FEET_PER_MILE:
                stops_gdf_proj.loc[stop_idx, "model_node_id"] = transit_nodes_gdf.iloc[indices[i][0]][
                    "model_node_id"
                ]
                stops_gdf_proj.loc[stop_idx, "match_distance_ft"] = distances[i][0]
            else:
                distance_ft = distances[i][0]
                distance_miles = distance_ft / FEET_PER_MILE
                WranglerLogger.warning(
                    f"Stop {stops_gdf_proj.loc[stop_idx, 'stop_id']} ({stops_gdf_proj.loc[stop_idx, 'stop_name']}) "
                    f"is {distance_miles:.2f} mi ({distance_ft:.1f} ft) from nearest transit node - exceeds {MAX_STOP_DISTANCE_MILES} mi threshold"
                )

        non_street_avg_dist = np.mean(distances)
        non_street_max_dist = np.max(distances)
        WranglerLogger.info(f"Matched {len(non_street_transit_stops):,} non-street transit stops to transit-accessible nodes")
        WranglerLogger.debug(
            f"  Non-street transit stops - Average distance: {non_street_avg_dist:.1f} ft, Max distance: {non_street_max_dist:.1f} ft"
        )

    # Log statistics about the matching
    avg_distance = stops_gdf_proj["match_distance_ft"].mean()
    max_distance = stops_gdf_proj["match_distance_ft"].max()
    
    # Count stops by matching method
    connectivity_matched = 0
    if transit_graph and connectivity_routes:
        # Count stops that were matched through connectivity
        connectivity_stop_ids = set()
        for route_stops in route_stop_patterns.values():
            connectivity_stop_ids.update(route_stops['stop_id'].tolist())
        connectivity_matched = len(stops_gdf_proj[stops_gdf_proj['stop_id'].isin(connectivity_stop_ids) & stops_gdf_proj['model_node_id'].notna()])
    
    total_matched = stops_gdf_proj['model_node_id'].notna().sum()
    regular_matched = total_matched - connectivity_matched
    
    WranglerLogger.info(
        f"Stop matching complete. Total: {total_matched:,} stops matched"
    )
    WranglerLogger.info(
        f"  - Connectivity-based: {connectivity_matched:,} stops"
    )
    WranglerLogger.info(
        f"  - Nearest-node: {regular_matched:,} stops"
    )
    WranglerLogger.info(
        f"  - Average distance: {avg_distance:.1f} ft, Max distance: {max_distance:.1f} ft"
    )

    # Warn about stops that are far from nodes (more than 1000 feet)
    far_stops = stops_gdf_proj[stops_gdf_proj["match_distance_ft"] > 1000]
    if len(far_stops) > 0:
        WranglerLogger.warning(f"{len(far_stops)} stops are more than 1000 ft from nearest node")
        far_street_non_station = far_stops[far_stops["has_street_transit"] & ~far_stops["is_station"]]
        far_street_station = far_stops[far_stops["has_street_transit"] & far_stops["is_station"]]
        far_transit_stops = far_stops[~far_stops["has_street_transit"]]
        
        if len(far_street_non_station) > 0:
            WranglerLogger.warning(
                f"  - {len(far_street_non_station)} are street-level non-station stops far from drive-accessible nodes"
            )
        if len(far_street_station) > 0:
            WranglerLogger.warning(
                f"  - {len(far_street_station)} are street-level station stops far from transit-accessible nodes"
            )
        if len(far_transit_stops) > 0:
            WranglerLogger.warning(
                f"  - {len(far_transit_stops)} are non-street transit stops far from transit-accessible nodes"
            )

    # convert gtfs_model to use those new stops
    WranglerLogger.debug(f"Before convert_stops_to_wrangler_stops(), stops_gdf_proj:\n{stops_gdf_proj}")
    feed_tables["stops"] = convert_stops_to_wrangler_stops(stops_gdf_proj)
    WranglerLogger.debug(
        f"After convert_stops_to_wrangler_stops(), feed_tables['stops']:\n{feed_tables['stops']}"
    )

    # Convert stop_times to wrangler format
    # Use the modified stops_df with model_node_id for conversion
    feed_tables["stop_times"] = convert_stop_times_to_wrangler_stop_times(
        gtfs_model.stop_times.copy(),
        stops_gdf_proj,  # Use our modified stops_gdf_proj that has model_node_id
    )
    WranglerLogger.debug(
        f"After convert_stop_times_to_wrangler_stop_times(), feed_tables['stop_times']:\n{feed_tables['stop_times']}"
    )

    # create frequencies table from GTFS stop_times (if no frequencies table is specified)
    if hasattr(gtfs_model, "frequencies") and gtfs_model.frequencies is not None:
        feed_tables["frequencies"] = gtfs_model.frequencies.copy()
    elif (
        time_periods is not None
        and hasattr(gtfs_model, "stop_times")
        and gtfs_model.stop_times is not None
    ):
        # Create frequencies table from actual stop_times data
        frequencies_df = create_frequencies_from_stop_times(
            feed_tables, time_periods, default_frequency_for_onetime_route
        )
        if not frequencies_df.empty:
            feed_tables["frequencies"] = frequencies_df

            # Analyze pattern coverage
            WranglerLogger.info("Analyzing trip coverage by stop patterns...")
            coverage_df = analyze_frequency_coverage(gtfs_model, frequencies_df)

    # route gtfs street-level transit (buses, cable cars, and light rail) along roadway network
    # Handle shapes - map shape points to all roadway nodes (not just drive-accessible)
    if hasattr(gtfs_model, "shapes") and gtfs_model.shapes is not None:
        shapes_df = gtfs_model.shapes.copy()

        if "shape_model_node_id" not in shapes_df.columns:
            WranglerLogger.info(f"Mapping {len(shapes_df)} shape points to all roadway nodes")

            # Create GeoDataFrame from shape points
            shape_geometry = [
                Point(lon, lat)
                for lon, lat in zip(shapes_df["shape_pt_lon"], shapes_df["shape_pt_lat"])
            ]
            shapes_gdf = gpd.GeoDataFrame(shapes_df, geometry=shape_geometry, crs="EPSG:4326")
            shapes_gdf_proj = shapes_gdf.to_crs("EPSG:2227")

            # Use the all_nodes_tree from stop matching if available, or create new one
            if "all_nodes_tree" not in locals():
                # Reuse all_nodes_gdf_proj if available
                if "all_nodes_gdf_proj" not in locals():
                    if "geometry" not in all_nodes_df.columns:
                        node_geometry = [
                            Point(x, y) for x, y in zip(all_nodes_df["X"], all_nodes_df["Y"])
                        ]
                        all_nodes_gdf = gpd.GeoDataFrame(
                            all_nodes_df, geometry=node_geometry, crs="EPSG:4326"
                        )
                    else:
                        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, crs="EPSG:4326")
                    all_nodes_gdf_proj = all_nodes_gdf.to_crs("EPSG:2227")

                all_node_coords = np.array(
                    [(geom.x, geom.y) for geom in all_nodes_gdf_proj.geometry]
                )
                all_nodes_tree = BallTree(all_node_coords)

            # Extract shape point coordinates
            shape_coords = np.array([(geom.x, geom.y) for geom in shapes_gdf_proj.geometry])

            # Find nearest nodes for all shape points at once
            WranglerLogger.info(f"Finding nearest nodes for {len(shapes_df)} shape points")
            shape_distances, shape_indices = all_nodes_tree.query(shape_coords, k=1)

            # Extract node IDs
            shape_node_ids = [all_nodes_gdf.iloc[idx[0]]["model_node_id"] for idx in shape_indices]

            shapes_df["shape_model_node_id"] = shape_node_ids
            WranglerLogger.info("Shape point mapping complete (using all roadway nodes)")

        feed_tables["shapes"] = shapes_df
        WranglerLogger.debug(f"feed_tables['shapes']=\n{feed_tables['shapes']}")
    else:
        # Create empty shapes table with required columns
        feed_tables["shapes"] = pd.DataFrame(
            columns=[
                "shape_id",
                "shape_pt_lat",
                "shape_pt_lon",
                "shape_pt_sequence",
                "shape_model_node_id",
            ]
        )

    # create Feed object from results of the above
    try:
        feed = Feed(**feed_tables)
        WranglerLogger.info(f"Successfully created Feed with {len(feed_tables)} tables")
        return feed
    except Exception as e:
        WranglerLogger.error(f"Error creating Feed: {e}")
        # Return a minimal Feed if there's an error
        WranglerLogger.warning("Returning empty Feed due to error")
        return Feed(
            frequencies=pd.DataFrame(
                columns=["trip_id", "start_time", "end_time", "headway_secs"]
            ),
            routes=pd.DataFrame(columns=["route_id", "route_short_name", "route_long_name"]),
            shapes=pd.DataFrame(
                columns=[
                    "shape_id",
                    "shape_pt_lat",
                    "shape_pt_lon",
                    "shape_pt_sequence",
                    "shape_model_node_id",
                ]
            ),
            stops=pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"]),
            trips=pd.DataFrame(columns=["trip_id", "route_id", "service_id"]),
            stop_times=pd.DataFrame(
                columns=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"]
            ),
        )


def filter_transit_by_boundary(
    transit_data: Union[GtfsModel, Feed],
    boundary_file: Union[str, Path, gpd.GeoDataFrame],
    remove_partial_routes: bool = False,
) -> Union[GtfsModel, Feed]:
    """Filter transit routes based on whether they have stops within a boundary.
    
    Removes routes that are entirely outside the boundary shapefile. By default,
    keeps routes that have at least one stop within the boundary. Optionally can
    remove routes that are only partially within the boundary.
    
    Args:
        transit_data: Either a GtfsModel or Feed object to filter
        boundary_file: Path to boundary shapefile or a GeoDataFrame with boundary polygon(s)
        remove_partial_routes: If True, removes routes that have ANY stops outside 
            the boundary. If False (default), only removes routes with ALL stops 
            outside the boundary.
    
    Returns:
        Filtered GtfsModel or Feed object of the same type as input
    
    Example:
        >>> # Remove routes entirely outside the Bay Area
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp"
        ... )
        >>> 
        >>> # Remove routes that extend outside the boundary at all
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp",
        ...     remove_partial_routes=True
        ... )
    """
    WranglerLogger.info("Filtering transit routes by boundary")
    
    # Load boundary if it's a file path
    if isinstance(boundary_file, (str, Path)):
        boundary_gdf = gpd.read_file(boundary_file)
    else:
        boundary_gdf = boundary_file
    
    # Ensure boundary is in a geographic CRS for spatial operations
    if boundary_gdf.crs is None:
        WranglerLogger.warning("Boundary has no CRS, assuming EPSG:4326")
        boundary_gdf = boundary_gdf.set_crs("EPSG:4326")
    
    # Get stops data
    if isinstance(transit_data, GtfsModel):
        stops_df = transit_data.stops.copy()
        routes_df = transit_data.routes.copy()
        trips_df = transit_data.trips.copy()
        stop_times_df = transit_data.stop_times.copy()
        is_gtfs = True
    else:  # Feed
        stops_df = transit_data.stops.copy()
        routes_df = transit_data.routes.copy()
        trips_df = transit_data.trips.copy()
        stop_times_df = transit_data.stop_times.copy()
        is_gtfs = False
    
    # Create GeoDataFrame from stops
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="EPSG:4326"
    )
    
    # Reproject to match boundary CRS if needed
    if stops_gdf.crs != boundary_gdf.crs:
        stops_gdf = stops_gdf.to_crs(boundary_gdf.crs)
    
    # Spatial join to find stops within boundary
    stops_in_boundary = gpd.sjoin(
        stops_gdf,
        boundary_gdf,
        how="inner",
        predicate="within"
    )
    stops_in_boundary_ids = set(stops_in_boundary.stop_id.unique())
    
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
    
    # Determine which routes to keep
    if remove_partial_routes:
        # Keep only routes with ALL stops within boundary
        route_stops['keep'] = route_stops['stop_ids'].apply(
            lambda x: x.issubset(stops_in_boundary_ids)
        )
        filter_desc = "routes with all stops within boundary"
    else:
        # Keep routes with AT LEAST ONE stop within boundary
        route_stops['keep'] = route_stops['stop_ids'].apply(
            lambda x: len(x.intersection(stops_in_boundary_ids)) > 0
        )
        filter_desc = "routes with at least one stop within boundary"
    
    routes_to_keep = set(route_stops[route_stops['keep']]['route_id'])
    routes_to_remove = set(routes_df.route_id) - routes_to_keep
    
    WranglerLogger.info(
        f"Keeping {len(routes_to_keep):,} {filter_desc} "
        f"out of {len(routes_df):,} total routes"
    )
    
    if routes_to_remove:
        WranglerLogger.info(f"Removing {len(routes_to_remove):,} routes entirely outside boundary")
        WranglerLogger.debug(f"Routes being removed: {sorted(routes_to_remove)[:10]}...")
    
    # Filter data
    filtered_routes = routes_df[routes_df.route_id.isin(routes_to_keep)]
    filtered_trips = trips_df[trips_df.route_id.isin(routes_to_keep)]
    filtered_trip_ids = set(filtered_trips.trip_id)
    filtered_stop_times = stop_times_df[stop_times_df.trip_id.isin(filtered_trip_ids)]
    
    # Find stops that are still referenced
    stops_still_used = set(filtered_stop_times.stop_id.unique())
    filtered_stops = stops_df[stops_df.stop_id.isin(stops_still_used)]
    
    WranglerLogger.info(
        f"After filtering: {len(filtered_routes):,} routes, "
        f"{len(filtered_trips):,} trips, {len(filtered_stops):,} stops"
    )
    
    # Create filtered transit object
    if is_gtfs:
        # For GtfsModel, also filter shapes and other tables if they exist
        filtered_data = {
            'stops': filtered_stops,
            'routes': filtered_routes,
            'trips': filtered_trips,
            'stop_times': filtered_stop_times,
        }
        
        # Copy over other tables if they exist
        if hasattr(transit_data, 'agency') and transit_data.agency is not None:
            # Keep only agencies referenced by remaining routes
            if 'agency_id' in filtered_routes.columns:
                agency_ids = set(filtered_routes.agency_id.dropna().unique())
                filtered_data['agency'] = transit_data.agency[
                    transit_data.agency.agency_id.isin(agency_ids)
                ]
            else:
                filtered_data['agency'] = transit_data.agency
        
        if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
            # Keep only shapes referenced by remaining trips
            if 'shape_id' in filtered_trips.columns:
                shape_ids = set(filtered_trips.shape_id.dropna().unique())
                filtered_data['shapes'] = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]
            else:
                filtered_data['shapes'] = transit_data.shapes
        
        if hasattr(transit_data, 'calendar') and transit_data.calendar is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(filtered_trips.service_id.unique())
            filtered_data['calendar'] = transit_data.calendar[
                transit_data.calendar.service_id.isin(service_ids)
            ]
        
        if hasattr(transit_data, 'calendar_dates') and transit_data.calendar_dates is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(filtered_trips.service_id.unique())
            filtered_data['calendar_dates'] = transit_data.calendar_dates[
                transit_data.calendar_dates.service_id.isin(service_ids)
            ]
        
        if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            filtered_data['frequencies'] = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(filtered_trip_ids)
            ]
        
        return GtfsModel(**filtered_data)
    
    else:  # Feed
        # For Feed, also handle frequencies and shapes
        filtered_data = {
            'stops': filtered_stops,
            'routes': filtered_routes,
            'trips': filtered_trips,
            'stop_times': filtered_stop_times,
        }
        
        if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
            # Keep only shapes referenced by remaining trips
            if 'shape_id' in filtered_trips.columns:
                shape_ids = set(filtered_trips.shape_id.dropna().unique())
                filtered_data['shapes'] = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]
            else:
                filtered_data['shapes'] = transit_data.shapes
        
        if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            filtered_data['frequencies'] = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(filtered_trip_ids)
            ]
        
        return Feed(**filtered_data)


def drop_transit_agency(
    transit_data: Union[GtfsModel, Feed],
    agency_id: Union[str, List[str]],
) -> Union[GtfsModel, Feed]:
    """Remove all routes, trips, stops, etc. for a specific agency or agencies.
    
    Filters out all data associated with the specified agency_id(s), ensuring
    the resulting transit data remains valid by removing orphaned stops and
    maintaining referential integrity.
    
    Args:
        transit_data: Either a GtfsModel or Feed object to filter
        agency_id: Single agency_id string or list of agency_ids to remove
    
    Returns:
        Filtered GtfsModel or Feed object of the same type as input, with the
        specified agency and all associated data removed
    
    Example:
        >>> # Remove a single agency
        >>> filtered_gtfs = drop_transit_agency(gtfs_model, "SFMTA")
        >>> 
        >>> # Remove multiple agencies
        >>> filtered_gtfs = drop_transit_agency(
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
    
    # Get data tables
    if isinstance(transit_data, GtfsModel):
        routes_df = transit_data.routes.copy()
        trips_df = transit_data.trips.copy()
        stop_times_df = transit_data.stop_times.copy()
        stops_df = transit_data.stops.copy()
        is_gtfs = True
    else:  # Feed
        routes_df = transit_data.routes.copy()
        trips_df = transit_data.trips.copy()
        stop_times_df = transit_data.stop_times.copy()
        stops_df = transit_data.stops.copy()
        is_gtfs = False
    
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
    stops_removed = len(stops_df) - len(stops_to_keep)
    
    WranglerLogger.info(
        f"Removed: {routes_removed:,} routes, {trips_removed:,} trips, "
        f"{stop_times_removed:,} stop_times, {stops_removed:,} stops"
    )
    
    WranglerLogger.info(
        f"Remaining: {len(routes_to_keep):,} routes, {len(trips_to_keep):,} trips, "
        f"{len(stops_to_keep):,} stops"
    )
    
    # Create filtered object
    if is_gtfs:
        # For GtfsModel, also filter other tables if they exist
        filtered_data = {
            'stops': stops_to_keep,
            'routes': routes_to_keep,
            'trips': trips_to_keep,
            'stop_times': stop_times_to_keep,
        }
        
        # Handle agency table
        if hasattr(transit_data, 'agency') and transit_data.agency is not None:
            # Keep agencies that are NOT being removed
            filtered_data['agency'] = transit_data.agency[
                ~transit_data.agency.agency_id.isin(agency_ids_to_remove)
            ]
            WranglerLogger.info(
                f"Removed {len(transit_data.agency) - len(filtered_data['agency']):,} agencies"
            )
        
        # Handle shapes table
        if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
            # Keep only shapes referenced by remaining trips
            if 'shape_id' in trips_to_keep.columns:
                shape_ids = set(trips_to_keep.shape_id.dropna().unique())
                filtered_data['shapes'] = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]
                WranglerLogger.info(
                    f"Removed {len(transit_data.shapes) - len(filtered_data['shapes']):,} shape points"
                )
            else:
                filtered_data['shapes'] = transit_data.shapes
        
        # Handle calendar table
        if hasattr(transit_data, 'calendar') and transit_data.calendar is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(trips_to_keep.service_id.unique())
            filtered_data['calendar'] = transit_data.calendar[
                transit_data.calendar.service_id.isin(service_ids)
            ]
        
        # Handle calendar_dates table
        if hasattr(transit_data, 'calendar_dates') and transit_data.calendar_dates is not None:
            # Keep only service_ids referenced by remaining trips
            service_ids = set(trips_to_keep.service_id.unique())
            filtered_data['calendar_dates'] = transit_data.calendar_dates[
                transit_data.calendar_dates.service_id.isin(service_ids)
            ]
        
        # Handle frequencies table
        if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            filtered_data['frequencies'] = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(trip_ids_to_keep)
            ]
        
        return GtfsModel(**filtered_data)
    
    else:  # Feed
        # For Feed, also handle frequencies and shapes
        filtered_data = {
            'stops': stops_to_keep,
            'routes': routes_to_keep,
            'trips': trips_to_keep,
            'stop_times': stop_times_to_keep,
        }
        
        # Handle shapes table
        if hasattr(transit_data, 'shapes') and transit_data.shapes is not None:
            # Keep only shapes referenced by remaining trips
            if 'shape_id' in trips_to_keep.columns:
                shape_ids = set(trips_to_keep.shape_id.dropna().unique())
                filtered_data['shapes'] = transit_data.shapes[
                    transit_data.shapes.shape_id.isin(shape_ids)
                ]
                WranglerLogger.info(
                    f"Removed {len(transit_data.shapes) - len(filtered_data['shapes']):,} shape points"
                )
            else:
                filtered_data['shapes'] = transit_data.shapes
        
        # Handle frequencies table
        if hasattr(transit_data, 'frequencies') and transit_data.frequencies is not None:
            # Keep only frequencies for remaining trips
            filtered_data['frequencies'] = transit_data.frequencies[
                transit_data.frequencies.trip_id.isin(trip_ids_to_keep)
            ]
        
        return Feed(**filtered_data)
