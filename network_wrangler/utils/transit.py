"""Utilities for getting GTFS into wrangler"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pathlib import Path

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
from ..roadway.network import RoadwayNetwork
from ..transit.feed.feed import Feed
from ..errors import NodeNotFoundError, TransitValidationError
from .time import time_to_seconds

# https://gtfs.org/documentation/schedule/reference/#routestxt
# GTFS route types that operate in mixed traffic so stops are nodes that are drive-accessible
# Route type 0: Tram, Streetcar, Light rail - operates in mixed traffic AND at stations
# Route type 3: Bus - operates in mixed traffic
# Route type 5: Cable tram/Cable car - street-level rail with underground cable
# Route type 11: Trolleybus - electric buses with overhead wires
MIXED_TRAFFIC_ROUTE_TYPES = [0, 3, 5, 11]

# GTFS route types that operate 
# Route type 0: Tram, Streetcar, Light rail - operates in mixed traffic AND at stations
# Route type 1: Subway, Metro
# Route type 2: Rail, intercity and long-distance
# Route type 4: Ferry
# Route type 6: Arial lift, ssupended cable car
# Route type 7: Funicular
# Route type 12: Monorail
STATION_ROUTE_TYPES = [0, 1, 2, 4, 6, 7, 12]

# GTFS route types that should use connectivity-aware matching
# Route type 1: Subway, Metro
# Route type 2: Rail
# Route type 4: Ferry
CONNECTIVITY_MATCH_ROUTE_TYPES = [1, 2, 4]

# GTFS route types that operate ONLY at stations (not in mixed traffic)
# This is STATION_ROUTE_TYPES minus MIXED_TRAFFIC_ROUTE_TYPES
STATION_ONLY_ROUTE_TYPES = list(set(STATION_ROUTE_TYPES) - set(MIXED_TRAFFIC_ROUTE_TYPES))

# GTFS route types that operate ONLY in mixed traffic
# This is MIXED_TRAFFIC_ROUTE_TYPES minus STATION_ROUTE_TYPES 
MIXED_TRAFFIC_ONLY_ROUTE_TYPES = list(set(MIXED_TRAFFIC_ROUTE_TYPES) - set(STATION_ROUTE_TYPES))

# route_types that operate in both mixed traffic and not mixed traffic
MIXED_TRAFFIC_AND_STATION_ROUTE_TYPES = list(set(MIXED_TRAFFIC_ROUTE_TYPES).intersection(STATION_ROUTE_TYPES))

# Feet to miles conversion
FEET_PER_MILE = 5280.0
# Meters to kilometers conversion
METERS_PER_KILOMETER = 1000.0

# Maximum distance in miles for a stop to match to a node
MAX_DISTANCE_STOP = {
    'feet': 0.25 * FEET_PER_MILE,
    'meters': 0.40 * METERS_PER_KILOMETER,
}
# Maximum distace for a shape node to match to a node
# This is much smaller, because shape nodes will often be mid-block and we don't need to keep all of those
MAX_DISTANCE_SHAPE = {
    'feet': 100,
    'meters': 30,
}



# WGS84 longitude/latitude coordinate system
WGS84 = "EPSG:4326"

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

    stop_times_df = feed_tables["stop_times"]
    trips_df = feed_tables["trips"]

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
        transit_links_df: DataFrame of links with transit-only links, must have columns A and B
        
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
        # Add node_a to node_a since GTFS may have double stops
        graph[node_a].add(node_a)
        # Only add edge from A to B (directed)
        graph[node_a].add(node_b)
    return graph


def match_route_stops_with_connectivity(
    route_stops_df: pd.DataFrame,
    candidate_nodes_gdf: gpd.GeoDataFrame,
    crs_units: str,
    transit_graph: Dict[int, set],
    stops_gdf_proj: gpd.GeoDataFrame,
    max_distance: float
) -> Tuple[Dict[str, int], bool]:
    """Match stops for a route considering transit link connectivity.
    
    Args:
        route_stops_df: DataFrame with stops for one route, ordered by stop_sequence
        candidate_nodes_gdf: GeoDataFrame of candidate nodes (projected coordinates)
        crs_units: 'feet' or 'meters', corresponding to CRS of candidate_nodes_gdf
        transit_graph: Adjacency graph of transit links
        stops_gdf_proj: GeoDataFrame of all stops with projected coordinates and all stop attributes
        max_distance: Maximum allowed distance from stop to node, in crs_units
        
    Returns:
        Tuple of:
        - Dict mapping stop_id to model_node_id
        - bool indicating if connectivity matching succeeded
    """
    # WranglerLogger.debug(f"==== match_route_stops_with_connectivity() ===")
    # WranglerLogger.debug(f"route_stops_df:\n{route_stops_df}")
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    stop_matches = {}
    
    # Get ordered list of stops with names
    ordered_stops_df = route_stops_df.sort_values('stop_sequence')
    ordered_stops = ordered_stops_df['stop_id'].tolist()
    
    if len(ordered_stops) == 0:
        return stop_matches, True
    
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
        nearby_mask = distances <= max_distance
        nearby_nodes = candidate_nodes_gdf[nearby_mask]['model_node_id'].tolist()
        stop_candidates[stop_id] = nearby_nodes
    
    # Try to find a connected path through the candidates
    # Start with the first stop and try each candidate
    first_stop = ordered_stops[0]
    best_match_count = 0
    best_matches = None
    
    WranglerLogger.debug(f"  First stop candidates: {stop_candidates.get(first_stop, [])}")
    for first_node in stop_candidates.get(first_stop, []):
        matches = {first_stop: first_node}
        current_node = first_node
        
        WranglerLogger.debug(f"      [OK]   [01] {stop_info[first_stop]} ({first_stop}) -> Node {first_node}")
        
        # Try to match subsequent stops
        success = True
        
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
                current_node = best_node
                WranglerLogger.debug(f"      [OK]   [{i+1:02d}] {stop_info[stop_id]} ({stop_id}) -> Node {best_node}")
            else:
                # No connected candidate found
                success = False
                if len(candidates) == 0:
                    reason = f"No nodes within {max_distance} {crs_units}"
                else:
                    reason = f"{len(candidates)} candidates ({candidates}) but none connected from node {current_node}"
                WranglerLogger.debug(f"      [FAIL] [{i+1:02d}] {stop_info[stop_id]} ({stop_id}) - {reason}")
                break
        
        if success:
            WranglerLogger.debug(f"      [SUCCESS] Matched all {len(matches)} stops using connectivity")
            return matches, True
        
        # Track best attempt (most stops matched)
        if len(matches) > best_match_count:
            best_match_count = len(matches)
            best_matches = matches
    
    # If no connected path found, fall back to nearest node matching
    WranglerLogger.debug(f"      Connectivity failed - falling back to nearest node matching")
    
    # If we had no candidates for first stop, log that
    if len(stop_candidates.get(first_stop, [])) == 0:
        WranglerLogger.debug(f"      [FAIL] [01] {stop_info[first_stop]} ({first_stop}) - No nodes within {max_distance} {crs_units}")
    
    # Use best partial match if we have one, otherwise start fresh
    if best_matches:
        stop_matches = best_matches
    
    # Complete with nearest node matching for remaining stops
    for i, stop_id in enumerate(ordered_stops):
        if stop_id in stop_matches:
            continue  # Already matched
            
        stop_geom = stops_gdf_proj.loc[stops_gdf_proj['stop_id'] == stop_id, 'geometry'].iloc[0]
        distances = candidate_nodes_gdf.geometry.distance(stop_geom)
        nearest_idx = distances.idxmin()
        nearest_node = candidate_nodes_gdf.loc[nearest_idx, 'model_node_id']
        nearest_distance = distances[nearest_idx]
        
        # Only match if within threshold
        if nearest_distance <= max_distance:
            stop_matches[stop_id] = nearest_node
            WranglerLogger.debug(f"      [NEAR] [{i+1:02d}] {stop_info[stop_id]} ({stop_id}) -> Node {nearest_node} ({nearest_distance} {crs_units}")
        else:
            WranglerLogger.debug(f"      [NONE] [{i+1:02d}] {stop_info[stop_id]} ({stop_id}) - nearest is {nearest_distance} {crs_units} away")
    
    return stop_matches, False  # Return False to indicate connectivity matching failed


def match_stops_with_connectivity_for_station_sequences(
    gtfs_model: GtfsModel,
    stops_gdf_proj: gpd.GeoDataFrame,
    crs_units: str,
    transit_nodes_gdf_proj: gpd.GeoDataFrame,
    transit_graph: Dict[int, set],
    max_distance: float
) -> int:
    """Match stops using connectivity for sequences of stations (non-mixed-traffic stops).
    
    This function processes all routes and identifies sequences of consecutive stations. 
    These sequences are matched using connectivity constraints to ensure
    stops are matched to connected nodes in the transit network graph.
    
    Args:
        gtfs_model: GTFS model containing routes, trips, and stop_times
        stops_gdf_proj: GeoDataFrame of stops with projected coordinates
        crs_units: 'feet' or 'meters', corresponding to CRS of stops_gdf_proj
        transit_nodes_gdf_proj: GeoDataFrame of transit-accessible nodes with projected coordinates
        transit_graph: Adjacency graph of transit links
        max_distance: Maximum allowed distance from stop to node, in crs_units units
        
    Returns:
        Number of stops successfully matched through connectivity
        
    Note:
        This function modifies stops_gdf_proj in place, setting model_node_id and 
        match_distance for matched stops.
    """
    WranglerLogger.info("match_stops_with_connectivity_for_station_sequences(): " + \
                        " Processing routes for connectivity-aware matching of station sequences")

    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    # Get all routes
    all_routes = gtfs_model.routes['route_id'].tolist()
    
    if not all_routes:
        return 0
        
    WranglerLogger.info(f"Analyzing {len(all_routes)} routes for station stop sequences")
    
    # Process each route to find sequences of station stops
    failed_connectivity_sequences = []
    total_sequences_processed = 0
    connectivity_matched_count = 0
    
    for route_id in all_routes:
        route_trips = gtfs_model.trips[gtfs_model.trips['route_id'] == route_id]['trip_id']
        if len(route_trips) == 0:
            continue
            
        # Get stop times for this route's trips
        route_stop_times = gtfs_model.stop_times[
            gtfs_model.stop_times['trip_id'].isin(route_trips)
        ]
        
        # Use the first trip's pattern as representative
        first_trip = route_trips.iloc[0]
        route_stops = route_stop_times[route_stop_times['trip_id'] == first_trip][
            ['stop_id', 'stop_sequence']
        ].sort_values('stop_sequence')
        
        if len(route_stops) == 0:
            continue
        
        # Find sequences of consecutive stations (non-mixed-traffic stops)
        # First, we need to know which stops are in mixed traffic
        route_stop_ids = route_stops['stop_id'].tolist()
        stop_traffic_info = stops_gdf_proj[stops_gdf_proj['stop_id'].isin(route_stop_ids)][
            ['stop_id', 'stop_in_mixed_traffic']
        ].set_index('stop_id')['stop_in_mixed_traffic'].to_dict()
        
        # Find sequences of stations (non-mixed-traffic stops)
        sequences = []
        current_sequence = []
        
        for _, stop_row in route_stops.iterrows():
            stop_id = stop_row['stop_id']
            is_mixed_traffic = stop_traffic_info.get(stop_id, True)  # Default to True if not found
            
            if not is_mixed_traffic:
                current_sequence.append(stop_row)
            else:
                # End current sequence if we have one
                if len(current_sequence) >= 2:  # Only process sequences of 2+ stops
                    sequences.append(pd.DataFrame(current_sequence))
                current_sequence = []
        
        # Don't forget the last sequence
        if len(current_sequence) >= 2:
            sequences.append(pd.DataFrame(current_sequence))
        
        # Process each sequence with connectivity matching
        for seq_idx, sequence_df in enumerate(sequences):
            total_sequences_processed += 1
            
            # Get route info
            route_info = gtfs_model.routes[gtfs_model.routes['route_id'] == route_id].iloc[0]
            route_name = route_info.get('route_short_name', route_id)
            agency_id = route_info.get('agency_id', 'N/A')
            
            WranglerLogger.debug(f"  Processing sequence {seq_idx + 1} of {len(sequences)} for route {route_name} ({route_id})")
            WranglerLogger.debug(f"  Agency: {agency_id}, Route ID: {route_id}, Route Name: {route_name}")
            WranglerLogger.debug(f"  Sequence has {len(sequence_df)} stations")
            
            # Match stops with connectivity
            stop_matches, connectivity_success = match_route_stops_with_connectivity(
                sequence_df,
                transit_nodes_gdf_proj,
                crs_units,
                transit_graph,
                stops_gdf_proj,
                max_distance,
            )
            
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
                    stops_gdf_proj.loc[stop_idx, 'match_distance'] = distance
                    
                    connectivity_matched_count += 1
            
            if not connectivity_success:
                # Track for summary statistics
                failed_connectivity_sequences.append({
                    'agency_id': agency_id,
                    'route_id': route_id,
                    'route_short_name': route_name,
                    'sequence_index': seq_idx,
                    'sequence_length': len(sequence_df)
                })
    
    WranglerLogger.info(f"Processed {total_sequences_processed} station sequences across all routes")
    
    # Log summary of failed connectivity sequences
    if failed_connectivity_sequences:
        WranglerLogger.info(f"\nSummary: {len(failed_connectivity_sequences)} station sequences failed connectivity matching")
        
        # Group by route for summary
        sequences_by_route = {}
        for seq in failed_connectivity_sequences:
            route_key = (seq['agency_id'], seq['route_id'], seq['route_short_name'])
            if route_key not in sequences_by_route:
                sequences_by_route[route_key] = []
            sequences_by_route[route_key].append(seq)
        
        WranglerLogger.debug("\nFailed connectivity sequences by route:")
        for (agency_id, route_id, route_name), route_sequences in sequences_by_route.items():
            WranglerLogger.debug(f"  {route_name} ({route_id}): {len(route_sequences)} sequences")
    else:
        WranglerLogger.info("All station sequences successfully matched using connectivity")
    
    return connectivity_matched_count


def match_stops_for_mixed_traffic(
    stops_gdf_proj: gpd.GeoDataFrame,
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    max_distance: float
) -> int:
    """Match mixed traffic stops to drive-accessible nodes in the roadway network.
    
    This function finds all unmatched stops that are in mixed traffic (e.g., bus stops)
    and matches them to the nearest drive-accessible nodes in the roadway network.
    
    Args:
        stops_gdf_proj: GeoDataFrame of stops with projected coordinates. Must contain
            columns: stop_id, stop_name, stop_in_mixed_traffic, model_node_id, 
            match_distance (in crs_units), agency_ids, agency_names, route_names
        roadway_net: RoadwayNetwork containing nodes with drive_access information
        local_crs: Coordinate reference system to use for projection, in feet. (e.g., "EPSG:2227")
        crs_units: 'feet' or 'meters', corresonding to local_crs
        max_distance: Maximum allowed distance from stop to node in crs_units
        
    Returns:
        Number of mixed traffic stops matched
        
    Notes:
        If roadway_net.nodes_df is not a GeoDataFrame, converts it to one.
        Updates stops_gdf_proj in place, setting model_node_id and match_distance (in crs_units)
        for matched stops.
    """
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    from sklearn.neighbors import BallTree
    
    # Get nodes from roadway network
    if not isinstance(roadway_net.nodes_df, gpd.GeoDataFrame):
        # Convert to GeoDataFrame if needed
        roadway_net.nodes_df = gpd.GeoDataFrame(
            roadway_net.nodes_df, 
            geometry=gpd.points_from_xy(roadway_net.nodes_df.X, roadway_net.nodes_df.Y),
            crs=WGS84
        )
    
    # Extract drive-accessible nodes
    drive_accessible_nodes_gdf = roadway_net.nodes_df[roadway_net.nodes_df["drive_access"] == True]
    WranglerLogger.debug(
        f"Found {len(drive_accessible_nodes_gdf):,} drive-accessible nodes (for mixed-traffic transit) "
        f"out of {len(roadway_net.nodes_df):,} total"
    )
    
    # Project drive nodes to specified CRS
    drive_nodes_gdf_proj = drive_accessible_nodes_gdf.to_crs(local_crs)
    
    # Find unmatched mixed traffic stops
    unmatched_mask = stops_gdf_proj["model_node_id"].isna()
    mixed_traffic_stops = stops_gdf_proj[stops_gdf_proj["stop_in_mixed_traffic"]]
    mixed_traffic_stops_unmatched = mixed_traffic_stops[mixed_traffic_stops.index.isin(stops_gdf_proj[unmatched_mask].index)]
    
    if len(mixed_traffic_stops_unmatched) == 0:
        WranglerLogger.info("No unmatched mixed traffic stops to process")
        return 0
    
    # Get indices of mixed traffic stops that need matching
    mixed_traffic_indices = stops_gdf_proj[stops_gdf_proj["stop_in_mixed_traffic"] & unmatched_mask].index
    
    WranglerLogger.info(
        f"Matching {len(mixed_traffic_indices):,} remaining mixed traffic stops to drive-accessible nodes"
    )
    
    # Build spatial index for drive nodes
    drive_node_coords = np.array([(geom.x, geom.y) for geom in drive_nodes_gdf_proj.geometry])
    drive_nodes_tree = BallTree(drive_node_coords)
    WranglerLogger.debug(f"Created BallTree for {len(drive_node_coords):,} drive nodes")
    
    # Get coordinates of stops to match
    mixed_traffic_coords = np.array(
        [(geom.x, geom.y) for geom in stops_gdf_proj.loc[mixed_traffic_indices].geometry]
    )
    
    WranglerLogger.debug(
        f"Matching {len(mixed_traffic_coords):,} mixed traffic stops to {len(drive_accessible_nodes_gdf):,} drive-accessible nodes"
    )
    
    # Query nearest neighbors
    distances, indices = drive_nodes_tree.query(mixed_traffic_coords, k=1)
    
    # Process matches
    matched_count = 0
    for i, stop_idx in enumerate(mixed_traffic_indices):
        if distances[i][0] <= max_distance:
            # Match successful
            stops_gdf_proj.loc[stop_idx, "model_node_id"] = drive_nodes_gdf_proj.iloc[indices[i][0]]["model_node_id"]
            stops_gdf_proj.loc[stop_idx, f"match_distance_{crs_units}"] = distances[i][0]
            matched_count += 1
        else:
            # Distance exceeds threshold - log warning with details
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
                f"is {distance_miles:.2f} mi ({distance_ft:.1f} ft) from nearest drive node - exceeds {MAX_DISTANCE_STOP[crs_units]} {crs_units} threshold. "
                f"{agency_info}. {route_info}"
            )
    
    # Log statistics
    if len(distances) > 0:
        avg_dist = np.mean(distances)
        max_dist = np.max(distances)
        WranglerLogger.debug(
            f"  Mixed traffic stops - Average distance: {avg_dist:.1f} ft, Max distance: {max_dist:.1f} ft"
        )
    
    WranglerLogger.info(f"Matched {matched_count:,} mixed traffic stops (excluding connectivity-matched)")
    
    return matched_count


def create_feed_shapes(
    feed_tables: Dict[str, pd.DataFrame],
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    max_distance: float
):
    """Create feed shapes table by mapping shape points to roadway nodes.
    
    Shape points that are farther than max_distance crs_units from any node are removed.
    This method updates feed_tables["shapes"] directly to make it compatible with a Feed object
    and compatible with the given RoadwayNetwork object.
    
    After mapping shape points to roadway nodes, this function validates that consecutive
    shape points correspond to actual links in the roadway network. If invalid segments
    are found (segments where no roadway link exists between consecutive shape points),
    a TransitValidationError is raised.
    
    Args:
        feed_tables: Current Feed tables, including shapes
        roadway_net: RoadwayNetwork object containing nodes to match against
        local_crs: Local coordinate reference system for projection
        crs_units: 'feet' or 'meters', corresonding to local_crs
        max_distance: Maximum distance in feet to match shape points to nodes, in crs_units
    
    Raises:
        TransitValidationError: If any shape segments don't correspond to roadway links.
            The exception contains:
            - invalid_shape_sequences_gdf: GeoDataFrame with LineString geometries of invalid segments
            - invalid_shape_ids: List of shape IDs that have invalid segments
    """
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    if 'trips' not in feed_tables:
        raise ValueError(f"'trips' table not found in feed_tables")

    if 'routes' not in feed_tables:
        raise ValueError(f"'routes' table not found in feed_tables")

    if 'agencies' not in feed_tables:
        raise ValueError(f"'agencies' table not found in feed_tables")

    WranglerLogger.debug(f"create_feed_shapes() with feed_tables['shapes'].head():\n{feed_tables['shapes'].head()}")
    WranglerLogger.debug(f"feed_tables['stops'].head():\n{feed_tables['stops'].head()}")

    from sklearn.neighbors import BallTree    
    WranglerLogger.info(f"Mapping {len(feed_tables['shapes']):,} shape points to all roadway nodes")

    # Create GeoDataFrame from shape points if not already one
    if not isinstance(feed_tables['shapes'], gpd.GeoDataFrame):
        shape_geometry = [
            shapely.geometry.Point(lon, lat) for lon, lat in \
                zip(feed_tables['shapes']["shape_pt_lon"], feed_tables['shapes']["shape_pt_lat"])
        ]
        feed_tables['shapes'] = gpd.GeoDataFrame(feed_tables['shapes'], geometry=shape_geometry, crs=WGS84)
        WranglerLogger.debug(f"Converted feed_tables['shapes'] to GeoDataFrame")
    else:
        WranglerLogger.debug(f"feed_tables['shapes'].crs={feed_tables['shapes'].crs}")

    # Add agency, route, and trip information to shapes by joining with trips and routes
    # Get unique shape_ids to trips mapping
    shape_to_trips = feed_tables['trips'][['shape_id', 'trip_id', 'route_id']].drop_duplicates()
        
    # Group by shape_id to collect all trips and routes for each shape
    shape_info = shape_to_trips.groupby('shape_id').agg({
        'trip_id': lambda x: list(x.unique()),
        'route_id': lambda x: list(x.unique())
    }).reset_index()
    shape_info.columns = ['shape_id', 'trip_ids', 'route_ids']
        
    # Get route information: Join routes with agencies to get agency names
    routes_with_agency = pd.merge(
        feed_tables['routes'][['route_id', 'agency_id', 'route_short_name', 'route_long_name', 'route_type']],
        feed_tables['agencies'][['agency_id', 'agency_name']],
        on='agency_id',
        how='left'
    )
        
    # For each shape, get the agency and route information
    def get_route_info(route_ids):
        if not route_ids:
            return [], [], [], [], []
        route_info = routes_with_agency[routes_with_agency['route_id'].isin(route_ids)]
        return (
            list(route_info['agency_id'].dropna().unique()),
            list(route_info['agency_name'].dropna().unique()) if 'agency_name' in route_info.columns else [],
            list(route_info['route_id'].unique()),
            list(route_info['route_short_name'].dropna().unique()),
            list(route_info['route_type'].dropna().unique())
        )
        
    # Apply to each shape
    shape_info[['agency_ids', 'agency_names', 'route_ids_full', 'route_names', 'route_types']] = shape_info.apply(
        lambda row: pd.Series(get_route_info(row['route_ids'])),
        axis=1
    )
        
    # Drop temporary route_ids column and rename route_ids_full to route_ids
    shape_info = shape_info.drop(columns=['route_ids']).rename(columns={'route_ids_full': 'route_ids'})
        
    # Merge with shapes based on shape_id
    feed_tables['shapes'] = pd.merge(
        feed_tables['shapes'],
        shape_info,
        on='shape_id',
        how='left'
    )
        
    # Fill NaN values with empty lists for list columns
    list_columns = ['trip_ids', 'route_ids', 'agency_ids', 'agency_names', 'route_names', 'route_types']
    for col in list_columns:
        if col in feed_tables['shapes'].columns:
            feed_tables['shapes'][col] = feed_tables['shapes'][col].apply(
                lambda x: x if isinstance(x, list) else []
            )
        
    WranglerLogger.debug(f"Added agency and route information to shapes table")
    WranglerLogger.debug(f"feed_tables['shapes'].head():\n{feed_tables['shapes'].head()}")
    
    # Project both GeoDataFrames to specified CRS for distance calculations
    feed_tables['shapes'].to_crs(local_crs, inplace=True)
    # Select only drive access for this
    drive_gdf_proj = roadway_net.nodes_df.loc[ roadway_net.nodes_df.drive_access == True].to_crs(local_crs)

    # Build BallTree from projected nodes
    drive_node_coords = np.array(
        [(geom.x, geom.y) for geom in drive_gdf_proj.geometry]
    )
    all_nodes_tree = BallTree(drive_node_coords)
    WranglerLogger.debug(f"Created BallTree for shapes with {len(drive_node_coords):,} nodes")

    # Extract shape shapely.geometry.Point coordinates
    shape_coords = np.array([(geom.x, geom.y) for geom in  feed_tables['shapes'].geometry])

    # Find nearest nodes for all shape points at once
    WranglerLogger.info(f"Finding nearest nodes for {len( feed_tables['shapes']):,} shape points within {max_distance} {crs_units}")
    shape_distances, shape_indices = all_nodes_tree.query(shape_coords, k=1)
    
    # Flatten distances array (it's 2D from query)
    shape_distances = shape_distances.flatten()
    shape_indices = shape_indices.flatten()

    # Filter out shape points that are too far from nodes
    close_mask = shape_distances <= max_distance
    kept_count = close_mask.sum()
    removed_count = len( feed_tables['shapes']) - kept_count
    
    if removed_count > 0:
        WranglerLogger.warning(
            f"Removing {removed_count:,} shape points that are >{max_distance} {crs_units} from nearest node"
        )
        # Get stats on removed points
        far_distances = shape_distances[~close_mask]
        if len(far_distances) > 0:
            WranglerLogger.debug(
                f"  Removed points: min dist={far_distances.min():.1f} {crs_units}, "
                f"max dist={far_distances.max():.1f} {crs_units}, avg={far_distances.mean():.1f} {crs_units}"
            )
    
    # Keep only close shape points
    feed_tables['shapes'] = feed_tables['shapes'][close_mask]
    
    # Extract node IDs for kept points
    shape_node_ids = [drive_gdf_proj.iloc[idx]["model_node_id"] for idx in shape_indices[close_mask]]
    feed_tables['shapes']["shape_model_node_id"] = shape_node_ids

    
    WranglerLogger.info(f"Shape shapely.geometry.Point mapping complete: kept {kept_count:,} of {kept_count + removed_count:,} points")

    # For each shape_id, when sorted by shape_pt_sequence, each consecutive pair of nodes in feed_tables['shapes']
    # needs to correspond to a link in roadway_net.links_df.

    # Validate that consecutive shape points correspond to actual links in the network using vectorized operations
    
    # Create shape_links_df:
    # Sort shapes by shape_id and sequence
    shape_links_df = feed_tables['shapes'].sort_values(['shape_id', 'shape_pt_sequence']).copy()
    
    # Create consecutive node pairs for each shape
    shape_links_df['next_node'  ] = shape_links_df.groupby('shape_id')['shape_model_node_id'].shift(-1)
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
    WranglerLogger.debug(f"Before merging shape_links_df with roadway_net.links_df: shape_links_df.head():\n{shape_links_df.head()}")

    # Left join shape segments with roadway links to find invalid segments
    shape_links_cols = shape_links_df.columns.tolist()
    shape_links_df = pd.merge(
        shape_links_df,
        roadway_net.links_df[['A','B','model_link_id']],
        on=['A', 'B'],
        how='left',
        validate="many_to_one",
        suffixes=('','_rd'),
        indicator=True
    )
    WranglerLogger.debug(f"After merging shape_links_df with roadway_net.links_df: shape_links_df.head():\n{shape_links_df.head()}")
    WranglerLogger.debug(f"shape_links_df._merge.value_counts():\n{shape_links_df._merge.value_counts()}")
    
    # Find invalid segments (where _merge is left_only)
    invalid_shape_sequences_gdf= shape_links_df[shape_links_df['_merge']=="left_only"]

    # For links that don't match, try to find a path from A to B through intermediate nodes
    
    if len(invalid_shape_sequences_gdf) > 0:
        
        # Use the RoadwayNetwork's built-in graph for pathfinding
        # transit includes drive_access and transit links
        G = roadway_net.get_modal_graph('transit')
        
        # Create node coordinates lookup for intermediate nodes
        node_coords = {}
        for _, node in roadway_net.nodes_df.iterrows():
            node_id = node['model_node_id']
            if 'geometry' in roadway_net.nodes_df.columns:
                # If geometry exists, use it
                geom = node['geometry']
                node_coords[node_id] = (geom.x, geom.y)
            elif 'X' in roadway_net.nodes_df.columns and 'Y' in roadway_net.nodes_df.columns:
                # Otherwise use X, Y coordinates
                node_coords[node_id] = (node['X'], node['Y'])
        
        # Try to find paths for invalid segments
        resolved_segments = []
        still_invalid_segments = []
        new_shape_points = []  # New points to add to feed_tables['shapes']
        
        for idx, row in invalid_shape_sequences_gdf.iterrows():
            node_a = row['A']
            node_b = row['B']
            
            try:
                # Try to find shortest path
                path = nx.shortest_path(G, node_a, node_b)
                path_length = len(path) - 1  # Number of edges
                
                # Path found - add to resolved segments
                resolved_segments.append({
                    'shape_id': row['shape_id'],
                    'shape_pt_sequence': row['shape_pt_sequence'],
                    'A': node_a,
                    'B': node_b,
                    'path': path,
                    'path_length': path_length
                })
                
                # Add intermediate nodes to shapes (skip first and last as they already exist)
                if path_length > 1:  # There are intermediate nodes
                    for i, intermediate_node in enumerate(path[1:-1], start=1):
                        if intermediate_node in node_coords:
                            lon, lat = node_coords[intermediate_node]
                            # Create new shape shapely.geometry.Point with fractional sequence number
                            new_point = {
                                'shape_id': row['shape_id'],
                                'shape_pt_sequence': row['shape_pt_sequence'] + i * (0.1 / path_length),
                                'shape_pt_lat': lat,
                                'shape_pt_lon': lon,
                                'shape_model_node_id': intermediate_node,
                                # Copy other fields from the original row if they exist
                            }
                            # Add fields that might exist in the original shape
                            for col in ['trip_ids', 'route_ids', 'agency_ids', 'agency_names', 'route_names', 'route_types']:
                                if col in row.index:
                                    new_point[col] = row[col]
                            
                            new_shape_points.append(new_point)
                
                WranglerLogger.debug(
                    f"Found path for shape {row['shape_id']} segment ({node_a} -> {node_b}): "
                    f"{path_length} hop(s) through nodes {path}"
                )
            except nx.NetworkXNoPath:
                # No path exists
                still_invalid_segments.append(idx)
                WranglerLogger.warning(
                    f"No path exists from {node_a} to {node_b} in shape {row['shape_id']}"
                )
            except nx.NodeNotFound as e:
                # One or both nodes don't exist in the network
                still_invalid_segments.append(idx)
                WranglerLogger.warning(
                    f"Node not found in network for shape {row['shape_id']}: {e}"
                )
        
        # Add new intermediate points to feed_tables['shapes']
        if new_shape_points:
            new_points_df = pd.DataFrame(new_shape_points)
            
            # If feed_tables['shapes'] is a GeoDataFrame, convert new points too
            if isinstance(feed_tables['shapes'], gpd.GeoDataFrame):
                new_points_df['geometry'] = [
                    shapely.geometry.Point(p['shape_pt_lon'], p['shape_pt_lat']) 
                    for _, p in new_points_df.iterrows()
                ]
                new_points_df = gpd.GeoDataFrame(new_points_df, crs=feed_tables['shapes'].crs)
            
            # Append new points and re-sort by shape_id and sequence
            feed_tables['shapes'] = pd.concat([feed_tables['shapes'], new_points_df], ignore_index=True)
            feed_tables['shapes'] = feed_tables['shapes'].sort_values(['shape_id', 'shape_pt_sequence'])
            
            WranglerLogger.info(
                f"Added {len(new_shape_points):,} intermediate nodes to shapes from pathfinding"
            )
        
        # Log summary of resolution attempts
        if resolved_segments:
            WranglerLogger.info(
                f"Resolved {len(resolved_segments):,} segments through pathfinding "
                f"(found indirect routes through transit network)"
            )
        
        # Update invalid_shape_sequences_gdf to only include segments that couldn't be resolved
        invalid_shape_sequences_gdf = invalid_shape_sequences_gdf.loc[still_invalid_segments]
    
    if len(invalid_shape_sequences_gdf) > 0:
        # Convert invalid segments to 2-Point LineString geodataframe
        
        # Create LineStrings from start and end points
        segment_lines = []
        for _, row in invalid_shape_sequences_gdf.iterrows():
            line = shapely.geometry.LineString([
                (row['shape_pt_lon'], row['shape_pt_lat']),
                (row['next_pt_lon'], row['next_pt_lat'])
            ])
            segment_lines.append(line)
        
        # Create GeoDataFrame with invalid segments
        shape_links_cols.remove('geometry')
        WranglerLogger.debug(f"shape_links_cols={shape_links_cols}")
        invalid_shape_sequences_gdf = gpd.GeoDataFrame(
            invalid_shape_sequences_gdf[shape_links_cols],
            geometry=segment_lines,
            crs=WGS84
        )
        WranglerLogger.debug(f"invalid_shape_sequences_gdf=\n{invalid_shape_sequences_gdf}")
        
        # Get unique shape IDs with invalid segments
        invalid_shape_ids = invalid_shape_sequences_gdf['shape_id'].unique()
        
        # Create error message
        error_msg = (
            f"Found {len(invalid_shape_ids):,} shape(s) with {len(invalid_shape_sequences_gdf):,} invalid segment(s) "
            f"that don't correspond to roadway links.\n"
            f"Invalid shapes: {list(invalid_shape_ids[:10])}"
            f"{'...' if len(invalid_shape_ids) > 10 else ''}\n"
            f"These segments need corresponding links in the roadway network."
        )
        
        # Raise exception with the invalid segments GeoDataFrame attached
        exception = TransitValidationError(error_msg)
        exception.invalid_shape_sequences_gdf = invalid_shape_sequences_gdf
        exception.invalid_shape_ids = list(invalid_shape_ids)
        raise exception

    # project back to WGS84
    # feed_tables['shapes'].set_crs(WGS84, inplace=True)
    WranglerLogger.debug(f"create_feed_shapes() complete; feed_tables['shapes']:\n{feed_tables['shapes']}")    


def create_feed_from_gtfs_model(
    gtfs_model: GtfsModel,
    roadway_net: RoadwayNetwork,
    local_crs: str,
    crs_units: str,
    time_periods: Optional[List[Dict[str, str]]] = None,
    default_frequency_for_onetime_route: int = 10800,
) -> Feed:
    """Converts a GTFS feed to a Wrangler Feed object compatible with the given RoadwayNetwork.
    
    This function modifies the roadway_net.nodes_df in place by:
    - Converting it to a GeoDataFrame if it isn't already
    - Adding transit accessibility attributes based on connected links:
        - rail_only: True if node is connected to any rail_only link
        - bus_only: True if node is connected to any bus_only link  
        - ferry_only: True if node is connected to any ferry_only link
        - drive_access: True if node is connected to any drive_access link

    Args:
        gtfs_model (GtfsModel): Standard GTFS model to convert
        roadway_net (RoadwayNetwork): RoadwayNetwork to map stops to. The nodes_df will be
            modified in place to add transit accessibility attributes.
        time_periods (Optional[List[Dict[str, str]]]): List of time period definitions for frequencies.
            Each dict should have 'start_time' and 'end_time' keys.
            Example: [{'start_time': '03:00:00', 'end_time': '06:00:00'}, ...]
            If None, frequencies table will not be created from stop_times.
        local_crs: Local coordinate reference system for projection, for calculating distances in feet or meters
        crs_units: distance measurement for local_crs, "feet" or "meters"
        default_frequency_for_onetime_route (int, optional): Default headway in seconds
            for routes with only one trip in a period. Defaults to 10800 (3 hours).

    Returns:
        Feed: Wrangler Feed object with stops mapped to roadway network nodes
    
    Raises:
        NodeNotFoundError: If any GTFS stops cannot be matched to roadway network nodes.
            This occurs when stops are too far from available nodes (beyond MAX_DISTANCE_STOP).
            The exception will have an 'unmatched_stops_gdf' attribute containing a GeoDataFrame
            of the unmatched stops with their details for debugging purposes.
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")
    if crs_units not in ["feet","meters"]:
        raise ValueError(f"crs_units must be on of 'feet' or 'meters'; received {crs_units}")

    # Start with the tables from the GTFS model
    feed_tables = {}

    # Copy over standard tables that don't need modification
    # GtfsModel guarantees routes and trips exist
    feed_tables["routes"] = gtfs_model.routes
    feed_tables["trips"] = gtfs_model.trips
    feed_tables["agencies"] = gtfs_model.agency # feed calls this agencies
    feed_tables["stops"] = gtfs_model.stops.copy() # create a copy of this which we'll manipulate

    # Get all nodes and links for spatial matching
    # Convert to GeoDataFrame if needed (modifying in place)
    if not isinstance(roadway_net.nodes_df, gpd.GeoDataFrame):
        if "geometry" not in roadway_net.nodes_df.columns:
            node_geometry = [shapely.geometry.Point(x, y) for x, y in zip(roadway_net.nodes_df["X"], roadway_net.nodes_df["Y"])]
            roadway_net.nodes_df = gpd.GeoDataFrame(roadway_net.nodes_df, geometry=node_geometry, crs=WGS84)
        else:
            roadway_net.nodes_df = gpd.GeoDataFrame(roadway_net.nodes_df, crs=WGS84)
    else:
        if roadway_net.nodes_df.crs is None:
            roadway_net.nodes_df = roadway_net.nodes_df.set_crs(WGS84)
    
    # Add transit accessibility attributes to nodes based on connected links
    # Initialize columns if they don't exist
    if "rail_only" not in roadway_net.nodes_df.columns:
        roadway_net.nodes_df["rail_only"] = False
    if "bus_only" not in roadway_net.nodes_df.columns:
        roadway_net.nodes_df["bus_only"] = False
    if "ferry_only" not in roadway_net.nodes_df.columns:
        roadway_net.nodes_df["ferry_only"] = False
    if "drive_access" not in roadway_net.nodes_df.columns:
        roadway_net.nodes_df["drive_access"] = False
    
    # Get nodes connected by each type of link
    rail_only_links = roadway_net.links_df[roadway_net.links_df["rail_only"] == True]
    rail_node_ids = set(rail_only_links["A"].unique()) | set(rail_only_links["B"].unique())
    roadway_net.nodes_df.loc[roadway_net.nodes_df["model_node_id"].isin(rail_node_ids), "rail_only"] = True
    
    bus_only_links = roadway_net.links_df[roadway_net.links_df["bus_only"] == True]
    bus_node_ids = set(bus_only_links["A"].unique()) | set(bus_only_links["B"].unique())
    roadway_net.nodes_df.loc[roadway_net.nodes_df["model_node_id"].isin(bus_node_ids), "bus_only"] = True
    
    ferry_only_links = roadway_net.links_df[roadway_net.links_df["ferry_only"] == True]
    ferry_node_ids = set(ferry_only_links["A"].unique()) | set(ferry_only_links["B"].unique())
    roadway_net.nodes_df.loc[roadway_net.nodes_df["model_node_id"].isin(ferry_node_ids), "ferry_only"] = True
    
    drive_access_links = roadway_net.links_df[roadway_net.links_df["drive_access"] == True]
    drive_node_ids = set(drive_access_links["A"].unique()) | set(drive_access_links["B"].unique())
    roadway_net.nodes_df.loc[roadway_net.nodes_df["model_node_id"].isin(drive_node_ids), "drive_access"] = True
    
    # Count transit-only nodes using the attributes we just set
    transit_only_count = len(roadway_net.nodes_df[
        (roadway_net.nodes_df["rail_only"] == True) | 
        (roadway_net.nodes_df["bus_only"] == True) | 
        (roadway_net.nodes_df["ferry_only"] == True)
    ])
    
    # Log statistics about node accessibility
    WranglerLogger.info(
        f"Node accessibility: {roadway_net.nodes_df['rail_only'].sum():,} rail_only, " + \
        f"{roadway_net.nodes_df['bus_only'].sum():,} bus_only, " + \
        f"{roadway_net.nodes_df['ferry_only'].sum():,} ferry_only, " + \
        f"{roadway_net.nodes_df['drive_access'].sum():,} drive_access"
    )
    WranglerLogger.info(
        f"Found {transit_only_count:,} transit-only nodes (rail/bus/ferry) " + \
        f"out of {len(roadway_net.nodes_df):,} total"
    )

    # create mapping from gtfs_model stop to RoadwayNetwork nodes
    # GtfsModel guarantees stops exists
    if not isinstance(feed_tables["stops"], gpd.GeoDataFrame):
        stop_geometry = [
            shapely.geometry.Point(lon, lat) for lon, lat in zip(gtfs_model.stops["stop_lon"], gtfs_model.stops["stop_lat"])
        ]
        feed_tables["stops"] = gpd.GeoDataFrame(gtfs_model.stops, geometry=stop_geometry, crs=WGS84)

    # Determine which stops are used by street-level transit vs other route types
    # Need to join stops -> stop_times -> trips -> routes to get route types
    WranglerLogger.info("Determining route types that serve each stop")

    # Join stop_times with trips and routes
    stop_trips = pd.merge(
        gtfs_model.stop_times[["stop_id", "trip_id"]].drop_duplicates(),
        gtfs_model.trips[["trip_id", "route_id"]],
        on="trip_id",
        how="left",
    )
    WranglerLogger.debug(f"After joining stop_times with trips: {len(stop_trips):,} records")

    # Create stop to route, agency mapping
    stop_agencies = pd.merge(
        stop_trips, 
        gtfs_model.routes[["route_id", "agency_id", "route_short_name", "route_type"]], 
        on="route_id", 
        how="left"
    )[["stop_id", "agency_id", "route_id","route_short_name","route_type"]].drop_duplicates()
    
    # pick up agency information; agency is a required table for gtfs_model
    stop_agencies = pd.merge(
        stop_agencies,
        gtfs_model.agency[["agency_id", "agency_name"]],
        on="agency_id",
        how="left"
    )
    WranglerLogger.debug(f"stop_agencies.head():\n{stop_agencies.head()}")
    # Group by stop to get all agencies and routes serving each stop
    stop_agency_info = stop_agencies.groupby("stop_id").agg({
        "agency_id": lambda x: list(x.dropna().unique()),
        "agency_name": lambda x: list(x.dropna().unique()) if x.notna().any() else [],
        "route_id": lambda x: list(x.dropna().unique()),
        "route_short_name": lambda x: list(x.dropna().unique()),
        "route_type": lambda x: list(x.dropna().unique()),
    }).reset_index()
    stop_agency_info.columns = ["stop_id", "agency_ids", "agency_names", "route_ids", "route_names", "route_types"]
    stop_agency_info["stop_in_mixed_traffic"] = stop_agency_info["route_types"].apply(
        lambda x: any(rt in MIXED_TRAFFIC_ONLY_ROUTE_TYPES for rt in x) if x else False
    )  # A stop is in mixed traffic if it serves MIXED_TRAFFIC_ONLY_ROUTE_TYPES

    # columns: stop_id (str), agency_ids (list of str), agency_names (list of str), 
    #   route_ids (list of str), route_names (list of str), route_types (list of int), stop_in_mixed_traffic (bool)
    WranglerLogger.debug(f"stop_agency_info.head():\n{stop_agency_info.head()}")

    # Merge this information back to stops
    feed_tables["stops"] = pd.merge(
        feed_tables["stops"], stop_agency_info, on="stop_id", how="left"
    )
    WranglerLogger.debug(f"MIXED_TRAFFIC_ONLY_ROUTE_TYPES={MIXED_TRAFFIC_ONLY_ROUTE_TYPES}")
    WranglerLogger.debug(f"MIXED_TRAFFIC_AND_STATION_ROUTE_TYPES={MIXED_TRAFFIC_AND_STATION_ROUTE_TYPES}")

    # For stops with route_types that are both MIXED_TRAFFIC_ROUTE_TYPES and STATION_ROUTE_TYPES (like light rail)
    # the stops can be a mix of stations and on-road bus stops.
    # For those stops only, set stop_in_mixed_traffic to True iff 'station' is NOT in the stop_name (case-insensitive)
    feed_tables["stops"]["name_includes_station"] = feed_tables["stops"]["stop_name"].str.lower().str.contains("station", na=False)
    
    # Check if stop serves any route types that are in MIXED_TRAFFIC_AND_STATION_ROUTE_TYPES
    feed_tables["stops"]["serves_mixed_and_station_types"] = feed_tables["stops"]["route_types"].apply(
        lambda route_types: any(rt in MIXED_TRAFFIC_AND_STATION_ROUTE_TYPES for rt in route_types) if isinstance(route_types, list) else False
    )
    
    # For stops serving mixed traffic and station route types (like light rail),
    # override stop_in_mixed_traffic based on whether "station" is NOT in the stop name
    mask = feed_tables["stops"]["serves_mixed_and_station_types"]
    feed_tables["stops"].loc[mask, "stop_in_mixed_traffic"] = ~feed_tables["stops"].loc[mask, "name_includes_station"]
    
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
            feed_tables["stops"]["is_parent"].fillna(value=False, inplace=True)
            feed_tables["stops"]["child_stop_in_mixed_traffic"].fillna(value=False, inplace=True)
            
            # Log parent stations with child stops in mixed traffic
            WranglerLogger.debug(
                f"Parent stations with mixed traffic children:\n" + \
                f"{feed_tables['stops'].loc[feed_tables['stops']['child_stop_in_mixed_traffic'] == True, ['stop_id', 'stop_name', 'child_stop_in_mixed_traffic']]}"
            )
    else:
        feed_tables["stops"]["is_parent"] = False
        feed_tables["stops"]["child_stop_in_mixed_traffic"] = False

    # for stops with stop_in_mixed_traffic.isna() that are parents, inherit child's state
    feed_tables["stops"].loc[ 
        feed_tables["stops"]["stop_in_mixed_traffic"].isna() & feed_tables["stops"]["is_parent"], 
        "stop_in_mixed_traffic"
    ] = feed_tables["stops"]["child_stop_in_mixed_traffic"]

    # Log stops without stop_in_mixed_traffic set.
    unmatched_stops = feed_tables["stops"][feed_tables["stops"]["stop_in_mixed_traffic"].isna()]
    if len(unmatched_stops) > 0:
        WranglerLogger.debug(f"unmatched_stops:\n{unmatched_stops}")
    feed_tables["stops"]["stop_in_mixed_traffic"] = feed_tables["stops"]["stop_in_mixed_traffic"].fillna(False)

    stop_in_mixed_traffic_value_counts = feed_tables["stops"]["stop_in_mixed_traffic"].value_counts()
    WranglerLogger.info(
        f"Found {stop_in_mixed_traffic_value_counts[True]:,} stops in mixed traffic, " + \
        f"{stop_in_mixed_traffic_value_counts[False]:,} stations"
    )

    # Columns from GTFS stops.txt: 
    #   stop_id, stop_name, stop_code, stop_lat, stop_lon, zone_id, stop_url, 
    #   tts_stop_name, platform_code, location_type, parent_station, stop_timezone, 
    #   wheelchair_boarding, level_id, geometry
    # added by this method: 
    #   agency_ids, agency_names, route_ids, route_names, route_types, 
    #   stop_in_mixed_traffic, name_includes_station, serves_mixed_and_station_types
    #   child_stop_in_mixed_traffic, is_parent
    # Log some examples
    if stop_in_mixed_traffic_value_counts[True] > 0:
        WranglerLogger.debug(f"Mixed-traffic transit stops:\n{feed_tables['stops'].loc[feed_tables['stops'].stop_in_mixed_traffic == True]}")
    if stop_in_mixed_traffic_value_counts[False] > 0:
        WranglerLogger.debug(f"Stations:\n{feed_tables['stops'].loc[feed_tables['stops'].stop_in_mixed_traffic == False]}")

    # Use spatial index for efficient nearest neighbor search
    from sklearn.neighbors import BallTree

    WranglerLogger.info("Building spatial indices for stop-to-node matching")

    # Project nodes to local coordinate system temporarily
    roadway_net.nodes_df.to_crs(local_crs, inplace=True)
    # Filter to transit-accessible nodes using the node attributes
    transit_nodes_gdf_proj = roadway_net.nodes_df[
        (roadway_net.nodes_df["rail_only"] == True) | 
        (roadway_net.nodes_df["bus_only"] == True) | 
        (roadway_net.nodes_df["ferry_only"] == True)
    ]

    # NOTE: this will be converted back
    # match_stops_with_connectivity_for_station_sequences() expects it to be in crs_units
    feed_tables["stops"].to_crs(local_crs, inplace=True)
    # Initialize results
    feed_tables["stops"]["model_node_id"] = None
    feed_tables["stops"]["match_distance"] = None # in crs_units
    
    # Build transit graph for connectivity-aware matching
    connectivity_matched_count = 0
    
    transit_only_links = roadway_net.links_df[ (roadway_net.links_df["rail_only"] == True) | (roadway_net.links_df["ferry_only"] == True)]
    if len(transit_only_links) > 0:
        transit_graph = build_transit_graph(transit_only_links)
        WranglerLogger.info(f"Built transit graph with {len(transit_graph)} nodes for connectivity matching")
        WranglerLogger.debug(f"transit_graph:{transit_graph}")
    
        # Handle connectivity-based matching for sequences of stations (non-mixed-traffic stops)
        # This will update stops_gdf_proj, adding model_node_id and match_distance to stops 
        # that are not in mixed traffic.
        connectivity_matched_count = match_stops_with_connectivity_for_station_sequences(
            gtfs_model,
            feed_tables["stops"],
            crs_units,
            transit_nodes_gdf_proj,
            transit_graph,
            max_distance = MAX_DISTANCE_STOP[crs_units]
        )
    

    # Match remaining mixed-traffic transit stops (not already matched by connectivity)
    mixed_traffic_matched_count = match_stops_for_mixed_traffic(
        feed_tables["stops"],
        roadway_net,
        local_crs=local_crs,
        crs_units=crs_units,
        max_distance = MAX_DISTANCE_STOP[crs_units]
    )

    # Match remaining non-street transit stops to transit-only nodes (not already matched by connectivity)
    unmatched_mask = feed_tables["stops"]["model_node_id"].isna()
    station_stops = feed_tables["stops"][~feed_tables["stops"]["stop_in_mixed_traffic"]]
    station_stops_unmatched = station_stops[station_stops.index.isin(feed_tables["stops"][unmatched_mask].index)]
    WranglerLogger.debug(f"station_stops_unmatched:\n{station_stops_unmatched}")
    if len(station_stops_unmatched) > 0:
        non_street_stop_indices = feed_tables["stops"][~feed_tables["stops"]["stop_in_mixed_traffic"] & unmatched_mask].index
        non_street_stop_coords = np.array(
            [(geom.x, geom.y) for geom in feed_tables["stops"].loc[non_street_stop_indices].geometry]
        )

        WranglerLogger.debug(
            f"Matching {len(non_street_stop_coords)} non-street transit stops to {len(transit_nodes_gdf_proj)} transit-only nodes"
        )
        # Build spatial indices for transit nodes (non-mixed traffic)
        transit_node_coords = np.array([(geom.x, geom.y) for geom in transit_nodes_gdf_proj.geometry])
    
        # Create BallTree for transit nodes with timing
        transit_nodes_tree = BallTree(transit_node_coords)
        WranglerLogger.info(f"Created BallTree for {len(transit_node_coords):,} transit nodes")
        distances, indices = transit_nodes_tree.query(non_street_stop_coords, k=1)

        for i, stop_idx in enumerate(non_street_stop_indices):
            if distances[i][0] <= MAX_DISTANCE_STOP[crs_units]:
                feed_tables["stops"].loc[stop_idx, "model_node_id"] = transit_nodes_gdf_proj.iloc[indices[i][0]][
                    "model_node_id"
                ]
                feed_tables["stops"].loc[stop_idx, "match_distance"] = distances[i][0]
            else:
                distance_ft = distances[i][0]
                distance_miles = distance_ft / FEET_PER_MILE
                WranglerLogger.warning(
                    f"Stop {feed_tables['stops'].loc[stop_idx, 'stop_id']} ({feed_tables['stops'].loc[stop_idx, 'stop_name']}) "
                    f"is {distance_miles:.2f} mi ({distance_ft:.1f} ft) from nearest transit node - exceeds {MAX_DISTANCE_STOP[crs_units]} {crs_units} threshold"
                )

        non_street_avg_dist = np.mean(distances)
        non_street_max_dist = np.max(distances)
        WranglerLogger.info(f"Matched {len(station_stops):,} non-street transit stops to transit-only nodes")
        WranglerLogger.debug(
            f"  Non-street transit stops - Average distance: {non_street_avg_dist:.1f} {crs_units}, Max distance: {non_street_max_dist:.1f} {crs_units}"
        )

    # Log statistics about the matching
    avg_distance = feed_tables["stops"]["match_distance"].mean()
    max_distance = feed_tables["stops"]["match_distance"].max()
    
    total_matched = feed_tables["stops"]['model_node_id'].notna().sum()
    regular_matched = total_matched - connectivity_matched_count
    
    WranglerLogger.info(
        f"Stop matching complete. Total: {total_matched:,} stops matched"
    )
    WranglerLogger.info(
        f"  - Connectivity-based: {connectivity_matched_count:,} stops"
    )
    WranglerLogger.info(
        f"  - Nearest-node: {regular_matched:,} stops"
    )
    WranglerLogger.info(
        f"  - Average distance: {avg_distance:.1f} ft, Max distance: {max_distance:.1f} ft"
    )

    # Warn about stops that are far from nodes (more than 1000 feet)
    far_stops = feed_tables["stops"][feed_tables["stops"]["match_distance"] > 1000]
    if len(far_stops) > 0:
        WranglerLogger.warning(f"{len(far_stops)} stops are more than 1000 ft from nearest node")
        far_mixed_traffic_stops = far_stops[far_stops["stop_in_mixed_traffic"]]
        far_transit_stops = far_stops[~far_stops["stop_in_mixed_traffic"]]
        
        if len(far_mixed_traffic_stops) > 0:
            WranglerLogger.warning(
                f"  - {len(far_mixed_traffic_stops)} are mixed traffic stops far from drive-accessible nodes"
            )
        if len(far_transit_stops) > 0:
            WranglerLogger.warning(
                f"  - {len(far_transit_stops)} are non-mixed-traffic stops far from transit-accessible nodes"
            )

    # if there are stops with no model_node_id, raise NodeNotFoundError and return stops_proj_gdf with those stops
    # in the exception
    unmatched_stops_gdf = feed_tables["stops"].loc[feed_tables["stops"]["model_node_id"].isna()]
    if len(unmatched_stops_gdf) > 0:
        
        error_msg = f"Failed to match {len(unmatched_stops_gdf)} stops to roadway network nodes. See exception.unmatched_stops_gdf"

        # Create exception with the unmatched stops dataframe attached
        exception = NodeNotFoundError(error_msg)
        exception.unmatched_stops_gdf = unmatched_stops_gdf.to_crs(WGS84)
        raise exception


    # convert gtfs_model to use those new stops
    stops_crs = feed_tables['stops'].crs
    WranglerLogger.debug(f"Before convert_stops_to_wrangler_stops(), crs={feed_tables['stops'].crs} " + \
                         f" feed_tables[stops]:\n{feed_tables['stops']}")
    gtfs_model.stops = feed_tables["stops"].copy() # copy this version back; it has stop_model_id
    feed_tables["stops"] = convert_stops_to_wrangler_stops(feed_tables["stops"])
    
    # CRS got lost; reset to what it was before
    if isinstance(feed_tables["stops"], gpd.GeoDataFrame):
        if feed_tables["stops"].crs == None:
            feed_tables["stops"].set_crs(stops_crs, inplace=True)
        else:
            feed_tables["stops"].to_crs(stops_crs, inplace=True)
    
    WranglerLogger.debug(
        f"After convert_stops_to_wrangler_stops() and CRS conversion, " + \
        f"crs={feed_tables['stops'].crs if isinstance(feed_tables['stops'], gpd.GeoDataFrame) else 'Not a GeoDataFrame'} " + \
        f" feed_tables['stops']:\n{feed_tables['stops']}"
    )
    WranglerLogger.debug(f"feed_tables['stops'].dtypes:\n{feed_tables['stops'].dtypes}")

    # Convert stop_times to wrangler format
    # Use the modified stops_df with model_node_id for conversion
    feed_tables["stop_times"] = convert_stop_times_to_wrangler_stop_times(
        gtfs_model.stop_times,
        gtfs_model.stops,
    )
    WranglerLogger.debug(
        f"After convert_stop_times_to_wrangler_stop_times(), feed_tables['stop_times']:\n{feed_tables['stop_times']}"
    )
    WranglerLogger.debug(f"feed_tables['stop_times'].dtypes:\n{feed_tables['stop_times'].dtypes}")

    # create frequencies table from GTFS stop_times (if no frequencies table is specified)
    if hasattr(gtfs_model, "frequencies") and gtfs_model.frequencies is not None:
        feed_tables["frequencies"] = gtfs_model.frequencies
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
    # Create feed shapes by mapping shape points to roadway nodes
    if hasattr(gtfs_model, "shapes") and gtfs_model.shapes is not None:
        feed_tables["shapes"] = gtfs_model.shapes.copy()
        create_feed_shapes(
            feed_tables, 
            roadway_net,  # Pass RoadwayNetwork object
            local_crs=local_crs,
            crs_units=crs_units,
            max_distance=MAX_DISTANCE_SHAPE[crs_units]
        )
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
        WranglerLogger.debug("No shapes in GTFS model, created empty shapes table")

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
    partially_include_route_type_action: Optional[Dict[int, str]] = None,
) -> None:
    """Filter transit routes based on whether they have stops within a boundary.
    
    Removes routes that are entirely outside the boundary shapefile. Routes that are
    partially within the boundary are kept by default, but can be configured per 
    route type to be truncated at the boundary. Modifies transit_data in place.
    
    Args:
        transit_data: Either a GtfsModel or Feed object to filter. Modified in place.
        boundary_file: Path to boundary shapefile or a GeoDataFrame with boundary polygon(s)
        partially_include_route_type_action: Optional dictionary mapping route_type to 
            action for routes partially within boundary:
            - "truncate": Truncate route to only include stops within boundary
            Route types not specified in this dictionary will be kept entirely (default).
    
    Example:
        >>> # Remove routes entirely outside the Bay Area
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp"
        ... )
        >>> 
        >>> # Truncate bus routes at boundary, keep other route types unchanged
        >>> filtered_gtfs = filter_transit_by_boundary(
        ...     gtfs_model,
        ...     "bay_area_boundary.shp",
        ...     partially_include_route_type_action={
        ...         2: "truncate",  # Rail - will be truncated at boundary
        ...         # Other route types not listed will be kept entirely
        ...     }
        ... )
    """
    WranglerLogger.info("Filtering transit routes by boundary")
    
    # Log input parameters
    WranglerLogger.debug(f"partially_include_route_type_action: {partially_include_route_type_action}")
    
    # Load boundary if it's a file path
    if isinstance(boundary_file, (str, Path)):
        WranglerLogger.debug(f"Loading boundary from file: {boundary_file}")
        boundary_gdf = gpd.read_file(boundary_file)
    else:
        WranglerLogger.debug("Using provided boundary GeoDataFrame")
        boundary_gdf = boundary_file
    
    WranglerLogger.debug(f"Boundary has {len(boundary_gdf)} polygon(s)")
    
    # Ensure boundary is in a geographic CRS for spatial operations
    if boundary_gdf.crs is None:
        WranglerLogger.warning("Boundary has no CRS, assuming EPSG:4326")
        boundary_gdf = boundary_gdf.set_crs(WGS84)
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
        crs=WGS84
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
