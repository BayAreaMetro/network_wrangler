"""Utilities for getting GTFS into wrangler"""

from typing import Any, Dict, List, Optional

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
) -> Dict[str, int]:
    """Match stops for a route considering transit link connectivity.
    
    Args:
        route_stops_df: DataFrame with stops for one route, ordered by stop_sequence
        candidate_nodes_gdf: GeoDataFrame of candidate nodes (projected coordinates)
        transit_graph: Adjacency graph of transit links
        stops_gdf_proj: GeoDataFrame of all stops (projected coordinates)
        max_distance_ft: Maximum allowed distance from stop to node
        
    Returns:
        Dict mapping stop_id to model_node_id
    """
    stop_matches = {}
    
    # Get ordered list of stops
    ordered_stops = route_stops_df.sort_values('stop_sequence')['stop_id'].tolist()
    
    if len(ordered_stops) == 0:
        return stop_matches
    
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
    for first_node in stop_candidates.get(first_stop, []):
        matches = {first_stop: first_node}
        current_node = first_node
        
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
            else:
                # No connected candidate found
                success = False
                break
        
        if success:
            return matches, True  # Return True to indicate connectivity matching succeeded
    
    # If no connected path found, fall back to nearest node matching
    for stop_id in ordered_stops:
        stop_geom = stops_gdf_proj.loc[stops_gdf_proj['stop_id'] == stop_id, 'geometry'].iloc[0]
        distances = candidate_nodes_gdf.geometry.distance(stop_geom)
        nearest_idx = distances.idxmin()
        stop_matches[stop_id] = candidate_nodes_gdf.loc[nearest_idx, 'model_node_id']
    
    return stop_matches, False  # Return False to indicate connectivity matching failed


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
    stops_df["model_node_id"] = None
    stops_df["match_distance_ft"] = None
    
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
                    stop_matches, connectivity_success = match_route_stops_with_connectivity(
                        route_stops,
                        transit_nodes_gdf_proj,
                        transit_graph,
                        stops_gdf_proj
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
                                'direction_id': trip_row['direction_id']
                            })
                    
                    # Apply matches
                    for stop_id, node_id in stop_matches.items():
                        if stop_id in stops_df['stop_id'].values:
                            stop_idx = stops_df[stops_df['stop_id'] == stop_id].index[0]
                            stops_df.loc[stop_idx, 'model_node_id'] = node_id
                            
                            # Calculate distance
                            stop_geom = stops_gdf_proj.loc[stop_idx, 'geometry']
                            node_geom = transit_nodes_gdf_proj[
                                transit_nodes_gdf_proj['model_node_id'] == node_id
                            ].geometry.iloc[0]
                            distance = stop_geom.distance(node_geom)
                            stops_df.loc[stop_idx, 'match_distance_ft'] = distance
                    
                    WranglerLogger.info(f"Matched {len(stop_matches)} stops for route {route_name}")
            
            # Log failed connectivity routes
            if failed_connectivity_routes:
                WranglerLogger.debug("\n=== Routes that failed connectivity matching ===")
                WranglerLogger.debug("These routes fell back to nearest-node matching:")
                for route in failed_connectivity_routes:
                    WranglerLogger.debug(
                        f"  Agency: {route['agency_id']}, "
                        f"Route ID: {route['route_id']}, "
                        f"Route Name: {route['route_short_name']}, "
                        f"Direction: {route['direction_id']}"
                    )
                WranglerLogger.debug(f"Total: {len(failed_connectivity_routes)} route-directions failed")
                WranglerLogger.debug("===============================================\n")
    
    # Identify station stops (stops with "station" in the name, case-insensitive)
    stops_df["is_station"] = stops_df["stop_name"].str.lower().str.contains("station", na=False)

    # Match remaining street-level transit stops (not already matched by connectivity)
    unmatched_mask = stops_df["model_node_id"].isna()
    street_transit_stops_unmatched = street_transit_stops[street_transit_stops.index.isin(stops_df[unmatched_mask].index)]
    
    if len(street_transit_stops_unmatched) > 0:
        # Split street transit stops into station and non-station stops
        street_station_mask = stops_df["has_street_transit"] & stops_df["is_station"] & unmatched_mask
        street_non_station_mask = stops_df["has_street_transit"] & ~stops_df["is_station"] & unmatched_mask
        
        street_station_indices = stops_df[street_station_mask].index
        street_non_station_indices = stops_df[street_non_station_mask].index
        
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
                stops_df.loc[stop_idx, "model_node_id"] = transit_nodes_gdf.iloc[indices[i][0]][
                    "model_node_id"
                ]
                stops_df.loc[stop_idx, "match_distance_ft"] = distances[i][0]
            
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
                stops_df.loc[stop_idx, "model_node_id"] = drive_nodes_gdf.iloc[indices[i][0]][
                    "model_node_id"
                ]
                stops_df.loc[stop_idx, "match_distance_ft"] = distances[i][0]
            
            if len(distances) > 0:
                non_station_avg_dist = np.mean(distances)
                non_station_max_dist = np.max(distances)
                WranglerLogger.debug(
                    f"  Street-level non-stations - Average distance: {non_station_avg_dist:.1f} ft, Max distance: {non_station_max_dist:.1f} ft"
                )
        
        WranglerLogger.info(f"Matched {len(street_transit_stops_unmatched):,} street-level transit stops (excluding connectivity-matched)")

    # Match remaining non-street transit stops to transit-accessible nodes (not already matched by connectivity)
    non_street_transit_stops_unmatched = non_street_transit_stops[non_street_transit_stops.index.isin(stops_df[unmatched_mask].index)]
    
    if len(non_street_transit_stops_unmatched) > 0:
        non_street_stop_indices = stops_df[~stops_df["has_street_transit"] & unmatched_mask].index
        non_street_stop_coords = np.array(
            [(geom.x, geom.y) for geom in stops_gdf_proj.loc[non_street_stop_indices].geometry]
        )

        WranglerLogger.debug(
            f"Matching {len(non_street_stop_coords)} non-street transit stops to {len(transit_nodes_gdf)} transit-accessible nodes"
        )
        distances, indices = transit_nodes_tree.query(non_street_stop_coords, k=1)

        for i, stop_idx in enumerate(non_street_stop_indices):
            stops_df.loc[stop_idx, "model_node_id"] = transit_nodes_gdf.iloc[indices[i][0]][
                "model_node_id"
            ]
            stops_df.loc[stop_idx, "match_distance_ft"] = distances[i][0]

        non_street_avg_dist = np.mean(distances)
        non_street_max_dist = np.max(distances)
        WranglerLogger.info(f"Matched {len(non_street_transit_stops):,} non-street transit stops to transit-accessible nodes")
        WranglerLogger.debug(
            f"  Non-street transit stops - Average distance: {non_street_avg_dist:.1f} ft, Max distance: {non_street_max_dist:.1f} ft"
        )

    # Log statistics about the matching
    avg_distance = stops_df["match_distance_ft"].mean()
    max_distance = stops_df["match_distance_ft"].max()
    
    # Count stops by matching method
    connectivity_matched = 0
    if transit_graph and connectivity_routes:
        # Count stops that were matched through connectivity
        connectivity_stop_ids = set()
        for route_stops in route_stop_patterns.values():
            connectivity_stop_ids.update(route_stops['stop_id'].tolist())
        connectivity_matched = len(stops_df[stops_df['stop_id'].isin(connectivity_stop_ids) & stops_df['model_node_id'].notna()])
    
    total_matched = stops_df['model_node_id'].notna().sum()
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
    far_stops = stops_df[stops_df["match_distance_ft"] > 1000]
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
    WranglerLogger.debug(f"Before convert_stops_to_wrangler_stops(), stops_df:\n{stops_df}")
    feed_tables["stops"] = convert_stops_to_wrangler_stops(stops_df)
    WranglerLogger.debug(
        f"After convert_stops_to_wrangler_stops(), feed_tables['stops']:\n{feed_tables['stops']}"
    )

    # Convert stop_times to wrangler format
    # Use the modified stops_df with model_node_id for conversion
    feed_tables["stop_times"] = convert_stop_times_to_wrangler_stop_times(
        gtfs_model.stop_times.copy(),
        stops_df,  # Use our modified stops_df that has model_node_id
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
