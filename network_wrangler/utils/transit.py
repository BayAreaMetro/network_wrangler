"""Utilities for getting GTFS into wrangler"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from typing import Dict, Any, List, Optional
from ..logger import WranglerLogger
from ..models.gtfs.gtfs import GtfsModel
from ..transit.feed.feed import Feed
from ..roadway.network import RoadwayNetwork
from ..models.gtfs.converters import convert_stops_to_wrangler_stops, convert_stop_times_to_wrangler_stop_times

def create_feed_from_gtfs_model(
        gtfs_model: GtfsModel,
        roadway_net: RoadwayNetwork,
        time_periods: Optional[List[Dict[str, str]]] = None
    ) -> Feed:
    """Converts a GTFS feed to a Wrangler Feed object compatible with the given RoadwayNetwork.

    Args:
        gtfs_model (GtfsModel): Standard GTFS model to convert
        roadway_net (RoadwayNetwork): RoadwayNetwork to map stops to
        time_periods (Optional[List[Dict[str, str]]]): List of time period definitions for frequencies.
            Each dict should have 'start_time' and 'end_time' keys.
            Example: [{'start_time': '03:00:00', 'end_time': '06:00:00'}, ...]
            If None, frequencies table will not be created from stop_times.

    Returns:
        Feed: Wrangler Feed object with stops mapped to roadway network nodes
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")

    # Start with the tables from the GTFS model
    feed_tables = {}
    
    # Copy over standard tables that don't need modification
    # GtfsModel guarantees routes and trips exist
    feed_tables['routes'] = gtfs_model.routes.copy()
    feed_tables['trips'] = gtfs_model.trips.copy()
    
    if hasattr(gtfs_model, 'agencies') and gtfs_model.agencies is not None:
        feed_tables['agencies'] = gtfs_model.agencies.copy()

    # Get all nodes and links for spatial matching
    all_nodes_df = roadway_net.nodes_df.copy()
    links_df = roadway_net.links_df
    
    # Prepare different node sets for different route types
    drive_accessible_nodes = None
    if 'drive_access' in links_df.columns:
        # Get nodes that are connected by drive-accessible links (for buses)
        drive_links = links_df[links_df['drive_access'] == True]
        drive_accessible_node_ids = set(drive_links['A'].unique()) | set(drive_links['B'].unique())
        drive_accessible_nodes = all_nodes_df[all_nodes_df['model_node_id'].isin(drive_accessible_node_ids)].copy()
        WranglerLogger.info(f"Found {len(drive_accessible_nodes):,} drive-accessible nodes (for buses) out of {len(all_nodes_df):,} total")
    else:
        WranglerLogger.warning("No drive_access column found in links, all nodes will be used for all route types")
        drive_accessible_nodes = all_nodes_df.copy()
    
    # create mapping from gtfs_model stop to RoadwayNetwork nodes
    # GtfsModel guarantees stops exists
    stops_df = gtfs_model.stops.copy()
    
    # Determine which stops are used by buses vs other route types
    # Need to join stops -> stop_times -> trips -> routes to get route types
    WranglerLogger.info("Determining route types that serve each stop")
    
    # GtfsModel guarantees stops, stop_times, trips and routes exist
    WranglerLogger.debug(f"Processing {len(gtfs_model.stops):,} stops, {len(gtfs_model.stop_times):,} stop_times, {len(gtfs_model.trips):,} trips, {len(gtfs_model.routes):,} routes")
    
    # Join stop_times with trips and routes
    stop_trips = pd.merge(
        gtfs_model.stop_times[['stop_id', 'trip_id']].drop_duplicates(),
        gtfs_model.trips[['trip_id', 'route_id']],
        on='trip_id',
        how='left'
    )
    WranglerLogger.debug(f"After joining stop_times with trips: {len(stop_trips):,} records")
    
    stop_route_types = pd.merge(
        stop_trips,
        gtfs_model.routes[['route_id', 'route_type']],
        on='route_id',
        how='left'
    )[['stop_id', 'route_type']].drop_duplicates()
    WranglerLogger.debug(f"After joining with routes: {len(stop_route_types):,} unique stop-route_type combinations")
    WranglerLogger.debug(f"stop_route_types:\n{stop_route_types}")
    
    # Log route type distribution
    route_type_counts = stop_route_types['route_type'].value_counts()
    WranglerLogger.debug("Route type distribution in stop_route_types:")
    for rt, count in route_type_counts.items():
        WranglerLogger.debug(f"  Route type {rt}: {count:,} stop-route combinations")
    
    # Group by stop to find which route types serve each stop
    stop_route_types_agg = stop_route_types.groupby('stop_id')['route_type'].apply(list).reset_index()
    stop_route_types_agg['has_bus'] = stop_route_types_agg['route_type'].apply(lambda x: 3 in x)  # Route type 3 is bus
    
    # Merge back to stops
    stops_df = pd.merge(stops_df, stop_route_types_agg[['stop_id', 'has_bus']], on='stop_id', how='left')
    
    # Check for stops without route type info
    unmatched_stops = stops_df[stops_df['has_bus'].isna()]
    if len(unmatched_stops) > 0:
        WranglerLogger.warning(f"{len(unmatched_stops)} stops have no route type information (not found in stop_times)")
        WranglerLogger.debug(f"Example unmatched stops: {unmatched_stops['stop_id'].head().tolist()}")
    
    stops_df['has_bus'] = stops_df['has_bus'].fillna(False)
    
    bus_stops = stops_df[stops_df['has_bus']]
    non_bus_stops = stops_df[~stops_df['has_bus']]
    WranglerLogger.info(f"Found {len(bus_stops):,} stops served by buses, {len(non_bus_stops):,} stops served by other modes")
    
    # Log some examples
    if len(bus_stops) > 0:
        WranglerLogger.debug(f"Example bus stops: {bus_stops['stop_id'].head().tolist()}")
    if len(non_bus_stops) > 0:
        WranglerLogger.debug(f"Example non-bus stops: {non_bus_stops['stop_id'].head().tolist()}")
    
    # Create GeoDataFrames for spatial matching
    stop_geometry = [Point(lon, lat) for lon, lat in zip(stops_df['stop_lon'], stops_df['stop_lat'])]
    stops_gdf = gpd.GeoDataFrame(stops_df, geometry=stop_geometry, crs='EPSG:4326')
    
    # Project to local coordinate system
    stops_gdf_proj = stops_gdf.to_crs('EPSG:2227')
    
    # Prepare node GeoDataFrames
    if 'geometry' not in all_nodes_df.columns:
        node_geometry = [Point(x, y) for x, y in zip(all_nodes_df['X'], all_nodes_df['Y'])]
        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, geometry=node_geometry, crs='EPSG:4326')
    else:
        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, crs='EPSG:4326')
    
    if 'geometry' not in drive_accessible_nodes.columns:
        node_geometry = [Point(x, y) for x, y in zip(drive_accessible_nodes['X'], drive_accessible_nodes['Y'])]
        drive_nodes_gdf = gpd.GeoDataFrame(drive_accessible_nodes, geometry=node_geometry, crs='EPSG:4326')
    else:
        drive_nodes_gdf = gpd.GeoDataFrame(drive_accessible_nodes, crs='EPSG:4326')
    
    all_nodes_gdf_proj = all_nodes_gdf.to_crs('EPSG:2227')
    drive_nodes_gdf_proj = drive_nodes_gdf.to_crs('EPSG:2227')
    
    # Use spatial index for efficient nearest neighbor search
    from sklearn.neighbors import BallTree
    import numpy as np
    
    WranglerLogger.info("Building spatial indices for stop-to-node matching")
    
    # Build spatial indices
    all_node_coords = np.array([(geom.x, geom.y) for geom in all_nodes_gdf_proj.geometry])
    drive_node_coords = np.array([(geom.x, geom.y) for geom in drive_nodes_gdf_proj.geometry])
    
    all_nodes_tree = BallTree(all_node_coords)
    drive_nodes_tree = BallTree(drive_node_coords)
    
    # Initialize results
    stops_df['model_node_id'] = None
    stops_df['match_distance_ft'] = None
    
    # Match bus stops to drive-accessible nodes
    if len(bus_stops) > 0:
        bus_stop_indices = stops_df[stops_df['has_bus']].index
        bus_stop_coords = np.array([(geom.x, geom.y) for geom in stops_gdf_proj.loc[bus_stop_indices].geometry])
        
        WranglerLogger.debug(f"Matching {len(bus_stop_coords):,} bus stops to {len(drive_nodes_gdf):,} drive-accessible nodes")
        distances, indices = drive_nodes_tree.query(bus_stop_coords, k=1)
        
        for i, stop_idx in enumerate(bus_stop_indices):
            stops_df.loc[stop_idx, 'model_node_id'] = drive_nodes_gdf.iloc[indices[i][0]]['model_node_id']
            stops_df.loc[stop_idx, 'match_distance_ft'] = distances[i][0]
        
        bus_avg_dist = np.mean(distances)
        bus_max_dist = np.max(distances)
        WranglerLogger.info(f"Matched {len(bus_stops):,} bus stops to drive-accessible nodes")
        WranglerLogger.debug(f"  Bus stops - Average distance: {bus_avg_dist:.1f} ft, Max distance: {bus_max_dist:.1f} ft")
    
    # Match non-bus stops to any nodes
    if len(non_bus_stops) > 0:
        non_bus_stop_indices = stops_df[~stops_df['has_bus']].index
        non_bus_stop_coords = np.array([(geom.x, geom.y) for geom in stops_gdf_proj.loc[non_bus_stop_indices].geometry])
        
        WranglerLogger.debug(f"Matching {len(non_bus_stop_coords)} non-bus stops to {len(all_nodes_gdf)} total nodes")
        distances, indices = all_nodes_tree.query(non_bus_stop_coords, k=1)
        
        for i, stop_idx in enumerate(non_bus_stop_indices):
            stops_df.loc[stop_idx, 'model_node_id'] = all_nodes_gdf.iloc[indices[i][0]]['model_node_id']
            stops_df.loc[stop_idx, 'match_distance_ft'] = distances[i][0]
        
        non_bus_avg_dist = np.mean(distances)
        non_bus_max_dist = np.max(distances)
        WranglerLogger.info(f"Matched {len(non_bus_stops):,} non-bus stops to any roadway nodes")
        WranglerLogger.debug(f"  Non-bus stops - Average distance: {non_bus_avg_dist:.1f} ft, Max distance: {non_bus_max_dist:.1f} ft")
    
    # Log statistics about the matching
    avg_distance = stops_df['match_distance_ft'].mean()
    max_distance = stops_df['match_distance_ft'].max()
    WranglerLogger.info(f"Stop matching complete. Average distance: {avg_distance:.1f} ft, Max distance: {max_distance:.1f} ft")
    
    # Warn about stops that are far from nodes (more than 1000 feet)
    far_stops = stops_df[stops_df['match_distance_ft'] > 1000]
    if len(far_stops) > 0:
        WranglerLogger.warning(f"{len(far_stops)} stops are more than 1000 ft from nearest node")
        far_bus_stops = far_stops[far_stops['has_bus']]
        if len(far_bus_stops) > 0:
            WranglerLogger.warning(f"  - {len(far_bus_stops)} are bus stops far from drive-accessible nodes")
    
    # convert gtfs_model to use those new stops
    WranglerLogger.debug(f"Before convert_stops_to_wrangler_stops(), stops_df:\n{stops_df}")
    feed_tables['stops'] = convert_stops_to_wrangler_stops(stops_df)
    WranglerLogger.debug(f"After convert_stops_to_wrangler_stops(), feed_tables['stops']:\n{feed_tables['stops']}")
    
    # Convert stop_times to wrangler format
    # Use the modified stops_df with model_node_id for conversion
    feed_tables['stop_times'] = convert_stop_times_to_wrangler_stop_times(
        gtfs_model.stop_times.copy(), 
        stops_df  # Use our modified stops_df that has model_node_id
    )
    WranglerLogger.debug(f"After convert_stop_times_to_wrangler_stop_times(), feed_tables['stop_times']:\n{feed_tables['stop_times']}")
    
    # create frequencies table from GTFS stop_times (if no frequencies table is specified)
    if hasattr(gtfs_model, 'frequencies') and gtfs_model.frequencies is not None:
        feed_tables['frequencies'] = gtfs_model.frequencies.copy()
    elif time_periods is not None and hasattr(gtfs_model, 'stop_times') and gtfs_model.stop_times is not None:
        # Create frequencies table from actual stop_times data
        WranglerLogger.info("Creating frequencies table from GTFS stop_times data")
        
        # Convert time strings to seconds for easier calculation
        def time_to_seconds(time_obj):
            """Convert HH:MM:SS or Timestamp to seconds since midnight"""
            if isinstance(time_obj, str):
                h, m, s = map(int, time_obj.split(':'))
                return h * 3600 + m * 60 + s
            else:
                # Handle pandas Timestamp or datetime objects
                return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        
        def seconds_to_time(seconds):
            """Convert seconds since midnight to HH:MM:SS"""
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        stop_times_df = gtfs_model.stop_times.copy()
        stop_times_df['departure_seconds'] = stop_times_df['departure_time'].apply(time_to_seconds)
        
        # Get first stop for each trip to use as the reference time
        first_stops = stop_times_df.groupby('trip_id')['departure_seconds'].min().reset_index()
        first_stops.columns = ['trip_id', 'first_departure_seconds']
        
        # Join with trips to get route info
        trips_with_times = pd.merge(
            first_stops,
            gtfs_model.trips[['trip_id', 'route_id']],
            on='trip_id',
            how='left'
        )
        
        frequencies_data = []
        
        # Get stop patterns for all trips to identify distinct patterns
        stop_sequences = gtfs_model.stop_times.groupby('trip_id')['stop_id'].apply(list).reset_index()
        stop_sequences.columns = ['trip_id', 'stop_sequence']
        stop_sequences['stop_pattern'] = stop_sequences['stop_sequence'].apply(lambda x: ','.join(x))
        
        # Join with trip times
        trips_with_patterns = pd.merge(
            trips_with_times,
            stop_sequences[['trip_id', 'stop_pattern']],
            on='trip_id',
            how='left'
        )
        
        # Process each route
        for route_id, route_trips in trips_with_patterns.groupby('route_id'):
            if len(route_trips) == 0:
                WranglerLogger.debug(f"Route {route_id}: no trips found, skipping")
                continue
                
            # Find distinct stop patterns for this route
            pattern_groups = route_trips.groupby('stop_pattern')
            num_patterns = len(pattern_groups)
            
            if num_patterns > 1:
                WranglerLogger.info(f"Route {route_id}: Found {num_patterns} distinct stop patterns")
            
            # Process each pattern separately
            for pattern_idx, (stop_pattern, pattern_trips) in enumerate(pattern_groups):
                if len(pattern_trips) == 0:
                    continue
                    
                # Sort trips by departure time
                pattern_trips = pattern_trips.sort_values('first_departure_seconds')
                departures = pattern_trips['first_departure_seconds'].values
                
                # Use first trip of this pattern as template
                template_trip_id = pattern_trips.iloc[0]['trip_id']
                
                # Log pattern info
                num_stops = len(stop_pattern.split(','))
                WranglerLogger.debug(f"Route {route_id} pattern {pattern_idx + 1}/{num_patterns}: {len(pattern_trips)} trips, {num_stops} stops, template trip {template_trip_id}")
                
                # Calculate headways for each time period
                for period in time_periods:
                    start_seconds = time_to_seconds(period['start_time'])
                    end_seconds = time_to_seconds(period['end_time'])
                    
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
                        WranglerLogger.debug(f"Route {route_id} pattern {pattern_idx + 1}: no departures in period {period['start_time']}-{period['end_time']}, skipping")
                        continue
                    elif len(period_departures) == 1:
                        # For single trip, use 3-hour default headway
                        avg_headway = 10800  # 3 hours in seconds
                        WranglerLogger.debug(f"Route {route_id} pattern {pattern_idx + 1}: only 1 departure in period {period['start_time']}-{period['end_time']}, using default 3-hour headway")
                    else:
                        # Calculate average headway for multiple trips
                        headways = np.diff(np.sort(period_departures))
                        avg_headway = int(np.mean(headways))
                    
                    frequencies_data.append({
                        'trip_id': template_trip_id,
                        'start_time': period['start_time'],
                        'end_time': period['end_time'],
                        'headway_secs': avg_headway,
                        'exact_times': 0
                    })
                    
                    WranglerLogger.debug(f"Route {route_id} pattern {pattern_idx + 1} period {period['start_time']}-{period['end_time']}: {len(period_departures)} trips, avg headway {avg_headway//60:.1f} min")
        
        if frequencies_data:
            frequencies_df = pd.DataFrame(frequencies_data)
            feed_tables['frequencies'] = frequencies_df
            WranglerLogger.info(f"Created frequencies table with {len(frequencies_df)} entries from stop_times data")
            
            # Report on pattern coverage
            WranglerLogger.info("Analyzing trip coverage by stop patterns...")
            
            # Get all trips and their patterns
            all_trip_patterns = pd.merge(
                gtfs_model.trips[['trip_id', 'route_id']],
                stop_sequences[['trip_id', 'stop_pattern']],
                on='trip_id',
                how='left'
            )
            
            # Get template trips and their patterns
            template_trip_patterns = all_trip_patterns[all_trip_patterns['trip_id'].isin(frequencies_df['trip_id'].unique())]
            
            # Analyze coverage by route
            coverage_stats = []
            for route_id in all_trip_patterns['route_id'].unique():
                route_all_trips = all_trip_patterns[all_trip_patterns['route_id'] == route_id]
                route_template_trips = template_trip_patterns[template_trip_patterns['route_id'] == route_id]
                
                total_trips = len(route_all_trips)
                total_patterns = route_all_trips['stop_pattern'].nunique()
                covered_patterns = route_template_trips['stop_pattern'].nunique()
                
                # Count trips covered by template patterns
                covered_trip_count = route_all_trips[
                    route_all_trips['stop_pattern'].isin(route_template_trips['stop_pattern'])
                ]['trip_id'].nunique()
                
                coverage_pct = (covered_trip_count / total_trips * 100) if total_trips > 0 else 0
                
                coverage_stats.append({
                    'route_id': route_id,
                    'total_trips': total_trips,
                    'total_patterns': total_patterns,
                    'covered_patterns': covered_patterns,
                    'covered_trips': covered_trip_count,
                    'coverage_pct': coverage_pct
                })
                
                if covered_patterns < total_patterns:
                    WranglerLogger.warning(f"Route {route_id}: {covered_patterns}/{total_patterns} stop patterns covered by template trips ({coverage_pct:.1f}% of trips)")
            
            # Summary statistics
            coverage_df = pd.DataFrame(coverage_stats)
            avg_coverage = coverage_df['coverage_pct'].mean()
            routes_with_incomplete_coverage = len(coverage_df[coverage_df['coverage_pct'] < 100])
            
            WranglerLogger.info(f"Pattern coverage summary: {avg_coverage:.1f}% average trip coverage across all routes")
            if routes_with_incomplete_coverage > 0:
                WranglerLogger.warning(f"{routes_with_incomplete_coverage} routes have incomplete pattern coverage")
        else:
            WranglerLogger.warning("No frequency data could be calculated from stop_times")

    # route gtfs buses, cable cars and light rail along roadway network
    # Handle shapes - map shape points to all roadway nodes (not just drive-accessible)
    if hasattr(gtfs_model, 'shapes') and gtfs_model.shapes is not None:
        shapes_df = gtfs_model.shapes.copy()
        
        if 'shape_model_node_id' not in shapes_df.columns:
            WranglerLogger.info(f"Mapping {len(shapes_df)} shape points to all roadway nodes")
            
            # Create GeoDataFrame from shape points
            shape_geometry = [Point(lon, lat) for lon, lat in zip(shapes_df['shape_pt_lon'], shapes_df['shape_pt_lat'])]
            shapes_gdf = gpd.GeoDataFrame(shapes_df, geometry=shape_geometry, crs='EPSG:4326')
            shapes_gdf_proj = shapes_gdf.to_crs('EPSG:2227')
            
            # Use the all_nodes_tree from stop matching if available, or create new one
            if 'all_nodes_tree' not in locals():
                # Reuse all_nodes_gdf_proj if available
                if 'all_nodes_gdf_proj' not in locals():
                    if 'geometry' not in all_nodes_df.columns:
                        node_geometry = [Point(x, y) for x, y in zip(all_nodes_df['X'], all_nodes_df['Y'])]
                        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, geometry=node_geometry, crs='EPSG:4326')
                    else:
                        all_nodes_gdf = gpd.GeoDataFrame(all_nodes_df, crs='EPSG:4326')
                    all_nodes_gdf_proj = all_nodes_gdf.to_crs('EPSG:2227')
                
                all_node_coords = np.array([(geom.x, geom.y) for geom in all_nodes_gdf_proj.geometry])
                all_nodes_tree = BallTree(all_node_coords)
            
            # Extract shape point coordinates
            shape_coords = np.array([(geom.x, geom.y) for geom in shapes_gdf_proj.geometry])
            
            # Find nearest nodes for all shape points at once
            WranglerLogger.info(f"Finding nearest nodes for {len(shapes_df)} shape points")
            shape_distances, shape_indices = all_nodes_tree.query(shape_coords, k=1)
            
            # Extract node IDs
            shape_node_ids = [all_nodes_gdf.iloc[idx[0]]['model_node_id'] for idx in shape_indices]
            
            shapes_df['shape_model_node_id'] = shape_node_ids
            WranglerLogger.info("Shape point mapping complete (using all roadway nodes)")
        
        feed_tables['shapes'] = shapes_df
    else:
        # Create empty shapes table with required columns
        feed_tables['shapes'] = pd.DataFrame(columns=[
            'shape_id', 'shape_pt_lat', 'shape_pt_lon', 
            'shape_pt_sequence', 'shape_model_node_id'
        ])

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
            frequencies=pd.DataFrame(columns=['trip_id', 'start_time', 'end_time', 'headway_secs']),
            routes=pd.DataFrame(columns=['route_id', 'route_short_name', 'route_long_name']),
            shapes=pd.DataFrame(columns=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence', 'shape_model_node_id']),
            stops=pd.DataFrame(columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon']),
            trips=pd.DataFrame(columns=['trip_id', 'route_id', 'service_id']),
            stop_times=pd.DataFrame(columns=['trip_id', 'stop_id', 'arrival_time', 'departure_time', 'stop_sequence'])
        )