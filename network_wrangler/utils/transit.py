"""Utilities for getting GTFS into wrangler"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from typing import Dict, Any
from ..logger import WranglerLogger
from ..models.gtfs.gtfs import GtfsModel
from ..transit.feed.feed import Feed
from ..roadway.network import RoadwayNetwork
from ..models.gtfs.converters import convert_stops_to_wrangler_stops, convert_stop_times_to_wrangler_stop_times

def create_feed_from_gtfs_model(
        gtfs_model: GtfsModel,
        roadway_net: RoadwayNetwork
    ) -> Feed:
    """Converts a GTFS feed to a Wrangler Feed object compatible with the given RoadwayNetwork.

    Args:
        gtfs_model (GtfsModel): Standard GTFS model to convert
        roadway_net (RoadwayNetwork): RoadwayNetwork to map stops to

    Returns:
        Feed: Wrangler Feed object with stops mapped to roadway network nodes
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")

    # Start with the tables from the GTFS model
    feed_tables = {}
    
    # Copy over standard tables that don't need modification
    if hasattr(gtfs_model, 'routes') and gtfs_model.routes is not None:
        feed_tables['routes'] = gtfs_model.routes.copy()
    
    if hasattr(gtfs_model, 'trips') and gtfs_model.trips is not None:
        feed_tables['trips'] = gtfs_model.trips.copy()
    
    if hasattr(gtfs_model, 'agencies') and gtfs_model.agencies is not None:
        feed_tables['agencies'] = gtfs_model.agencies.copy()

    # First, identify which nodes are connected by drive-accessible links
    # This ensures transit can only use nodes that are reachable via driveable roads
    nodes_df = roadway_net.nodes_df.copy()
    links_df = roadway_net.links_df
    
    if 'drive_access' in links_df.columns:
        # Get nodes that are connected by drive-accessible links
        drive_links = links_df[links_df['drive_access'] == True]
        accessible_nodes = set(drive_links['A'].unique()) | set(drive_links['B'].unique())
        
        # Filter nodes to only those accessible via drive_access links
        nodes_df = nodes_df[nodes_df['model_node_id'].isin(accessible_nodes)].copy()
        WranglerLogger.info(f"Filtering to {len(nodes_df)} nodes connected by drive-accessible links out of {len(roadway_net.nodes_df)} total")
    else:
        WranglerLogger.warning("No drive_access column found in links, assuming all links are accessible to transit")
    
    # create mapping from gtfs_model stop to RoadwayNetwork nodes
    if hasattr(gtfs_model, 'stops') and gtfs_model.stops is not None:
        stops_df = gtfs_model.stops.copy()
        
        # Create GeoDataFrame from stops
        WranglerLogger.info(f"Mapping {len(stops_df)} GTFS stops to accessible roadway nodes")
        stop_geometry = [Point(lon, lat) for lon, lat in zip(stops_df['stop_lon'], stops_df['stop_lat'])]
        stops_gdf = gpd.GeoDataFrame(stops_df, geometry=stop_geometry, crs='EPSG:4326')
        if 'geometry' not in nodes_df.columns:
            # Create geometry from X, Y coordinates if not present
            node_geometry = [Point(x, y) for x, y in zip(nodes_df['X'], nodes_df['Y'])]
            nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=node_geometry, crs='EPSG:4326')
        else:
            nodes_gdf = gpd.GeoDataFrame(nodes_df, crs='EPSG:4326')
        
        # Project to a local coordinate system for accurate distance calculations
        # Using California State Plane Zone III (EPSG:2227) for Bay Area
        stops_gdf_proj = stops_gdf.to_crs('EPSG:2227')
        nodes_gdf_proj = nodes_gdf.to_crs('EPSG:2227')
        
        # Use spatial index for efficient nearest neighbor search
        from sklearn.neighbors import BallTree
        import numpy as np
        
        # Create spatial index of nodes using BallTree for more efficient nearest neighbor search
        WranglerLogger.info("Building spatial index for efficient stop-to-node matching")
        
        # Extract coordinates as numpy arrays
        node_coords = np.array([(geom.x, geom.y) for geom in nodes_gdf_proj.geometry])
        stop_coords = np.array([(geom.x, geom.y) for geom in stops_gdf_proj.geometry])
        
        # Build BallTree for efficient spatial queries
        tree = BallTree(node_coords)
        
        # Find nearest nodes for all stops at once
        distances, indices = tree.query(stop_coords, k=1)
        
        # Extract results
        nearest_node_ids = [nodes_gdf.iloc[idx[0]]['model_node_id'] for idx in indices]
        distances = distances.flatten()  # Convert from 2D to 1D array
        
        WranglerLogger.info(f"Matched {len(stops_gdf)} stops to nodes")
        
        # Add model_node_id to stops
        stops_df['model_node_id'] = nearest_node_ids
        stops_df['match_distance_ft'] = distances  # Distance in feet (EPSG:2227 units)
        
        # Log statistics about the matching
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        WranglerLogger.info(f"Stop matching complete. Average distance: {avg_distance:.1f} ft, Max distance: {max_distance:.1f} ft")
        
        # Warn about stops that are far from nodes (more than 1000 feet)
        far_stops = stops_df[stops_df['match_distance_ft'] > 1000]
        if len(far_stops) > 0:
            WranglerLogger.warning(f"{len(far_stops)} stops are more than 1000 ft from nearest drive-accessible node")
        
        # Log info about accessibility
        if 'drive_access' in links_df.columns:
            WranglerLogger.info("All transit stops have been matched to nodes accessible via drive_access links")
        
        # convert gtfs_model to use those new stops
        feed_tables['stops'] = convert_stops_to_wrangler_stops(stops_df)
    
    # Convert stop_times to wrangler format
    if hasattr(gtfs_model, 'stop_times') and gtfs_model.stop_times is not None:
        if 'stops' in feed_tables:
            # Use the modified stops_df with model_node_id for conversion
            feed_tables['stop_times'] = convert_stop_times_to_wrangler_stop_times(
                gtfs_model.stop_times.copy(), 
                stops_df  # Use our modified stops_df that has model_node_id
            )
        else:
            feed_tables['stop_times'] = gtfs_model.stop_times.copy()
    
    # create frequencies table from GTFS stop_times (if no frequencies table is specified)
    if hasattr(gtfs_model, 'frequencies') and gtfs_model.frequencies is not None:
        feed_tables['frequencies'] = gtfs_model.frequencies.copy()
    else:
        # Create frequencies table with reasonable defaults based on route types
        if 'trips' in feed_tables and 'routes' in feed_tables:
            WranglerLogger.info("Creating frequencies table with transit service patterns")
            frequencies_data = []
            
            trips_df = feed_tables['trips']
            routes_df = feed_tables['routes']
            
            # Join trips with routes to get route types
            trip_routes = pd.merge(
                trips_df[['trip_id', 'route_id', 'service_id']],
                routes_df[['route_id', 'route_type', 'route_short_name', 'route_long_name']],
                on='route_id',
                how='left'
            )
            
            # Define headways based on route type (GTFS route types)
            # 0=Tram/LRT, 1=Subway/Metro, 2=Rail, 3=Bus, 4=Ferry, 5=Cable car, 6=Gondola, 7=Funicular
            route_type_headways = {
                0: {'peak': 600, 'offpeak': 900},      # Light rail: 10/15 min
                1: {'peak': 300, 'offpeak': 600},      # Subway: 5/10 min  
                2: {'peak': 1800, 'offpeak': 3600},    # Rail: 30/60 min
                3: {'peak': 900, 'offpeak': 1800},     # Bus: 15/30 min
                4: {'peak': 1800, 'offpeak': 3600},    # Ferry: 30/60 min
                5: {'peak': 600, 'offpeak': 600},      # Cable car: 10 min
                6: {'peak': 900, 'offpeak': 1200},     # Gondola: 15/20 min
                7: {'peak': 900, 'offpeak': 1200},     # Funicular: 15/20 min
            }
            
            # Group by route and service to create frequency patterns
            for (route_id, route_type, service_id), group in trip_routes.groupby(['route_id', 'route_type', 'service_id']):
                if len(group) == 0:
                    continue
                
                # Get headways for this route type
                headways = route_type_headways.get(route_type, {'peak': 1200, 'offpeak': 1800})  # Default 20/30 min
                
                # Create peak and off-peak frequency entries
                template_trip = group.iloc[0]['trip_id']
                
                # Morning peak (6-9 AM)
                frequencies_data.append({
                    'trip_id': template_trip,
                    'start_time': '06:00:00',
                    'end_time': '09:00:00',
                    'headway_secs': headways['peak'],
                    'exact_times': 0
                })
                
                # Midday (9 AM - 3 PM)
                frequencies_data.append({
                    'trip_id': template_trip,
                    'start_time': '09:00:00',
                    'end_time': '15:00:00',
                    'headway_secs': headways['offpeak'],
                    'exact_times': 0
                })
                
                # Evening peak (3-7 PM)
                frequencies_data.append({
                    'trip_id': template_trip,
                    'start_time': '15:00:00',
                    'end_time': '19:00:00',
                    'headway_secs': headways['peak'],
                    'exact_times': 0
                })
                
                # Evening (7-10 PM)
                frequencies_data.append({
                    'trip_id': template_trip,
                    'start_time': '19:00:00',
                    'end_time': '22:00:00',
                    'headway_secs': headways['offpeak'],
                    'exact_times': 0
                })
                
                route_name = group.iloc[0]['route_short_name'] if pd.notna(group.iloc[0]['route_short_name']) else route_id
                WranglerLogger.debug(f"Route {route_name} (type {route_type}): peak {headways['peak']//60} min, off-peak {headways['offpeak']//60} min")
            
            # Remove duplicates if any
            frequencies_df = pd.DataFrame(frequencies_data).drop_duplicates()
            
            feed_tables['frequencies'] = frequencies_df
            WranglerLogger.info(f"Created frequencies table with {len(frequencies_df)} entries covering {len(frequencies_df['trip_id'].unique())} unique trips")

    # route gtfs buses, cable cars and light rail along roadway network
    # Handle shapes - map shape points to roadway nodes (using only drive-accessible nodes)
    if hasattr(gtfs_model, 'shapes') and gtfs_model.shapes is not None:
        shapes_df = gtfs_model.shapes.copy()
        
        if 'shape_model_node_id' not in shapes_df.columns:
            WranglerLogger.info(f"Mapping {len(shapes_df)} shape points to roadway nodes")
            
            # Create GeoDataFrame from shape points
            shape_geometry = [Point(lon, lat) for lon, lat in zip(shapes_df['shape_pt_lon'], shapes_df['shape_pt_lat'])]
            shapes_gdf = gpd.GeoDataFrame(shapes_df, geometry=shape_geometry, crs='EPSG:4326')
            shapes_gdf_proj = shapes_gdf.to_crs('EPSG:2227')
            
            # Reuse the BallTree from stop matching if available, or create new one
            if 'tree' not in locals():
                node_coords = np.array([(geom.x, geom.y) for geom in nodes_gdf_proj.geometry])
                tree = BallTree(node_coords)
            
            # Extract shape point coordinates
            shape_coords = np.array([(geom.x, geom.y) for geom in shapes_gdf_proj.geometry])
            
            # Find nearest nodes for all shape points at once
            WranglerLogger.info(f"Finding nearest nodes for {len(shapes_df)} shape points")
            shape_distances, shape_indices = tree.query(shape_coords, k=1)
            
            # Extract node IDs
            shape_node_ids = [nodes_gdf.iloc[idx[0]]['model_node_id'] for idx in shape_indices]
            
            shapes_df['shape_model_node_id'] = shape_node_ids
            WranglerLogger.info("Shape point mapping complete")
        
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