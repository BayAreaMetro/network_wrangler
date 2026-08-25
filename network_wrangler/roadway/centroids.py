"""Functions to create centroid connectors.

Example usage:

```python
# zone_id_col is the name of the identifier column in zones_gdf.
add_centroid_nodes(road_net, zones_gdf, zone_id_col="TAZ_NODE")
add_centroid_connectors(
    road_net,
    zones_gdf,
    zone_id_col="TAZ_NODE",
    mode="drive",
    local_crs="EPSG:26915",
)
```

See docs/how_to.md for a fuller end-to-end network creation workflow.
"""

from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
import shapely.geometry

from ..logger import WranglerLogger
from ..models.roadway.tables import ZonesTable
from ..params import LAT_LON_CRS, MODES_TO_NETWORK_LINK_VARIABLES
from ..utils.geo import point_bearings_degrees
from ..utils.models import validate_df_to_model
from .network import RoadwayNetwork


class FitForCentroidConnection(IntEnum):
    """Indicates the fitness of this link to be connected to a centroid connector.

    Since the connector will connect to a node, the highest (worst) of this value
    for the links will apply to the node. So if one link is DO_NOT_USE, it won't be used.

    Used by add_centroid_connectors().
    """

    NA_IS_CONNECTOR = 0
    BEST = 1
    GOOD = 2
    OKAY = 3
    DO_NOT_USE = 100


def _validate_zones_for_centroids(
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str,
) -> DataFrame[ZonesTable]:
    """Validate centroid-zone input schema and coerce zone IDs to integers.

    Internally validates against ``ZonesTable`` by temporarily normalizing the
    user-specified ``zone_id_col`` column name to ``zone_id``.
    """
    if zone_id_col not in zones_gdf.columns:
        msg = f"zones_gdf is missing required zone id column: {zone_id_col}"
        raise ValueError(msg)

    normalized = zones_gdf.rename(columns={zone_id_col: "zone_id"}).copy()
    validated = validate_df_to_model(normalized, ZonesTable)
    return validated.rename(columns={"zone_id": zone_id_col})


def calculate_angle_from_centroid(
    gdf: gpd.GeoDataFrame,
    centroid_col: str = "geometry_centroid",
    angle_col: str = "angle_from_north",
) -> gpd.GeoDataFrame:
    """Add the bearing from each zone centroid to its point, clockwise from north.

    Adds a new column to the GeoDataFrame containing the bearing in degrees (0-360)
    measured clockwise from north of the line from each centroid to its corresponding
    point geometry.

    Args:
        gdf: GeoDataFrame with point geometries in the geometry column.
        centroid_col: Name of the column containing the centroid point geometry
            (default: 'geometry_centroid').
        angle_col: Name of the new column to create with bearing values
            (default: 'angle_from_north').

    Returns:
        GeoDataFrame with the new bearing column added.

    Note:
        - Bearing is measured clockwise from north (0=north, 90=east, 180=south, 270=west)
          in the coordinate system of ``gdf``, so the geometries should be in a projected
          CRS for a planar bearing.
        - Returns values in the range [0, 360).
    """
    gdf[angle_col] = point_bearings_degrees(gdf[centroid_col], gdf.geometry)
    return gdf


def add_centroid_nodes(
    road_net: RoadwayNetwork,
    zones_gdf: DataFrame[ZonesTable],
    zone_id_col: str,
    default_node_attribute_dict: dict[str, any] | None = None,
):
    """Adds the given centroid nodes to the roadway network.

    Args:
        road_net: the RoadwayNetwork to update by adding centroids
        zones_gdf: zone definitions with polygon geometry in ``geometry`` and
            centroid point geometry in ``geometry_centroid``.
        zone_id_col: name of the zone identifier column in zones_gdf. Values in this
            column are used as ``model_node_id`` for the created centroid nodes.
            This argument is the *column name* (for example ``"TAZ_NODE"``),
            not a specific zone value. The values must be integer-like.
        default_node_attribute_dict: node attributes to set for the new centroid nodes.
            Defaults to None.
    """
    zones_gdf = _validate_zones_for_centroids(zones_gdf, zone_id_col)

    centroid_nodes_gdf = (
        zones_gdf[[zone_id_col, "geometry_centroid"]]
        .rename(columns={"geometry_centroid": "geometry", zone_id_col: "model_node_id"})
        .set_geometry("geometry", crs=LAT_LON_CRS)
    )
    centroid_nodes_gdf["X"] = centroid_nodes_gdf["geometry"].x
    centroid_nodes_gdf["Y"] = centroid_nodes_gdf["geometry"].y
    # Centroids are synthetic nodes, so do not assign fabricated OSM IDs.
    centroid_nodes_gdf["osm_node_id"] = None

    # set default node attributes
    centroid_nodes_gdf = centroid_nodes_gdf.assign(**(default_node_attribute_dict or {}))

    # assume the model_node_id
    len_road_net_nodes = len(road_net.nodes_df)
    WranglerLogger.debug(f"centroid_nodes_gdf:\n{centroid_nodes_gdf}")
    road_net.add_nodes(centroid_nodes_gdf)
    WranglerLogger.info(
        f"Added node centroids for {zone_id_col}: "
        f"increased size of nodes_df from {len_road_net_nodes:,} to {len(road_net.nodes_df):,}"
    )


def add_centroid_connectors(  # noqa: PLR0912, PLR0915
    road_net: RoadwayNetwork,
    zones_gdf: DataFrame[ZonesTable],
    zone_id_col: str,
    mode: str,
    local_crs: str,
    zone_buffer_distance: int,
    num_centroid_connectors: int,
    max_mode_graph_degrees: int,
    default_link_attribute_dict: dict[str, any] | None = None,
) -> gpd.GeoDataFrame:
    """Creates centroid connector links between zone centroids and roadway network nodes.

    This function identifies suitable roadway nodes for each zone and creates connector links
    from the zone centroid to those nodes. The selection process prioritizes nodes based on
    their fitness for centroid connections and ensures good spatial distribution.

    Selection Algorithm:
        1. **Node Evaluation**: For each node in the modal graph, calculates:
            - Outgoing degree (number of outbound links)
            - Fitness for centroid connection (worst fitness of connected links)

        2. **Spatial Filtering**: Identifies nodes within each zone boundary and filters out:
            - Nodes with `{mode}_centroid_fit` = DO_NOT_USE
            - Nodes outside zone boundaries
            - Nodes with outgoing degree > max_mode_graph_degrees

        3. **Connector Selection** (per zone):
            - First connector: Node with best fitness and closest to centroid
            - Additional connectors: For each subsequent connector (up to num_centroid_connectors):
                * Among nodes with the best available fitness level
                * Select the one with maximum angular separation from existing connectors
                * This ensures spatial distribution while prioritizing network suitability

        4. **Link Creation**: Creates bidirectional links between zone centroid and selected nodes

    Centroid Connector Link Attributes:
        - **model_link_id**: Auto-incremented from max existing link ID
        - **A, B**: Origin and destination node IDs (bidirectional, so both directions created)
        - **name**: Set to "node to {zone_id}" or "{zone_id} to node"
        - **length**: Euclidean distance between centroid and node (in local_crs units)
        - **geometry**: LineString from origin to destination
        - **highway**: Set to zone_id value (if highway column exists in network)
        - **Mode access variables**:
            * All mode variables set to False by default
            * Variables for specified mode set to True (from MODES_TO_NETWORK_LINK_VARIABLES)
            * Example: For mode='drive', sets drive_access=True, bike_access=False, etc.
        - **Custom attributes**: Any attributes from default_link_attribute_dict parameter

    Args:
        road_net: the RoadwayNetwork to update by adding centroid connectors.
            Assumes centroids exist as nodes already. Also assumes links have
            an attribute, `{mode}_centroid_fit`, set to one of the FitForCentroidConnect values.
        zones_gdf: zone definitions with polygon geometry in ``geometry`` and
            centroid point geometry in ``geometry_centroid`` (in ``LAT_LON_CRS``).
            Must also include the identifier column named by ``zone_id_col``.
        zone_id_col: name of the identifier column in ``zones_gdf`` (for example
            ``"TAZ_NODE"``). Values from this column map to
            centroid ``model_node_id`` values.
            Zone identifier values must be integer-like.
        mode: one of the keys in [`MODES_TO_NETWORK_LINK_VARIABLES`][network_wrangler.params.MODES_TO_NETWORK_LINK_VARIABLES]
        local_crs: CRS to use for distance calculations
        zone_buffer_distance: buffer distance from zone shape to consider node for centroid connector.
            This should be in the units of the local_crs.
        num_centroid_connectors: maximum number of centroid connectors per zone
        max_mode_graph_degrees: maximum outgoing degree for a node to be eligible
        default_link_attribute_dict: link attributes to set for the new centroid connector links.
            Defaults to None.

    Returns:
        A copy of zones_gdf with an additional column, `num_connectors`. The road_net is
            updated in place with new centroid connector links, and the nodes table has an
                additional column: `{zone_id_col}_num_connectors`.

    """
    zones_gdf = _validate_zones_for_centroids(zones_gdf, zone_id_col)

    WranglerLogger.info(f"Adding centroid connectors for zone:{zone_id_col} and mode:{mode}")
    WranglerLogger.debug(f"zones_gdf:\n{zones_gdf}")

    degrees_col = f"{mode}_graph_degrees"
    fit_col = f"{mode}_centroid_fit"

    # Evaluate each node's fitness to host a centroid connector from the modal links.
    # A node's out-degree is the number of modal links starting at it (A == node) and
    # its fitness is the worst (max) fit of those links.
    modal_links_df = road_net.links_df.mode_query(mode)
    scores_df = modal_links_df.groupby("A")[fit_col].agg(
        **{degrees_col: "size", fit_col: "max"}
    )
    # drop nodes that must not be used as a centroid connector
    scores_df = scores_df[scores_df[fit_col] != FitForCentroidConnection.DO_NOT_USE]

    # attach all node attributes (including geometry) for the usable nodes
    mode_node_df = gpd.GeoDataFrame(
        pd.merge(
            left=road_net.nodes_df,
            right=scores_df,
            left_on="model_node_id",
            right_index=True,
            how="inner",
            validate="one_to_one",
        ),
        geometry="geometry",
        crs=LAT_LON_CRS,
    )
    WranglerLogger.debug(
        f"Evaluated {len(mode_node_df):,} usable {mode} nodes for centroid connectors:\n"
        f"{mode_node_df}"
    )

    # project to the local CRS for distance/angle calculations
    mode_node_df.to_crs(local_crs, inplace=True)
    zones_gdf.to_crs(local_crs, inplace=True)

    # spatial intersect nodes with zones
    mode_node_df = gpd.sjoin(
        left_df=mode_node_df,
        right_df=zones_gdf[[zone_id_col, "geometry", "geometry_centroid"]],
        how="left",
        predicate="dwithin",  # give zones a little buffer because of edge cases
        distance=zone_buffer_distance,
    )
    # This means that model_node_id is not unique -- but that's ok, a node can
    # be connected to multiple centroid connectors
    WranglerLogger.debug(
        f"After spatial join, non-unique model_node_ids:\n"
        f"{mode_node_df.loc[mode_node_df['model_node_id'].duplicated(keep=False)]}"
    )

    # calculate distance from centroid
    gs = gpd.GeoSeries(mode_node_df["geometry_centroid"], crs=LAT_LON_CRS)  # source CRS
    mode_node_df["geometry_centroid"] = gs.to_crs(local_crs).values  # target CRS
    mode_node_df["distance_from_centroid"] = mode_node_df.geometry.distance(
        gpd.GeoSeries(mode_node_df["geometry_centroid"], crs=local_crs)
    )
    WranglerLogger.debug(
        f"After spatial join, mode_node_df type={type(mode_node_df)}:\n{mode_node_df}"
    )

    # add angle fron centroid
    mode_node_df = calculate_angle_from_centroid(
        mode_node_df, "geometry_centroid", "centroid_angle"
    )
    WranglerLogger.debug(
        f"After adding angle from centroid, mode_node_df type={type(mode_node_df)}:\n{mode_node_df}"
    )

    # Filter to nodes within the given zones
    mode_node_df = mode_node_df.loc[mode_node_df[zone_id_col].notna()]
    mode_node_df[zone_id_col] = mode_node_df[zone_id_col].astype(int)
    # and mode_graph_degress <= max_mode_graph_degress
    mode_node_df = mode_node_df.loc[
        mode_node_df[f"{mode}_graph_degrees"] <= max_mode_graph_degrees
    ]

    # sort by drive_centroid_fit, centroid_angle
    mode_node_df.sort_values(
        by=[zone_id_col, f"{mode}_centroid_fit", "distance_from_centroid"], inplace=True
    )
    mode_node_df.reset_index(drop=True, inplace=True)
    mode_node_df["connector_num"] = 0

    WranglerLogger.debug(
        f"Before choosing centroid connector nodes, mode_node_df:\n{mode_node_df}"
    )

    fit_col = f"{mode}_centroid_fit"

    # Process each zone and select connectors with incremental min-angle updates.
    # Rows are pre-sorted by [zone_id, fit_col, distance_from_centroid], so within each zone
    # row 0 is always the best-fit and closest seed connector.
    for zone_num, zone_data in mode_node_df.groupby(zone_id_col, sort=False):
        if zone_data.empty:
            WranglerLogger.warning(f"No centroid connectors for {zone_id_col} {zone_num}")
            continue

        idx = zone_data.index.to_numpy()
        angles = zone_data["centroid_angle"].to_numpy()
        fit = zone_data[fit_col].to_numpy()
        conn = np.zeros(len(zone_data), dtype=int)

        def circ_sep(a: float) -> np.ndarray:
            d = np.abs(angles - a)
            return np.minimum(d, 360 - d)

        conn[0] = 1
        min_sep = circ_sep(angles[0])

        for connector_num in range(2, num_centroid_connectors + 1):
            unsel = conn == 0
            if not unsel.any():
                break

            best_fit = fit[unsel].min()
            eligible = unsel & (fit == best_fit)
            pick = int(np.argmax(np.where(eligible, min_sep, -np.inf)))
            conn[pick] = connector_num
            min_sep = np.minimum(min_sep, circ_sep(angles[pick]))

        mode_node_df.loc[idx, "connector_num"] = conn

    # Filter to only selected connectors
    mode_node_df = mode_node_df[mode_node_df["connector_num"] > 0]
    mode_node_df.sort_values(by=[zone_id_col, "connector_num"], inplace=True)
    mode_node_df.reset_index(drop=True, inplace=True)

    WranglerLogger.info(
        f"Selected {len(mode_node_df):,} centroid connectors for {len(zones_gdf):,} {zone_id_col}s"
    )
    WranglerLogger.debug(f"mode_node_df:\n{mode_node_df}")
    # create centroid connector links: zone to node
    links_taz_to_node_df = mode_node_df.copy()
    links_taz_to_node_df.rename(
        columns={zone_id_col: "A", "model_node_id": "B", "distance_from_centroid": "length"},
        inplace=True,
    )
    links_taz_to_node_df["name"] = f"{zone_id_col} to node"
    links_taz_to_node_df["geometry"] = links_taz_to_node_df.apply(
        lambda row: shapely.geometry.LineString([row["geometry_centroid"], row["geometry"]]),
        axis=1,
    )
    # create centroid connector links: node to zone
    links_node_to_taz_df = mode_node_df.copy()
    links_node_to_taz_df.rename(
        columns={"model_node_id": "A", zone_id_col: "B", "distance_from_centroid": "length"},
        inplace=True,
    )
    links_node_to_taz_df["name"] = f"node to {zone_id_col}"
    links_node_to_taz_df["geometry"] = links_node_to_taz_df.apply(
        lambda row: shapely.geometry.LineString([row["geometry"], row["geometry_centroid"]]),
        axis=1,
    )

    # Put together zone to node and node to zone
    centroid_links_df = pd.concat([links_taz_to_node_df, links_node_to_taz_df])
    centroid_links_df.reset_index(drop=False, inplace=True)

    # select minimal columns
    centroid_links_df = centroid_links_df[["A", "B", "name", "length", "geometry"]]
    WranglerLogger.debug(f"centroid_links_df:\n{centroid_links_df}")

    max_model_link_id = road_net.links_df.model_link_id.max()
    centroid_links_df["model_link_id"] = centroid_links_df.index + max_model_link_id + 1
    centroid_links_df["shape_id"] = "sh" + centroid_links_df["model_link_id"].astype("str")
    # default to False
    link_mode_variables = set()
    for _mode, link_vars in MODES_TO_NETWORK_LINK_VARIABLES.items():
        link_mode_variables.update(link_vars)
    WranglerLogger.debug(f"link_mode_variables:{link_mode_variables}")

    for link_var in link_mode_variables:
        centroid_links_df[link_var] = False
    # but set the ones for this mode to True
    for link_var in MODES_TO_NETWORK_LINK_VARIABLES[mode]:
        centroid_links_df[link_var] = True

    if "highway" in road_net.links_df.columns:
        centroid_links_df["highway"] = zone_id_col

    # set default link attributes
    if default_link_attribute_dict is None:
        default_link_attribute_dict = {}
    for colname, default_value in default_link_attribute_dict.items():
        centroid_links_df[colname] = default_value

    road_net.add_links(centroid_links_df)
    road_net.add_shapes(centroid_links_df)
    WranglerLogger.info("Added centroid connectors to roadway network")

    # summarize number of connectors per zone
    summary_df = (
        mode_node_df.groupby(by=zone_id_col)
        .aggregate(num_connectors=pd.NamedAgg(column="model_node_id", aggfunc="nunique"))
        .reset_index(drop=False)
    )
    # join to zones_gdf to see if we missed any zones
    summary_df = pd.merge(left=zones_gdf, right=summary_df, how="left", validate="one_to_one")
    summary_df["num_connectors"] = summary_df["num_connectors"].fillna(0)
    summary_df["num_connectors"] = summary_df["num_connectors"].astype(int)
    WranglerLogger.debug(f"summary_df:\n{summary_df}")
    WranglerLogger.info(
        f"num_connectors added per {zone_id_col} (target:{num_centroid_connectors}):\n"
        f"{summary_df['num_connectors'].value_counts()}"
    )

    # add column {zone_id_col}_num_connectors to nodes
    road_net.nodes_df = pd.merge(
        left=road_net.nodes_df,
        right=summary_df[[zone_id_col, "num_connectors"]].rename(
            columns={zone_id_col: "model_node_id", "num_connectors": f"{zone_id_col}_num_connectors"}
        ),
        how="left",
        validate="one_to_one",
    )
    WranglerLogger.debug(f"road_net.nodes_df:\n{road_net.nodes_df}")
    return summary_df
