"""Tests for roadway centroid nodes and centroid connectors.

Uses the ``small`` example network as a base and builds two synthetic zones so the
end-to-end ``add_centroid_nodes`` -> ``add_centroid_connectors`` flow can be exercised
as a regression guard.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from network_wrangler import load_roadway_from_dir
from network_wrangler.roadway.centroids import (
    FitForCentroidConnection,
    add_centroid_connectors,
    add_centroid_nodes,
    calculate_angle_from_centroid,
)

LAT_LON_CRS = 4326
LOCAL_CRS = "EPSG:26915"  # UTM 15N, appropriate for the St Paul small example


@pytest.fixture
def centroid_net(small_ex_dir):
    """Fresh copy of the small network prepped for centroid connectors.

    A fresh load (rather than the shared ``small_net`` fixture) avoids mutating a
    network used by other tests.
    """
    net = load_roadway_from_dir(small_ex_dir)
    # every link is a great candidate to connect a centroid to
    net.links_df["drive_centroid_fit"] = int(FitForCentroidConnection.BEST)
    return net


@pytest.fixture
def zones_gdf(centroid_net):
    """Two zones splitting the small network's nodes east/west of a longitude line."""
    nodes = centroid_net.nodes_df
    groups = [
        (1001, nodes[nodes["X"] < -93.092]),
        (1002, nodes[nodes["X"] >= -93.092]),
    ]
    zones = []
    for zone_id, sub in groups:
        hull = sub.geometry.union_all().convex_hull.buffer(0.001)
        zones.append({"TAZ": zone_id, "geometry": hull, "geometry_centroid": hull.centroid})
    return gpd.GeoDataFrame(zones, geometry="geometry", crs=LAT_LON_CRS)


def test_calculate_angle_from_centroid_cardinal_points():
    """Angle-from-centroid returns clockwise-from-north bearings for the 4 cardinals."""
    # points due N, E, S, W of the origin centroid
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 1), Point(1, 0), Point(0, -1), Point(-1, 0)]},
        geometry="geometry",
    )
    gdf["geometry_centroid"] = [Point(0, 0)] * 4

    result = calculate_angle_from_centroid(gdf, "geometry_centroid", "angle")

    assert result["angle"].round().tolist() == [0.0, 90.0, 180.0, 270.0]


def test_add_centroid_nodes(centroid_net, zones_gdf):
    """Centroid nodes are added with model_node_id = zone id and X/Y at the centroid."""
    n_nodes_before = len(centroid_net.nodes_df)

    add_centroid_nodes(centroid_net, zones_gdf, zone_id="TAZ")

    assert len(centroid_net.nodes_df) == n_nodes_before + len(zones_gdf)
    for _, zone in zones_gdf.iterrows():
        node = centroid_net.nodes_df.loc[
            centroid_net.nodes_df["model_node_id"] == zone["TAZ"]
        ].iloc[0]
        assert node["X"] == pytest.approx(zone["geometry_centroid"].x)
        assert node["Y"] == pytest.approx(zone["geometry_centroid"].y)


def test_add_centroid_connectors(centroid_net, zones_gdf):
    """Connectors are created bidirectionally for each zone up to the requested count."""
    num_connectors = 2
    add_centroid_nodes(centroid_net, zones_gdf, zone_id="TAZ")
    n_links_before = len(centroid_net.links_df)

    summary = add_centroid_connectors(
        centroid_net,
        zones_gdf.copy(),
        zone_id="TAZ",
        mode="drive",
        local_crs=LOCAL_CRS,
        zone_buffer_distance=50,
        num_centroid_connectors=num_connectors,
        max_mode_graph_degrees=6,
        default_link_attribute_dict={"lanes": 1},
    )

    per_zone = dict(zip(summary["TAZ"], summary["num_connectors"]))
    assert per_zone == {1001: num_connectors, 1002: num_connectors}

    n_added = len(centroid_net.links_df) - n_links_before
    # bidirectional: one link each way per connector
    assert n_added == 2 * sum(per_zone.values())

    new_links = centroid_net.links_df.tail(n_added)
    for taz, count in per_zone.items():
        outbound = new_links[new_links["A"] == taz]
        inbound = new_links[new_links["B"] == taz]
        assert len(outbound) == count
        assert len(inbound) == count
        # the same nodes are connected in both directions
        assert set(outbound["B"]) == set(inbound["A"])
