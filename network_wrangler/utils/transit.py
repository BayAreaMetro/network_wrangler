"""Utilities for getting GTFS into wrangler"""

from ..logger import WranglerLogger
from ..models.gtfs.gtfs import GtfsModel
from ..transit.feed.feed import Feed
from ..roadway.network import RoadwayNetwork

def create_feed_from_gtfs_model(
        gtfs_model: GtfsModel,
        roadway_net: RoadwayNetwork
    ) -> Feed:
    """Converts a GTFS feed to a Wrangler Feed object compatible with the given RoadwayNetwork.

    Args:
        gtfs_model (GtfsModel): _description_
        roadway_net (RoadwayNetwork): _description_

    Returns:
        Feed: _description_
    """
    WranglerLogger.debug(f"create_feed_from_gtfsmodel()")

    # create mapping from gtfs_model stop to RoadwayNetwork nodes

    # convert gtfs_model to use those new stops
    
    # create frequencies table from GTFS stop_times (if no frequencies table is specified)

    # route gtfs buses, cable cars and light rail along roadway network

    # create Feed object from results of the above

    return Feed()