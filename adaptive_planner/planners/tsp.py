from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import matplotlib

from adaptive_planner.location import Location

matplotlib.use("Agg")

from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np

from fast_tsp import find_tour
from tilemapbase import Extent, Plotter, project

from adaptive_planner import setup_logging
from adaptive_planner.georeference import get_field_of_view_m, get_fov_polygon
from adaptive_planner.planners import Planner
from adaptive_planner.utils import Waypoint, create_distance_matrix, parallel_execute
from adaptive_planner.visualisation import get_tiles

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from adaptive_planner.field import Field
    from adaptive_planner.utils import CameraParameters

log = setup_logging(__name__)


@dataclass
class TSPPlannerConfig:
    max_tsp_calculation_time: float = 2.0
    filter_waypoints: bool = True
    filter_min_overlap: float = 0.10

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TSPPlanner(Planner):
    def __init__(self, field: Field, camera_parameters: CameraParameters, config: TSPPlannerConfig) -> None:
        self.field = field
        self.camera_parameters = camera_parameters
        self.config = config

        self._objects_to_visit: list[Location] | None = None
        self._start_location: Location | None = None
        self._end_location: Location | None = None

    @property
    def config_dict(self) -> dict[str, Any]:
        return self.config.as_dict()

    def plan(self, start_location: Location, altitude: float, output_file: Path | None = None) -> list[Waypoint]:
        assert self._objects_to_visit is not None, "Run 'set_objects_and_end_location()' first!"
        assert all([loc.has_altitude for loc in self._objects_to_visit]), "All locations needs an alitude!"

        if len(self._objects_to_visit) == 0:
            return []

        self._start_location = start_location

        # Not best implementation ever...
        if self._end_location is None:
            start_time = time()
            log.info("Start brute-forcing open-ended TSP")
            routes = parallel_execute(
                {str(i): loc for i, loc in enumerate(self._objects_to_visit)}, self._calculate_waypoints_between_locations
            )

            min_distance = float("inf")
            route = None
            for possible_route in routes.values():
                distance = sum(
                    [possible_route[i - 1].location.get_distance(possible_route[i].location) for i in range(1, len(possible_route))]
                )
                if distance < min_distance:
                    min_distance = distance
                    route = possible_route

            log.info(f"Brute-forced TSP problem in {time() - start_time:.2f} seconds, shortest path {min_distance:.2f}m")
            assert route is not None
        else:
            # Create route
            route = self.locations_to_waypoints(self._objects_to_visit, start_location, self._end_location)

        if self.config.filter_waypoints:
            n_original = len(route)

            # Check all waypoints whether a next waypoint is already visible in the current waypoint. If so, we can remove that
            # waypoint and speed up the process.
            i = 0
            while i < len(route):
                fov_m = get_field_of_view_m(route[i].location.altitude, self.camera_parameters) * (1 - self.config.filter_min_overlap)
                fov_polygon = get_fov_polygon(route[i], fov_m)

                j = i + 1
                while j < len(route):
                    if fov_polygon.contains(route[j].location.to_point()):
                        # Check if the waypoint we want to remove is lower than the next waypoint. If that's the case, check
                        # if the waypoint is still visible when we lower the original waypoint. If not, keep this waypoint.
                        if route[j].location.altitude < route[i].location.altitude:
                            _tmp_fov_m = get_field_of_view_m(route[j].location.altitude, self.camera_parameters) * (
                                1 - self.config.filter_min_overlap
                            )
                            _tmp_fov_polygon = get_fov_polygon(route[i], _tmp_fov_m)
                            if not _tmp_fov_polygon.contains(route[j].location.to_point()):
                                j += 1
                                continue

                            route[i].location.gps_coordinate_lat_lon[2] = route[j].location.altitude

                        del route[j]

                        # Adapt the heading of the next waypoint
                        if j < len(route):
                            route[j].heading = route[j - 1].location.get_heading(route[j].location)
                    else:
                        j += 1
                i += 1

            log.info(f"Created route with {len(route)} waypoint(s), filtered out {n_original - len(route)} waypoint(s)")

        if output_file is not None:
            self.plot(start_location, route, self._end_location, output_file)

        return route

    def set_objects_and_end_location(self, locations: list[Location], end_location: Location | None) -> None:
        self._objects_to_visit = locations
        self._end_location = end_location

    def locations_to_waypoints(self, locations: list[Location], start_location: Location, end_location: Location) -> list[Waypoint]:
        assert self._objects_to_visit is not None

        locations = [start_location] + self._objects_to_visit + [end_location]

        # Create distance matrix. Add dummy node that forces the start and end location to be planned after each other by setting
        # their distance to zero. All other distances for dummy node should be high. See
        # https://stackoverflow.com/questions/14527815/how-to-fix-the-start-and-end-points-in-travelling-salesmen-problem. Roll
        # in such way that we start with the start location and remove both end location and the dummy location.
        distance_matrix = np.zeros((len(locations) + 1, len(locations) + 1), dtype=np.float32)
        distance_matrix[0:-1, 0:-1] = create_distance_matrix(locations, locations, 1e7)
        distance_matrix[1:-2, -1] = 655.0
        distance_matrix[-1, 1:-2] = 655.0

        # FastTSP only accept integers, so convert to floats in cm (max distance is 655.35 meter now)
        distance_matrix *= 100
        if np.any(distance_matrix > np.iinfo(np.uint16).max):
            log.warning("Error in converting distance matrix for TSP solver, some max values are present. Path is not optimal.")

        tour = np.array(find_tour(distance_matrix.astype(np.uint16), duration_seconds=self.config.max_tsp_calculation_time), dtype=np.uint8)

        # Roll to make dummy point last, remove dummy point
        tour = np.roll(tour, -np.where(tour == len(locations))[0][0] - 1)[:-1]

        # Reverse tour when start point is at the end
        if tour[-1] == 0:
            tour = np.flip(tour)

        assert len(tour) == len(locations)
        assert tour[0] == 0 and tour[-1] == len(locations) - 1

        # Create route from tour, skip the start point and end point
        route: list[Waypoint] = []
        prev_location = locations[0]
        for i in tour[1:-1]:
            route.append(Waypoint(locations[i], prev_location.get_heading(locations[i])))
            prev_location = locations[i]

        return route

    def _calculate_waypoints_between_locations(self, end_location: Location) -> list[Waypoint]:
        assert self._start_location is not None
        assert self._objects_to_visit is not None

        objects_to_visit = deepcopy(self._objects_to_visit)
        objects_to_visit = [obs for obs in objects_to_visit if obs != end_location]
        return self.locations_to_waypoints(self._objects_to_visit, self._start_location, end_location)

    @staticmethod
    def plot(start_location: Location, waypoints: list[Waypoint], end_location: Location | None, output_file: Path) -> None:
        waypoints_arr = np.array(
            [start_location.gps_coordinate_lon_lat[:2]]
            + [wp.location.gps_coordinate_lon_lat[:2] for wp in waypoints]
            + ([] if end_location is None else [end_location.gps_coordinate_lon_lat[:2]]),
            dtype=np.float64,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        expand = 0.0005
        extent = Extent.from_lonlat(
            waypoints_arr[:, 0].min() - expand,
            waypoints_arr[:, 0].max() + expand,
            waypoints_arr[:, 1].min() - expand,
            waypoints_arr[:, 1].max() + expand,
        )

        waypoints_projected = np.array(
            [project(lon, lat) for lon, lat in zip(waypoints_arr[:, 0], waypoints_arr[:, 1])],
            dtype=np.float64,
        )

        tiles = get_tiles("ArcGis")

        plotter = Plotter(extent, tiles, height=600)
        plotter.plot(ax, tiles, alpha=0.8)

        ax.plot(waypoints_projected[:, 0], waypoints_projected[:, 1], marker="o", linestyle="-", color="b", markersize=3, linewidth=1)
        ax.annotate(
            "Start",
            xy=tuple(waypoints_projected[0, :]),
            xytext=(-20, 20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
        )

        if end_location is not None:
            ax.annotate(
                "End",
                xy=tuple(waypoints_projected[-1, :]),
                xytext=(20, -20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->"),
            )

        plt.savefig(str(output_file))

        log.info(f"Saved inspection flight path as {output_file}")


if __name__ == "__main__":
    from pathlib import Path

    from adaptive_planner.utils import CameraParameters

    coordinates1 = [
        [5.667584777851703, 51.99122071405578, 12.0],
        [5.667630674929676, 51.99127707303309, 12.0],
        [5.667599004528068, 51.99125032034756, 12.0],
        [5.667629612897581, 51.9912254452522, 12.0],
        [5.6676683305091755, 51.991279376274264, 12.0],
        [5.66769246688835, 51.99126211968117, 12.0],
        [5.668256826573724, 51.99140589941254, 12.0],
        [5.668283427683491, 51.991339397608456, 12.0],
        [5.668333395554749, 51.991384722837516, 12.0],
        [5.668350799049933, 51.99141511049398, 12.0],
        [5.668347589997577, 51.99134062676789, 12.0],
        [5.668468248720187, 51.991413150005066, 12.0],
        [5.66843600144797, 51.99136561980934, 12.0],
        [5.668481743885395, 51.99139594060085, 12.0],
        [5.668580822720089, 51.991399029970786, 12.0],
        [5.668522657935431, 51.99139127898629, 12.0],
        [5.668509984903117, 51.99141877265734, 12.0],
        [5.668536546922321, 51.99134049587956, 12.0],
        [5.668637656762255, 51.991296661552006, 12.0],
        [5.668575152908792, 51.99135692461951, 12.0],
        [5.668398277318661, 51.99133629945695, 12.0],
        [5.668538476910557, 51.99130297790723, 12.0],
        [5.668500272009812, 51.99132248493283, 12.0],
        [5.668486921113372, 51.99135071395581, 12.0],
        [5.668301418514703, 51.99129448614687, 12.0],
        [5.668240036657179, 51.99113799832376, 12.0],
        [5.668217255502952, 51.9911132504913, 12.0],
        [5.66813285112839, 51.99113689336823, 12.0],
        [5.667565608794468, 51.991209088426004, 12.0],
        [5.6676621918730365, 51.99120501016205, 12.0],
        [5.6675900692170815, 51.991200909801535, 12.0],
        [5.667637762770028, 51.99088368348835, 12.0],
        [5.6676259921748136, 51.990882938375044, 12.0],
        [5.667575785234119, 51.99088358310053, 12.0],
        [5.667660859935959, 51.990881624345114, 12.0],
        [5.667669051169184, 51.99089257798221, 12.0],
        [5.6675948688763595, 51.990872003440515, 12.0],
        [5.667597699969292, 51.99089333195605, 12.0],
        [5.66763144137898, 51.990898840349615, 12.0],
        [5.6676573477197065, 51.99090027733002, 12.0],
        [5.667579901401265, 51.99089281121947, 12.0],
        [5.667559328312011, 51.99086398927092, 12.0],
        [5.667615082954917, 51.99087128835423, 12.0],
        [5.667599231594936, 51.99088264193727, 12.0],
        [5.667619034837152, 51.99089411128184, 12.0],
        [5.6677440683015865, 51.991026269731805, 12.0],
        [5.66774973819746, 51.99100066750135, 12.0],
        [5.667719997532704, 51.99101677165243, 12.0],
        [5.66777374785127, 51.99102428696146, 12.0],
        [5.6678052872409035, 51.99100818772208, 12.0],
        [5.667833366727379, 51.99100348521301, 12.0],
        [5.667846364680611, 51.991034219942776, 12.0],
        [5.667852338122615, 51.99101835770096, 12.0],
        [5.667821734261427, 51.99102198300441, 12.0],
        [5.668195603952817, 51.99106148003229, 12.0],
        [5.668142387492557, 51.99105944304999, 12.0],
        [5.668127568423111, 51.99109502822443, 12.0],
        [5.6682668580411075, 51.99106501397474, 12.0],
        [5.6682415859252435, 51.991102446208394, 12.0],
        [5.668285802705676, 51.99109088182371, 12.0],
        [5.668238907699002, 51.991136367502, 12.0],
        [5.668161730240213, 51.991090614200736, 12.0],
    ]
    coordinates2 = [
        [5.6683508729341785, 51.99141512463226, 12.0],
        [5.668256882340563, 51.991405901754526, 12.0],
        [5.668509947713698, 51.991418789565046, 12.0],
        [5.6688155532492175, 51.991341065593474, 12.0],
        [5.668582253369382, 51.99140023700125, 12.0],
        [5.668575086906041, 51.99135685248964, 12.0],
        [5.668536558215693, 51.991340508558324, 12.0],
        [5.668486962710177, 51.99135067550645, 12.0],
        [5.668538327891758, 51.99130296792472, 12.0],
        [5.668523605324069, 51.9913933135282, 12.0],
        [5.668500236271894, 51.99132257408095, 12.0],
        [5.668437500257347, 51.99136695653604, 12.0],
        [5.668469877864196, 51.99141445897827, 12.0],
        [5.668482678136141, 51.99139726216908, 12.0],
        [5.668334616002449, 51.991386010981756, 12.0],
        [5.668348893511259, 51.99134185277189, 12.0],
        [5.668398217375799, 51.991336302065875, 12.0],
        [5.66828507633718, 51.991340441519824, 12.0],
        [5.6683014996790915, 51.9912944625261, 12.0],
        [5.667693460509651, 51.99126331112294, 12.0],
        [5.6675863181881345, 51.991221918138756, 12.0],
        [5.66766957374949, 51.99128071382745, 12.0],
        [5.667631687430804, 51.991278405674265, 12.0],
        [5.667630818809318, 51.99122662052888, 12.0],
        [5.667590032995575, 51.99120098041062, 12.0],
        [5.667565627126157, 51.99120909286452, 12.0],
        [5.66760024019204, 51.99125153677176, 12.0],
        [5.667662001618064, 51.99120492970175, 12.0],
        [5.668215780092921, 51.991111883820096, 12.0],
        [5.6681274168271525, 51.99109501515548, 12.0],
        [5.668131493407777, 51.991135968156186, 12.0],
        [5.6682160569831765, 51.99111192296149, 12.0],
        [5.66823899202837, 51.99113649860726, 12.0],
        [5.6686361610124765, 51.99129535666083, 12.0],
        [5.668268194684343, 51.991066470978346, 12.0],
        [5.668286554343871, 51.991092136071636, 12.0],
        [5.668242738780879, 51.99110371099658, 12.0],
        [5.668196562785869, 51.991062865158845, 12.0],
        [5.6682681506316035, 51.991066464751015, 12.0],
        [5.6681442020739645, 51.991060745970564, 12.0],
        [5.667834441607074, 51.991004434219164, 12.0],
        [5.667847058471679, 51.99103554913583, 12.0],
        [5.6678229664295845, 51.99102307430468, 12.0],
        [5.667853573546181, 51.991019440225685, 12.0],
        [5.667807050346441, 51.991009224784136, 12.0],
        [5.667744919685084, 51.991027944439985, 12.0],
        [5.667720684658555, 51.99101820149307, 12.0],
        [5.667775449894441, 51.99102540774502, 12.0],
        [5.667599318301628, 51.990893582263276, 12.0],
        [5.667620940901332, 51.99089434862483, 12.0],
        [5.667600621222386, 51.99088286877931, 12.0],
        [5.667658981673451, 51.99090044667519, 12.0],
        [5.667671016398221, 51.99089283878365, 12.0],
        [5.667581142533779, 51.99089298916577, 12.0],
        [5.667633306011843, 51.99089919970646, 12.0],
        [5.667577567235218, 51.99088385794919, 12.0],
        [5.667535486013939, 51.990954709947104, 12.0],
        [5.667615003981307, 51.99087132349898, 12.0],
        [5.667594872599699, 51.990872026201885, 12.0],
        [5.667625954943811, 51.99088298159878, 12.0],
        [5.667559478351519, 51.99086396464736, 12.0],
        [5.667638460655683, 51.99088237478757, 12.0],
        [5.667661294316606, 51.990880523281064, 12.0],
        [5.668437039489089, 51.991402962626545, 12.0],
    ]
    loc = [Location(np.array(coord, dtype=np.float64)) for coord in coordinates1]
    end_location = Location(np.array([5.667294374220376, 51.99130512420879, 24.0], dtype=np.float64))
    # end_location = None
    start_location = Location(np.array([5.6675730919265845, 51.99087642389226, 24.0], dtype=np.float64))

    camera = CameraParameters((8192, 5460), (35.9, 24.0), 35.0)
    planner = TSPPlanner(Field("", Path(), None, None), camera, TSPPlannerConfig())

    planner.set_objects_and_end_location(loc, end_location)
    planner.plan(start_location, 12.0, output_file=Path("test.png"))
