from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntFlag
from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import linear_sum_assignment

from adaptive_planner import setup_logging
from adaptive_planner.field import Field
from adaptive_planner.georeference import georeference_detections, get_field_of_view_m, get_fov_polygon
from adaptive_planner.location import Location
from adaptive_planner.predict import Detection
from adaptive_planner.utils import Waypoint, create_distance_matrix

if TYPE_CHECKING:
    from adaptive_planner.utils import CameraParameters

log = setup_logging(__name__)


class TrackerStatus(IntFlag):
    ACCEPTED = 1
    INVESTIGATED = 2
    TO_BE_INVESTIGATED = 4
    REJECTED = 8


@dataclass
class GPSTrack:
    name: str
    status: TrackerStatus
    location: Location
    detections: list[Detection | None] = field(default_factory=list)
    altitudes: list[float] = field(default_factory=list)
    detected_locations: list[Location | None] = field(default_factory=list)

    def to_location(self) -> Location:
        location = self.location.copy()
        location.gps_coordinate_lon_lat = location.gps_coordinate_lon_lat[:2]
        location.properties["name"] = self.name
        location.properties["status"] = self.status.name
        location.properties["drone_altitudes"] = self.altitudes

        location.properties.setdefault("detected_locations", [])
        for loc in self.detected_locations:
            location.properties["detected_locations"].append(None if loc is None else loc.gps_coordinate_lon_lat.tolist()[:2])

        location.properties.setdefault("confidences", [])
        location.properties.setdefault("class_names", [])
        location.properties.setdefault("best_class_name", "")
        location.properties.setdefault("best_confidence", -float("inf"))
        location.properties.setdefault("class_confidences", defaultdict(list))
        for i, dt in enumerate(self.detections):
            if dt is None:
                location.properties["confidences"].append(None)
                location.properties["class_names"].append(None)
                continue

            location.properties["confidences"].append(dt.confidence)
            location.properties["class_names"].append(dt.class_name)

            if dt.confidence > location.properties["best_confidence"]:
                location.properties["best_class_name"] = dt.class_name
                location.properties["best_confidence"] = dt.confidence

            if dt.class_confidences is not None:
                for k, v in dt.class_confidences.items():
                    location.properties["class_confidences"][k].append(v)

            location.properties["class_confidences"] = dict(location.properties["class_confidences"])

        return location

    @classmethod
    def from_location(
        cls: type[GPSTrack], location: Location, rejection_confidence: float = 0.05, inspection_confidence: float = 0.4
    ) -> GPSTrack | None:
        name = location.properties["name"]

        _confidences = location.properties["confidences"]
        _confidences = [_confidences] if isinstance(_confidences, float) else list(_confidences)

        _altitudes = location.properties["drone_altitudes"]
        _altitudes = [_altitudes] if isinstance(_altitudes, float) else list(_altitudes)

        _class_names = location.properties["class_names"].split(",")

        _detected_locations = location.properties["detected_locations"]
        _detected_locations = [_detected_locations] if isinstance(_detected_locations, list) else list(_detected_locations)

        assert len(_confidences) == len(_altitudes) == len(_class_names) == len(_detected_locations)

        valid_location = None
        detection_list: list[Detection | None] = []
        detected_locations_list: list[Location | None] = []
        altitudes: list[float] = []

        for i, confidence in enumerate(_confidences):
            # Ignore missed detetions when there are no detections yet
            if confidence is None and len(detection_list) == 0:
                continue

            # Just add the missed detections again
            elif confidence is None:
                detection_list.append(None)
                detected_locations_list.append(None)
                altitudes.append(_altitudes[i])
                continue

            # Ignore detections that are more uncertain than the minimum threshold
            elif _confidences[i] < rejection_confidence:
                continue

            detection_location = Location(np.array([*_detected_locations[i], _altitudes[i]], dtype=np.float64))

            # For the first valid detection, keep the location
            if valid_location is None:
                valid_location = detection_location.copy()

            class_confidences = None
            if "class_confidences" in location.properties:
                class_confidences = {
                    k: v[len([dt for dt in detection_list if dt is not None])] for k, v in location.properties["class_confidences"].items()
                }

            detection_list.append(
                Detection(0, 0, 0, 0, _class_names[i], _confidences[i], "", np.array([0, 0], dtype=np.uint16), class_confidences)
            )
            detected_locations_list.append(detection_location)
            altitudes.append(_altitudes[i])

        # Don't use this tracker if there is no valid detection
        if len(detection_list) == 0:
            log.warning(f"Skipping detection '{name}'...")
            return None

        status = TrackerStatus.TO_BE_INVESTIGATED
        if max([dt.confidence for dt in detection_list if dt is not None]) >= inspection_confidence:
            status = TrackerStatus.ACCEPTED

        assert valid_location is not None
        assert len(detected_locations_list) == len(detection_list) == len(altitudes)

        return cls(name, status, valid_location, detection_list, altitudes, detected_locations_list)


class GPSTracker:
    def __init__(
        self,
        field: Field,
        camera_parameters: CameraParameters,
        distance_threshold: dict[float, float] | float = 0.35,
        inspection_confidence: float = 0.4,
        max_inspection_altitude: float = 12.0,
        use_adaptive: bool = True,
    ) -> None:
        self.field = field
        self.camera_parameters = camera_parameters
        self.distance_threshold = distance_threshold
        self.inspection_confidence = inspection_confidence
        self.max_inspection_altitude = max_inspection_altitude
        self.use_adaptive = use_adaptive

        self._trackers: dict[str, GPSTrack] = {}

    def get_locations(self, status_mask: TrackerStatus = TrackerStatus(~0)) -> list[Location]:
        return [trk.to_location() for trk in self._trackers.values() if trk.status & status_mask]

    def track(self, waypoint: Waypoint, detections: list[Detection]) -> None:
        visible_trackers = self.get_visible_trackers(waypoint)

        if len(detections) == 0:
            for missed_tracker in visible_trackers:
                self.reject_tracker(missed_tracker, waypoint)

            return

        detections_locations = georeference_detections(detections, waypoint, self.camera_parameters)
        distance_threshold = self.get_distance_threshold(waypoint.location.altitude)

        # Filter out locations for some detections when specified (GCP for example)
        if len(self.field.locations_to_ignore) > 0:
            distance_matrix = create_distance_matrix(detections_locations, self.field.locations_to_ignore, distance_threshold)
            for i in sorted(np.nonzero(distance_matrix.min(axis=1) <= distance_threshold)[0], reverse=True):
                log.info(f"Removing detection (x={detections[i].xywh[0]},y={detections[i].xywh[0]}) because it's on the ignore list...")
                del detections[i]
                del detections_locations[i]

        found_tracker_names = []

        # Match the detections with the detections that should be visible. Detections that are not visible and
        # are captured with a lower altitude are probably FP's.
        distance_matrix = create_distance_matrix(detections_locations, [vt.location for vt in visible_trackers], distance_threshold)

        dt_i, trk_j = linear_sum_assignment(distance_matrix)
        for i, j in zip(dt_i, trk_j):
            if float(distance_matrix[i, j]) <= distance_threshold:
                self.update_tracker(visible_trackers[j], detections[i], detections_locations[i], waypoint.location.altitude)
                found_tracker_names.append(visible_trackers[j].name)
            else:
                name = f"detection_{len(self._trackers)}"
                self.create_tracker(name, detections[i], detections_locations[i], waypoint.location.altitude)
                found_tracker_names.append(name)

                self.reject_tracker(visible_trackers[j], waypoint)

        for i in range(len(detections_locations)):
            if i not in dt_i:
                name = f"detection_{len(self._trackers)}"
                self.create_tracker(name, detections[i], detections_locations[i], waypoint.location.altitude)
                found_tracker_names.append(name)

        for j in range(len(visible_trackers)):
            if j not in trk_j:
                self.reject_tracker(visible_trackers[j], waypoint)

    def get_visible_trackers(self, waypoint: Waypoint) -> list[GPSTrack]:
        fov_m = get_field_of_view_m(waypoint.location.altitude, self.camera_parameters)

        # Increase the size of the FoV with the distance threshold to also include the area just beyond the image border
        fov_m += 2 * self.get_distance_threshold(waypoint.location.altitude)
        fov_polgon = get_fov_polygon(waypoint, fov_m)
        return [
            td for td in self._trackers.values() if fov_polgon.contains(td.location.to_point()) and td.status is not TrackerStatus.REJECTED
        ]

    def add_tracker(self, tracker: GPSTrack) -> None:
        self._trackers[tracker.name] = tracker

    def create_tracker(self, name: str, detection: Detection, detection_location: Location, altitude: float) -> None:
        status = TrackerStatus.TO_BE_INVESTIGATED if self.use_adaptive else TrackerStatus.INVESTIGATED

        if detection.confidence >= self.inspection_confidence:
            status = TrackerStatus.ACCEPTED
        elif altitude <= self.max_inspection_altitude:
            status = TrackerStatus.INVESTIGATED

        self._trackers[name] = GPSTrack(
            name,
            status,
            detection_location,
            [detection],
            [altitude],
            [detection_location],
        )

        log.info(
            f"Found new object '{name}' (conf={detection.confidence:.3f},gps={detection_location.gps_coordinate_lon_lat}) with status {status.name}..."
        )

    def reject_tracker(self, tracker: GPSTrack, waypoint: Waypoint) -> None:
        fov_m = get_field_of_view_m(waypoint.location.altitude, self.camera_parameters)
        fov_polygon = get_fov_polygon(waypoint, fov_m)

        if (distance := fov_polygon.exterior.distance(tracker.location.to_point())) < self.get_distance_threshold(
            waypoint.location.altitude
        ):
            log.info(
                f"Ignore missing object '{tracker.name}' with status {tracker.status.name} because it is too close ({distance:.3f}m) to the border of the image..."
            )
            return

        if waypoint.location.altitude < min(self._trackers[tracker.name].altitudes) and not tracker.status == TrackerStatus.ACCEPTED:
            self._trackers[tracker.name].status = TrackerStatus.REJECTED
        else:
            # Not rejecting it here, since it may be on the edge of an image
            self._trackers[tracker.name].altitudes.append(waypoint.location.altitude)
            self._trackers[tracker.name].detections.append(None)
            self._trackers[tracker.name].detected_locations.append(None)

        log.warning(
            f"Did not found object '{tracker.name}' (best conf="
            f"{max([dt.confidence for dt in self._trackers[tracker.name].detections if dt is not None]):.3f})"
            f" with status {self._trackers[tracker.name].status.name} while at should be visible..."
        )

    def update_tracker(self, tracker: GPSTrack, detection: Detection, detected_location: Location, altitude: float) -> None:
        if detection.confidence >= self.inspection_confidence:
            self._trackers[tracker.name].status = TrackerStatus.ACCEPTED
        elif altitude < min(self._trackers[tracker.name].altitudes):
            self._trackers[tracker.name].status = TrackerStatus.INVESTIGATED

        if detection.confidence > max([dt.confidence for dt in self._trackers[tracker.name].detections if dt is not None]):
            self._trackers[tracker.name].location.gps_coordinate_lon_lat = detected_location.gps_coordinate_lon_lat

        self._trackers[tracker.name].detections.append(detection)
        self._trackers[tracker.name].altitudes.append(altitude)
        self._trackers[tracker.name].detected_locations.append(detected_location)

        detected_locations = [loc for loc in self._trackers[tracker.name].detected_locations if loc is not None]
        detected_distances = [loc.get_distance(detected_locations[-1]) for loc in detected_locations[:-1]]

        log.info(
            f"Re-recognizing object '{tracker.name}' (conf={detection.confidence:.3f},"
            f" dist={np.mean(detected_distances):.3f}±{np.std(detected_distances):.3f}m) with status"
            f" {self._trackers[tracker.name].status.name}. Updated properties..."
        )

    def get_distance_threshold(self, altitude: float) -> float:
        """
        Function to calculate the height based distance threshold. If distance threshold is a single value, returns the single
        value. If it is a dictionary with multiple values, it interpolates between the closest two values.

        :param altitude: Altitude for the distance threshold.
        :returns: Distance threshold.
        """
        if isinstance(self.distance_threshold, float):
            return self.distance_threshold

        if altitude in self.distance_threshold:
            return self.distance_threshold[altitude]

        altitudes = np.array(sorted(self.distance_threshold.keys()), dtype=np.float32)
        thresholds = np.array([self.distance_threshold[alt] for alt in altitudes], dtype=np.float32)

        lower_idx = np.searchsorted(altitudes, altitude) - 1
        upper_idx = lower_idx + 1

        if lower_idx < 0:
            return thresholds[0]  # type: ignore[no-any-return]
        elif upper_idx >= len(altitudes):
            return thresholds[-1]  # type: ignore[no-any-return]

        return np.interp(altitude, [altitudes[lower_idx], altitudes[upper_idx]], [thresholds[lower_idx], thresholds[upper_idx]]).item()  # type: ignore[no-any-return]
