from __future__ import annotations

from ast import literal_eval
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast
from zipfile import ZipFile

import cv2
import numpy as np

from lxml import etree as ET
from simplekml import AltitudeMode, Kml, LineString, Point, Style

from adaptive_planner.location import Location
from adaptive_planner.utils import img_from_str, img_to_str

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


class KMLParseException(Exception):
    pass


ICON_URL = [
    "http://maps.google.com/mapfiles/kml/paddle/blu-blank.png",
    "http://maps.google.com/mapfiles/kml/paddle/red-blank.png",
    "http://maps.google.com/mapfiles/kml/paddle/grn-blank.png",
    "http://maps.google.com/mapfiles/kml/paddle/orange-blank.png",
    "http://maps.google.com/mapfiles/kml/paddle/ltblu-blank.png",
    "http://maps.google.com/mapfiles/kml/paddle/wht-blank.png",
]


def write_gps_flightpath_to_kml(output_file: Path, waypoints: list[Location], name: str, kmz: bool = False) -> None:
    kml = Kml()

    style = Style()
    style.linestyle.color = "ff0000ff"
    style.linestyle.width = 4

    linestring = cast(LineString, kml.newlinestring(name=name))
    linestring.coords = [wp.gps_coordinate_lon_lat.tolist() for wp in waypoints]
    linestring.style = style
    linestring.altitudemode = AltitudeMode.relativetoground
    linestring.extrude = 0
    linestring.tessellate = 0

    kml.savekmz(output_file) if kmz else kml.save(output_file)


def write_gps_locations_to_kml(output_file: Path, locations: list[Location], kmz: bool = False) -> None:
    kml = Kml()
    styles: dict[str, Style] = {}
    for i, location in enumerate(locations):
        point = cast(Point, kml.newpoint(name=location.properties.get("name", f"location_{i}")))
        point.coords = [location.gps_coordinate_lon_lat.tolist()]

        for k, v in location.properties.items():
            if k == "name":
                continue
            elif k == "object_image":
                if kmz:
                    with NamedTemporaryFile(prefix=point.name + "_", suffix=".png", delete=False) as temp_file:
                        cv2.imwrite(temp_file.name, v[..., ::-1])

                    path = kml.addfile(temp_file.name)
                    point.description = '<img src="' + path + '" alt="picture" width="400" align="left" />'
                    v = path
                else:
                    v = img_to_str(v)

            v = ",".join(map(str, v)) if isinstance(v, list) else str(v)
            point.extendeddata.newdata(k, v)

        class_name = location.properties.get("best_class_name") or location.properties.get("class_name", "default")
        if class_name not in styles:
            style = Style()
            style.iconstyle.icon.href = ICON_URL[len(styles) % len(ICON_URL)] if class_name != "default" else None
            style.labelstyle.scale = 0.0
            styles[class_name] = style

        point.style = styles[class_name]

    kml.savekmz(output_file) if kmz else kml.save(output_file)


def get_namespace(root: ET._Element) -> str:
    return root.tag.split("}")[0].strip("{")


def get_element_value(element: ET._Element, element_name: str, namespaces: dict[str, str] | None = None) -> str:
    if (tag := element.find(element_name, namespaces=namespaces)) is None or (text := tag.text) is None:
        raise KMLParseException(f"Invalid or empty element {element_name}")
    return text


def read_kml_file(kml_file: Path) -> list[Location]:
    kmz = kml_file.suffix == ".kmz"
    if kmz:
        with ZipFile(kml_file, "r") as zf:
            with zf.open("doc.kml") as kml_file_buffer:
                kml_root = ET.fromstring(kml_file_buffer.read())
    else:
        with kml_file.open("rb") as kml_file_buffer:
            kml_root = ET.fromstring(kml_file_buffer.read())

    namespace = {"kml": get_namespace(kml_root)}

    locations = []
    for placemark in kml_root.findall(".//kml:Placemark", namespaces=namespace):
        placemark_locations: list[Location] = []
        if (point := placemark.find("kml:Point", namespaces=namespace)) is not None:
            coordinates = get_element_value(point, "kml:coordinates", namespaces=namespace)
            placemark_locations.append(Location(np.fromstring(coordinates, dtype=np.float64, sep=",")))
        elif (linestring := placemark.find("kml:LineString", namespaces=namespace)) is not None:
            coordinates = get_element_value(linestring, "kml:coordinates", namespaces=namespace)

            # Can be split by newline or by space
            if len(coordinates.split("\n")) > 1:
                for coordinate in coordinates.strip().split("\n"):
                    placemark_locations.append(Location(np.fromstring(coordinate, dtype=np.float64, sep=",")))
            else:
                for coordinate in coordinates.split(" "):
                    placemark_locations.append(Location(np.fromstring(coordinate, dtype=np.float64, sep=",")))
        else:
            raise NotImplementedError("KML file should contain a 'Point' or a 'LineString'!")

        if (name := placemark.find("kml:name", namespaces=namespace)) is None:
            raise KMLParseException(f"Cannot find 'name' in {placemark}!")

        extended_data: dict[str, Any] = {}
        extended_data_tag = placemark.find("kml:ExtendedData", namespaces=namespace)
        if extended_data_tag is not None:
            for data in extended_data_tag.findall("kml:Data", namespaces=namespace):
                if not data.attrib.has_key("name"):
                    raise KMLParseException(f"Extended data element {extended_data_tag} has no name attribute!")

                key = str(data.attrib["name"])
                value = get_element_value(data, "kml:value", namespaces=namespace)

                if key == "object_image" and kmz:
                    with ZipFile(kml_file, "r") as zf:
                        with zf.open(value) as img_buffer:
                            image_data = np.frombuffer(img_buffer.read(), np.uint8)
                            extended_data[key] = cv2.imdecode(image_data, cv2.IMREAD_COLOR)[..., ::-1]
                elif key == "object_image":
                    extended_data[key] = img_from_str(value)
                else:
                    try:
                        extended_data[key] = literal_eval(value)
                    except (ValueError, SyntaxError):
                        extended_data[key] = value

        for i, pl in enumerate(placemark_locations):
            pl.properties["name"] = name.text if len(placemark_locations) == 1 else f"{name.text}_{i}"
            pl.properties.update(extended_data)

        locations.extend(placemark_locations)
    return locations


if __name__ == "__main__":
    from pathlib import Path

    # kml_file = Path("experiments/number_of_objects_1/uniform_1_object_locations.kmz")
    kml_file = Path("planner_evaluation_results/localization_uncertainty/12m_clustered_1_very_poor/flight_path.kml")
    # kml_file = Path("planner_evaluation_results/number_of_objects/baseline_12m_0_uniform_1/flight_path.kml")

    print(read_kml_file(kml_file))
    print(len(read_kml_file(kml_file)))
