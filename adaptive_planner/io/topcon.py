from __future__ import annotations

from pathlib import Path

import numpy as np

from pyproj import CRS, Transformer

from adaptive_planner.location import WGS_84, Location


def read_topcon_data(topcon_file: Path, crs: str = "epsg:28992", use_height: bool = True) -> list[Location]:
    markers = []

    with topcon_file.open("r") as filereader:
        header = next(filereader)
        is_latlon = header.index("Lon(East)") > header.index("Lat(North)")

        for line in filereader.readlines():
            line_content = line.split(",")

            name = line_content[0]
            rd_x = line_content[2] if is_latlon else line_content[1]
            rd_y = line_content[1] if is_latlon else line_content[2]
            ht = line_content[3]

            if use_height:
                marker = Location.from_crs(np.array([rd_x, rd_y, ht], dtype=np.float64), CRS(crs).to_3d())
            else:
                marker = Location.from_crs(np.array([rd_x, rd_y], dtype=np.float64), CRS(crs))
            marker.properties["name"] = name
            marker.properties["class_name"] = name.split("_")[0]
            markers.append(marker)

    return markers


def write_topcon_data(markers: list[Location], output_file: Path, crs: str = "epsg:28992") -> None:
    transformer = Transformer.from_crs(WGS_84.to_3d(), CRS(crs).to_3d())

    # TODO: check format of files based on CRS
    if crs == "epsg:28992":
        output_lines = ["Header>> Delimiter(,) FileFormat(Name,Lon(East),Lat(North),Ht(G)) <<"]
    else:
        output_lines = ["Header>> Delimiter(,) FileFormat(Name,Lat(North),Lon(East),Ht(G)) <<"]  # Use default header

    for marker in markers:
        rd_x, rd_y, ht = transformer.transform(*marker.gps_coordinate_lat_lon)  # type: ignore[misc]
        output_lines.append(f"{marker.properties['name']},{rd_x},{rd_y},{ht}")
    output_file.write_text("\n".join(output_lines))
