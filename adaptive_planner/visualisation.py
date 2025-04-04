from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, cast

from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

from moviepy.editor import VideoFileClip
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tabulate import tabulate
from tilemapbase import Extent, Plotter, init, project, to_lonlat
from tilemapbase.tiles import Tiles, build_OSM
from tqdm.auto import tqdm

from adaptive_planner.location import Location
from adaptive_planner.utils import calculate_total_distance, parallel_execute

if TYPE_CHECKING:
    from typing import Any, Literal

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray

    from tqdm.std import tqdm as _TQDM_T

    from adaptive_planner.evaluation import EvaluationResult


init(create=True)

COLOR = "#c8586c"
BASELINE_COLOR = "#6699CC"
DISTANCE_COLOR = "#f5ba98"


class VisualisationException(Exception):
    pass


def save_fig(path: str | Path, fig: Figure, **kwargs: Any) -> None:
    if not isinstance(path, Path):
        path = Path(path)

    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)

    fig.savefig(path, **kwargs)


def save_subfig(path: str | Path, fig: Figure, ax: Axes, **kwargs: Any) -> None:
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    save_fig(path, fig, bbox_inches=extent.expanded(1.35, 1.35), **kwargs)


def save_part_fig(path: Path | str, fig: Figure, xywh_fraction: tuple[float, float, float, float], **kwargs: Any) -> None:
    left = xywh_fraction[0] * fig.get_size_inches()[0]
    bottom = xywh_fraction[1] * fig.get_size_inches()[1]
    width = (xywh_fraction[2] - xywh_fraction[0]) * fig.get_size_inches()[0]
    height = (xywh_fraction[3] - xywh_fraction[1]) * fig.get_size_inches()[1]
    save_fig(path, fig, bbox_inches=Bbox.from_bounds(left, bottom, width, height), **kwargs)


def show_image(img: NDArray[np.uint8], figsize: tuple[int, int] = (4, 3)) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(img[:, :, ::-1], aspect="auto")
    plt.axis("off")
    plt.show()


def get_tiles(tile_type: Literal["OSM", "ArcGis"]) -> Tiles:
    if tile_type == "OSM":
        return build_OSM()
    elif tile_type == "ArcGis":
        ARCGIS_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        return Tiles(ARCGIS_URL, "satellite")
    else:
        raise NotImplementedError(f"Tile type {tile_type} is not implemented!")


def plot_img_with_zoom_section(image_path: Path, x1: float, x2: float, y1: float, y2: float, ax: Axes | None = None) -> None:
    img = np.asarray(Image.open(image_path))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    extent = (-3, 4, -4, 3)

    ax.imshow(img, extent=extent, origin="lower", aspect="auto")

    axins = ax.inset_axes((0.5, 0.5, 0.47, 0.47), xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    axins.imshow(img, extent=extent, origin="lower")
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    for spine in axins.spines.values():
        spine.set_color("white")

    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=1.0, alpha=1.0)

    _remove_all_axes(ax)


def plot_markers(
    markers: list[Location],
    class_to_color: dict[str, tuple[int, int, int]],
    ax: Axes | None = None,
    legend_loc: str | None = "upper right",
    marker_size: float = 0.02,
    scalebar_length: float | None = None,
    extent: Extent | None = None,
) -> None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

    marker_locations = np.vstack([m.gps_coordinate_lon_lat for m in markers])

    if extent is None:
        expand = 0.0005
        extent = Extent.from_lonlat(
            marker_locations[:, 0].min() - expand,
            marker_locations[:, 0].max() + expand,
            marker_locations[:, 1].min() - expand,
            marker_locations[:, 1].max() + expand,
        )

    tiles = get_tiles("ArcGis")

    plotter = Plotter(extent, tiles, height=600)
    plotter.plot(ax, tiles, alpha=0.8)

    _add_markers_to_plot(markers, class_to_color, ax, marker_size=marker_size)

    # Custom legend
    legend_handles = []
    for name, color in class_to_color.items():
        legend_handles.append(Patch(color=(np.array(color, dtype=np.float32) / 255).tolist(), label=name))

    if legend_loc is not None:
        ax.legend(handles=legend_handles, loc=legend_loc)

    if scalebar_length is not None:
        _add_scalebar(scalebar_length, extent, ax)

    _remove_all_axes(ax)


def create_table(
    evaluations: dict[str, EvaluationResult],
    baseline_evaluation: dict[str, EvaluationResult] | None,
    flight_paths: dict[str, list[Location]],
    baseline_flight_paths: dict[str, list[Location]] | None,
    keys: list[str],
    key_name: str = "key",
    class_agnostic: bool = False,
) -> str:
    header = [key_name, "Total flight distance [m]", "F1-score", "Precision", "Recall"]
    rows = []

    flight_path_lengths = parallel_execute(flight_paths, calculate_total_distance)

    if baseline_flight_paths is not None:
        baseline_flight_path_lengths = parallel_execute(baseline_flight_paths, calculate_total_distance)

    def _create_row(name: str, distance: float, result: EvaluationResult) -> list[str | float]:
        return [
            name,
            distance,
            result.f1,
            result.precision,
            result.recall,
        ]

    for k in keys:
        if baseline_evaluation is not None and baseline_flight_paths is not None:
            if k in baseline_evaluation.keys():
                _baseline_key = k
                _baseline_evaluation = baseline_evaluation[k].make_class_agnostic() if class_agnostic else baseline_evaluation[k]
            else:
                for _baseline_key in baseline_evaluation.keys():
                    if k.endswith(_baseline_key):
                        _baseline_evaluation = (
                            baseline_evaluation[_baseline_key].make_class_agnostic()
                            if class_agnostic
                            else baseline_evaluation[_baseline_key]
                        )
                        break
                else:
                    raise RuntimeError(f"Cannot find baseline for '{k}'!")

            rows.append(_create_row(f"baseline {_baseline_key}", baseline_flight_path_lengths[_baseline_key], _baseline_evaluation))

        _evaluation = evaluations[k].make_class_agnostic() if class_agnostic else evaluations[k]

        rows.append(_create_row(k, flight_path_lengths[k], _evaluation))

    return tabulate(rows, headers=header, tablefmt="html", floatfmt=".3f")


def plot_metrics_with_path_length(
    results: dict[str, EvaluationResult],
    baseline_results: dict[str, EvaluationResult],
    flight_paths: dict[str, list[Location]],
    baseline_flight_paths: dict[str, list[Location]],
    metric: str,
    key_value_map: dict[str, int | float],
    ax_left: Axes | None = None,
    ax_right: Axes | None = None,
    class_agnostic: bool = False,
    ylim_left: tuple[float, float] = (0.0, 1.0),
    ylim_right: tuple[int, int] = (0, 105),
    title: str | None = None,
    xlabel: str = "x",
    ylabel_left: str = "Metric",
    ylabel_right: str = "Normalized flight path length [%]",
    legend_loc: str | None = None,
) -> None:
    flight_path_lengths = parallel_execute(flight_paths, calculate_total_distance)
    baseline_flight_path_lengths = parallel_execute(baseline_flight_paths, calculate_total_distance)

    def _group_metrics(results: dict[str, EvaluationResult], group_keys: list[str], metric: str) -> dict[str, list[float]]:
        values = defaultdict(list)

        for group_key in group_keys:
            for evaluation_key, result in results.items():
                if evaluation_key.startswith(group_key):
                    if class_agnostic:
                        result = result.make_class_agnostic()
                    values[group_key].append(getattr(result, metric))

        return dict(values)

    def _group_and_normalize_flight_path_length(keys: list[str], baseline_keys: list[str], group_keys: list[str]) -> dict[str, list[float]]:
        values = defaultdict(list)

        for group_key in group_keys:
            for k in keys:
                if k.startswith(group_key):
                    for bk in baseline_keys:
                        if k.endswith(bk):
                            base_distance = baseline_flight_path_lengths[bk]
                            break
                    else:
                        raise RuntimeError(f"Cannot find baseline for '{group_key}'!")

                    distance = flight_path_lengths[k]
                    normalized_distance = distance / base_distance

                    values[group_key].append(normalized_distance)

        return dict(values)

    if ax_left is None:
        fig, ax_left = plt.subplots(1, 1, figsize=(4, 4))

    if ax_right is None:
        ax_right = ax_left.twinx()

    x = np.empty(len(key_value_map))
    y = np.empty(len(key_value_map))
    std = np.empty(len(key_value_map))
    for i, (k, v) in enumerate(_group_metrics(results, list(key_value_map.keys()), "f1").items()):
        x[i] = key_value_map[k]
        y[i] = np.mean(v)
        std[i] = np.std(v)

    if class_agnostic:
        baseline_values = [getattr(r.make_class_agnostic(), metric) for r in baseline_results.values()]
    else:
        baseline_values = [getattr(r, metric) for r in baseline_results.values()]

    baseline_y = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)

    ax_left.plot(
        [min(key_value_map.values()), max(key_value_map.values())],
        [baseline_y, baseline_y],
        color=BASELINE_COLOR,
        label="Baseline",
    )
    ax_left.fill_between(
        [min(key_value_map.values()), max(key_value_map.values())],
        [baseline_y - baseline_std, baseline_y - baseline_std],
        [baseline_y + baseline_std, baseline_y + baseline_std],
        color=BASELINE_COLOR,
        alpha=0.4,
    )
    ax_left.plot(x, y, color=COLOR, label="Adaptive planner")
    ax_left.fill_between(x, y - std, y + std, color=COLOR, alpha=0.4)

    x = np.empty(len(key_value_map))
    y = np.empty(len(key_value_map))
    std = np.empty(len(key_value_map))
    for i, (k, v) in enumerate(
        _group_and_normalize_flight_path_length(
            list(flight_paths.keys()), list(baseline_flight_paths.keys()), list(key_value_map.keys())
        ).items()
    ):
        x[i] = key_value_map[k]
        y[i] = np.mean(v)
        std[i] = np.std(v)

    ax_right.plot(x, y * 100, "--", color=DISTANCE_COLOR, label="Distance (right axis)")
    ax_right.fill_between(x, (y - std) * 100, (y + std) * 100, color=DISTANCE_COLOR, alpha=0.4, hatch="+")

    if title is not None:
        ax_left.set_title(title)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(ylabel_left)
    ax_left.set_ylim(*ylim_left)
    ax_right.set_ylabel(ylabel_right)
    ax_right.set_ylim(*ylim_right)

    if legend_loc is not None:
        lines, labels = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_left.legend(lines + lines2, labels + labels2, loc=legend_loc)


def plot_confusion_matrix(name: str, eval: EvaluationResult, ax: Axes, class_agnostic: bool = False) -> None:
    if class_agnostic:
        eval = eval.make_class_agnostic()

    disp = ConfusionMatrixDisplay(eval.confusion_matrix, display_labels=eval.labels)

    disp.plot(ax=ax)
    ax.set_title(name)


def plot_mean_std_interval(
    x: list[float],
    data: NDArray[np.float32],
    ax: Axes,
    color: str = "#c8586c",
    alpha: float = 0.5,
    hatch: str | None = None,
    linestyle: str | None = None,
    label: str | None = None,
) -> None:
    ax.plot(x, np.mean(data, axis=1), color=color, linestyle=linestyle, label=label)
    ax.fill_between(
        x, np.mean(data, axis=1) - np.std(data, axis=1), np.mean(data, axis=1) + np.std(data, axis=1), alpha=alpha, color=color, hatch=hatch
    )


def plot_metric_path_length(
    x: list[float | int],
    metric: NDArray[np.float32],
    flight_path_length: NDArray[np.float32],
    ax_left: Axes | None = None,
    ax_right: Axes | None = None,
    title: str | None = None,
    xlabel: str = "x",
    ylabel_left: str = "Metric",
    ylabel_right: str = "Flight path length [m]",
    xlim: tuple[float | int, float | int] = (0.0, 1.0),
    ylim_left: tuple[float, float] = (0.0, 1.0),
    ylim_right: tuple[float, float] = (0.0, 1.0),
) -> None:
    if ax_left is None:
        fig, ax_left = plt.subplots(figsize=(4, 4))

    if ax_right is None:
        ax_right = ax_left.twinx()

    plot_mean_std_interval(x, metric, ax_left, color=BASELINE_COLOR, label="Detection performance")
    plot_mean_std_interval(x, flight_path_length, ax_right, color=BASELINE_COLOR, linestyle="--", hatch="x", label="Flight path length")

    if title is not None:
        ax_left.set_title(title)

    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(ylabel_left)
    ax_right.set_ylabel(ylabel_right)

    ax_left.set_xlim(*xlim)
    ax_left.set_ylim(*ylim_left)
    ax_right.set_ylim(*ylim_right)


def plot_flight_path(
    flight_path: list[Location],
    markers: list[Location] | None = None,
    marker_class_to_color: dict[str, tuple[int, int, int]] | None = None,
    interpolation_steps: int = 100,
    ax: Axes | None = None,
    name: str | None = None,
    fontsize: int = 5,
    linewidth: float = 0.5,
    scalebar_length: float | None = None,
    scalebar_args: dict[str, Any] = {},
    marker_args: dict[str, Any] = {},
    colorbar: bool = True,
    colorbar_ticks: list[float | int] | None = None,
    cmap_name: str = "viridis",
    start_annotation_offset: tuple[int, int] | None = None,
    end_annotation_offset: tuple[int, int] | None = None,
    min_altitude: float | None = None,
    max_altitude: float | None = None,
    extent: Extent | None = None,
) -> None:
    """
    Function to plot a flight path on top of satellite images.

    :param flight_path: Flight path to plot.
    :param markers: Optionally markers to plot in the figure
    :param interpolation_steps: Number of interpolation steps for the color of the plot. Higher number gives smoother colors.
    :param ax: Optional ax to plot on.
    :param name: Optional title of the plot.
    :param fontsize: Fontsize of the text in the plot.
    :param linewidth: Width of the flight path line in the plot.
    :param scalebar_length: Length of the scalebar in meters.
    :param scalebar_args: Optional arguments for scalebar.
    :param marker_args: Optional arguments for marker.
    :param colorbar: True whether to plot the colorbar, False otherwise.
    :param colorbar_ticks: Optional override the ticks on the colorbar.
    :param cmap_name: Name of the colormap.
    :param start_annotation_offset: Optional location of an annotation indicating the start of the flight path.
    :param end_annotation_offset: Optional location of an annotation indicating the end of the flight path.
    :param min_altitude: Optional minimum altitude to normalize the flight path color.
    :param max_altitude: Optional maximum altitude to normalize the flight path color.
    """
    if markers is not None and marker_class_to_color is None:
        raise VisualisationException("Both markers and marker_class_to_color have to be specified!")

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

    flight_path_arr = np.vstack([fp.gps_coordinate_lon_lat for fp in flight_path])

    if extent is None:
        expand = 0.0005
        extent = Extent.from_lonlat(
            flight_path_arr[:, 0].min() - expand,
            flight_path_arr[:, 0].max() + expand,
            flight_path_arr[:, 1].min() - expand,
            flight_path_arr[:, 1].max() + expand,
        )

    flight_path_projected = np.array(
        [[*project(lon, lat), alt] for lon, lat, alt in zip(flight_path_arr[:, 0], flight_path_arr[:, 1], flight_path_arr[:, 2])],
        dtype=np.float64,
    )

    tiles = get_tiles("ArcGis")

    plotter = Plotter(extent, tiles, height=600)
    plotter.plot(ax, tiles, alpha=0.8)

    if min_altitude is None:
        min_altitude = flight_path_projected[:, 2].min()

    if max_altitude is None:
        max_altitude = flight_path_projected[:, 2].max()

    norm = Normalize(vmin=min_altitude, vmax=max_altitude)
    cmap = plt.get_cmap(cmap_name)
    sm = ScalarMappable(norm=norm, cmap=cmap_name)
    sm.set_array([])

    if markers is not None:
        _add_markers_to_plot(markers, cast(dict[str, tuple[int, int, int]], marker_class_to_color), ax, **marker_args)

    for i in range(len(flight_path_projected) - 1):
        flight_path_interpolated = np.linspace(flight_path_projected[i, :], flight_path_projected[i + 1, :], interpolation_steps, axis=0)
        colors = cmap(norm(flight_path_interpolated[:, 2]))
        for j in range(interpolation_steps - 1):
            ax.plot(
                flight_path_interpolated[j : j + 2, 0],
                flight_path_interpolated[j : j + 2, 1],
                color=colors[j],
                linewidth=linewidth,
                zorder=2,
            )

    if start_annotation_offset is not None:
        ax.annotate(
            "Start",
            xy=tuple(flight_path_projected[0, :2]),
            xytext=start_annotation_offset,
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
            fontsize=fontsize,
        )

    if end_annotation_offset is not None:
        ax.annotate(
            "End",
            xy=tuple(flight_path_projected[-1, :2]),
            xytext=end_annotation_offset,
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->"),
            fontsize=fontsize,
        )

    if colorbar:
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label("Altitude [m]", fontsize=fontsize)

    if scalebar_length is not None:
        _add_scalebar(scalebar_length, extent, ax, **scalebar_args)

    if name is not None:
        ax.set_title(name)

    _remove_all_axes(ax)


def plot_animated_flight_path(
    flight_path: list[Location],
    interpolation_steps_m: float = 0.5,
    fontsize: int = 5,
    plane_min_zoom: float = 0.020,
    plane_max_zoom: float = 0.025,
    line_min_width: float = 0.35,
    line_max_width: float = 0.5,
    colorbar: bool = True,
) -> FuncAnimation:
    flight_path_arr = np.vstack([fp.gps_coordinate_lon_lat for fp in flight_path])

    expand = 0.0005
    extent = Extent.from_lonlat(
        flight_path_arr[:, 0].min() - expand,
        flight_path_arr[:, 0].max() + expand,
        flight_path_arr[:, 1].min() - expand,
        flight_path_arr[:, 1].max() + expand,
    )

    flight_path_projected = np.array(
        [[*project(lon, lat), alt] for lon, lat, alt in zip(flight_path_arr[:, 0], flight_path_arr[:, 1], flight_path_arr[:, 2])],
        dtype=np.float64,
    )

    tiles = get_tiles("ArcGis")

    _need_cbar = colorbar and np.unique(flight_path_projected[:, 2]).shape[0] > 1

    fig, ax = plt.subplots(figsize=(2.25, 1.5) if _need_cbar else (2, 1.5), dpi=600)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plotter = Plotter(extent, tiles, zoom=25)
    plotter.plot(ax, tiles, alpha=0.8)

    norm = Normalize(vmin=flight_path_projected[:, 2].min(), vmax=flight_path_projected[:, 2].max())
    # norm = Normalize(vmin=12.0, vmax=40.0)
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    lines: list[Line2D] = []
    zoom_levels: list[float] = []
    for i in range(len(flight_path_projected) - 1):
        interpolation_steps = round(flight_path[i].get_distance(flight_path[i + 1], use_3d=True) / interpolation_steps_m)
        flight_path_interpolated = np.linspace(flight_path_projected[i, :], flight_path_projected[i + 1, :], interpolation_steps, axis=0)
        colors = cmap(norm(flight_path_interpolated[:, 2]))
        for j in range(interpolation_steps - 1):
            # Normalize altitude when needed
            normalized_altitude = 1.0
            if flight_path_projected[:, 2].min() != flight_path_projected[:, 2].max():
                normalized_altitude = (flight_path_interpolated[j, 2] - flight_path_projected[:, 2].min()) / (
                    flight_path_projected[:, 2].max() - flight_path_projected[:, 2].min()
                )
            line_width = normalized_altitude * (line_max_width - line_min_width) + line_min_width
            zoom_level = normalized_altitude * (plane_max_zoom - plane_min_zoom) + plane_min_zoom

            (line,) = ax.plot(
                flight_path_interpolated[j : j + 2, 0],
                flight_path_interpolated[j : j + 2, 1],
                color=colors[j],
                linewidth=line_width,
            )

            lines.append(line)
            zoom_levels.append(zoom_level)

    icon_path = Path(__file__).parent / "plane.png"

    def get_rotated_image(line: Line2D, zoom: float) -> OffsetImage:
        angle = -np.arctan2(line.get_ydata()[1] - line.get_ydata()[0], line.get_xdata()[-1] - line.get_xdata()[0]) - np.pi / 2  # type: ignore[operator,index]
        image = Image.open(icon_path).rotate(np.rad2deg(angle)).convert("RGBA")
        return OffsetImage(image, zoom=zoom, zorder=10)  # type: ignore[arg-type]

    ann_box = AnnotationBbox(
        get_rotated_image(lines[0], zoom_levels[0]),
        (lines[0].get_xdata()[-1], lines[0].get_ydata()[-1]),  # type: ignore[index,arg-type]
        frameon=False,
        zorder=10,
    )
    ax.add_artist(ann_box)

    def init() -> list[Artist]:
        for line in lines:
            line.set_alpha(0)

        ann_box.set_visible(True)
        return lines + [ann_box]

    def update(frame: int) -> list[Artist]:
        if frame < len(lines):
            lines[frame].set_alpha(1)

            # Update plane figure
            ann_box.xybox = (lines[frame].get_xdata()[-1], lines[frame].get_ydata()[-1])  # type: ignore[index,assignment]
            ann_box.offsetbox = get_rotated_image(lines[frame], zoom_levels[frame])
        return lines + [ann_box]

    # Add colorbar, makes no sense when there is only a single height
    if _need_cbar:
        cbar = plt.colorbar(sm, ax=ax, shrink=fontsize / 10)
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks(np.linspace(flight_path_projected[:, 2].min(), flight_path_projected[:, 2].max(), 4).round().astype(int))
        cbar.set_label("Altitude [m]", fontsize=fontsize)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    animation = FuncAnimation(fig, update, frames=len(lines), init_func=init, blit=True, repeat=False)

    return animation


def save_animation_with_progress_bar(animation: FuncAnimation, filename: Path, fps: int = 10, **kwargs: Any) -> None:
    class ProgressBar:
        def __init__(self) -> None:
            self.progress_bar: _TQDM_T | None = None  # type:ignore [type-arg]

        def __call__(self, current_frame: int, total_frames: int) -> None:
            if self.progress_bar is None:
                self.progress_bar = tqdm(total=total_frames, desc="Saving animation", unit="frame")

            self.progress_bar.update()
            if current_frame == total_frames - 1:
                self.progress_bar.close()

    # We cannot directly create the .gif. Create first a .mp4 video, then convert it to .gif and
    # afterwards, delete the .mp4 video.
    _convert_to_gif = False
    if filename.suffix == ".gif":
        _convert_to_gif = True

        gif_filename = filename
        filename = filename.parent / (filename.stem + ".mp4")

    video_writer = FFMpegWriter(fps=fps)
    animation.save(
        filename,
        writer=video_writer,
        progress_callback=ProgressBar(),
        # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"},
        **kwargs,
    )

    if _convert_to_gif:
        video = VideoFileClip(str(filename))
        video.write_gif(str(gif_filename), fps=fps, program="ffmpeg", opt="nq")

        filename.unlink()  # Delete .mp4 file


def _get_frequent_color(img: NDArray[np.uint8], frequency_index: int = 0) -> NDArray[np.uint8]:
    reshaped_img = img.reshape(-1, img.shape[-1])
    unique_colors, counts = np.unique(reshaped_img, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    return unique_colors[sorted_indices[frequency_index]]  # type: ignore[no-any-return]


def _load_icon(icon_file: Path, color: tuple[int, ...] | NDArray[np.float32] | None = None, zoom: float = 0.02) -> OffsetImage:
    arr_img = plt.imread(icon_file)

    if color is not None:
        mask = np.all(arr_img == _get_frequent_color(arr_img, frequency_index=1), axis=-1)
        color = np.array([*color, 255], dtype=np.float32) if len(color) == 3 else np.array(color, dtype=np.float32)
        arr_img[mask] = np.array(color, dtype=np.float32) / 255

    return OffsetImage(arr_img, zoom=zoom)


def _remove_all_axes(ax: Axes) -> None:
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_markers_to_plot(
    markers: list[Location],
    marker_class_to_color: dict[str, tuple[int, int, int]],
    ax: Axes,
    marker_size: float = 0.02,
    **kwargs: Any,
) -> None:
    icon_path = Path(__file__).parent / "marker.png"
    icons = {k: _load_icon(icon_path, color=v, zoom=marker_size) for k, v in marker_class_to_color.items()}

    marker_locations = np.vstack([m.gps_coordinate_lon_lat for m in markers])
    marker_locations_projected = np.array(
        [[*project(lon, lat)] for lon, lat in zip(marker_locations[:, 0], marker_locations[:, 1])],
        dtype=np.float64,
    )

    for i in range(marker_locations_projected.shape[0]):
        imagebox = deepcopy(icons[markers[i].properties["class_name"]])
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox,
            (marker_locations_projected[i, 0], marker_locations_projected[i, 1]),
            xybox=(0.0, 4.5),
            xycoords="data",
            boxcoords="offset points",
            bboxprops={"edgecolor": "#FF000000", "facecolor": "#FF000000"},
            **kwargs,
        )
        ax.add_artist(ab)


def _add_scalebar(scalebar_length: float, extent: Extent, ax: Axes, lw: float = 0.75, fontsize: float = 6.0, **kwargs: Any) -> None:
    def _utm_to_plt(utm_coordinate: NDArray[np.float64]) -> tuple[float, float]:
        return cast(tuple[float, float], project(*Location.from_utm(utm_coordinate).gps_coordinate_lon_lat[:2]))

    bottom_left_utm = Location(np.array(to_lonlat(extent.xmax, extent.ymax), dtype=np.float64)).utm_coordinate
    left_utm = bottom_left_utm - (1.10 * scalebar_length, -0.10 * scalebar_length)
    right_utm = bottom_left_utm - (0.10 * scalebar_length, -0.10 * scalebar_length)
    text_utm = left_utm.copy() + (0.5 * scalebar_length, 1.5)

    ax.annotate(
        "",
        xy=(_utm_to_plt(right_utm)[0], _utm_to_plt(left_utm)[1]),  # Rounding error (?)
        xytext=_utm_to_plt(left_utm),
        arrowprops=dict(arrowstyle="<->", lw=lw, color="black", **kwargs),
    )
    ax.text(*_utm_to_plt(text_utm), f"{scalebar_length:.0f}m", ha="center", va="bottom", fontsize=fontsize)
