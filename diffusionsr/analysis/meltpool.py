"""Dataset-configured melt-pool and keyhole measurements.

The public :class:`MeltPoolMetrics` object operates on physical, unscaled
samples. A sample contains a leading channel axis followed by either two or
three spatial axes; a batch adds one leading batch axis. Spatial arrays are
converted internally to semantic canonical order: ``(depth, lateral)`` for
2-D data and ``(depth, width, length)`` for 3-D data.

Keyholes are represented as mouth-connected cavity regions. Their complete
cavity--liquid interface is retained, so folded or re-entrant walls are not
reduced to one depth per lateral coordinate. Scalar quantities such as depth
and leading-wall angle are summaries derived from that topology.

This module intentionally does not manage files, model inference, unscaling,
plotting, or report generation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


AxisName = Literal["depth", "width", "length"]
RegionBasis = Literal["temperature", "liquid"]
KeyholeMethod = Literal["auto", "material_flood", "liquid_cap_fill"]
LeadingDirection = Literal["min_length", "max_length"]
MetricValue = float | int | bool | NDArray[np.generic]
MetricFunction = Callable[["MeltPoolContext"], float | int | bool]


class MetricUnavailableError(ValueError):
    """Raised when a requested metric has no meaning for the sample contract."""


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Dataset conventions required to interpret an in-memory sample.

    ``spatial_axes`` describes source-array order after the channel axis. For
    example, ``("length", "depth")`` means a 2-D sample has shape
    ``(channels, length, depth)``.

    ``voxel_size_um`` is always in semantic canonical order, independent of
    source-array order. Canonical order is ``(depth, width, length)`` with
    absent axes omitted. Thus a depth--length slice uses ``(depth, length)``.

    ``depth_origin_um`` is the coordinate of the first depth-axis voxel face.
    ``plate_height_um`` must align with a depth voxel face for cap-and-fill
    segmentation. Set ``depth_positive_into_plate=True`` when array coordinates
    increase down into the plate; otherwise they increase out of the plate.
    """

    field_names: tuple[str, ...]
    spatial_axes: tuple[AxisName, ...]
    voxel_size_um: tuple[float, ...]
    plate_height_um: float
    n_steps: int = 1
    time_index: int = -1
    depth_origin_um: float = 0.0
    depth_positive_into_plate: bool = False

    def __post_init__(self) -> None:
        if not self.field_names or len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be nonempty and unique.")
        spatial_dims = len(self.spatial_axes)
        if spatial_dims not in (2, 3):
            raise ValueError(
                "spatial_axes must describe either two or three spatial dimensions."
            )
        valid_axes = {"depth", "width", "length"}
        unknown_axes = set(self.spatial_axes) - valid_axes
        if unknown_axes:
            raise ValueError(f"Unknown spatial axes: {sorted(unknown_axes)}.")
        if len(set(self.spatial_axes)) != spatial_dims:
            raise ValueError("spatial_axes must be unique.")
        if "depth" not in self.spatial_axes:
            raise ValueError("Melt-pool analysis requires a physical depth axis.")
        if spatial_dims == 3 and set(self.spatial_axes) != valid_axes:
            raise ValueError("A 3-D sample must contain depth, width, and length.")
        if len(self.voxel_size_um) != spatial_dims:
            raise ValueError(
                "voxel_size_um must contain one value per canonical spatial axis."
            )
        if not all(
            np.isfinite(value) and float(value) > 0.0
            for value in self.voxel_size_um
        ):
            raise ValueError("voxel_size_um values must be positive and finite.")
        if int(self.n_steps) < 1:
            raise ValueError("n_steps must be at least one.")
        if not np.isfinite(self.plate_height_um):
            raise ValueError("plate_height_um must be finite.")
        if not np.isfinite(self.depth_origin_um):
            raise ValueError("depth_origin_um must be finite.")
        _resolve_time_index(self.time_index, self.n_steps)

    @property
    def spatial_dims(self) -> int:
        return len(self.spatial_axes)

    @property
    def canonical_axes(self) -> tuple[AxisName, ...]:
        """Present axes ordered as depth, width, then length."""

        return tuple(
            axis
            for axis in ("depth", "width", "length")
            if axis in self.spatial_axes
        )

    @property
    def canonical_voxel_size_um(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.voxel_size_um)

    def canonical_axis_index(self, axis: AxisName) -> int:
        if axis not in self.canonical_axes:
            raise MetricUnavailableError(
                f"Axis {axis!r} is unavailable; sample axes are "
                f"{self.spatial_axes!r}."
            )
        return self.canonical_axes.index(axis)

    def spacing_um(self, axis: AxisName) -> float:
        return self.canonical_voxel_size_um[self.canonical_axis_index(axis)]

    def depth_face_index(self) -> int:
        """Return the integer voxel-face index of the physical plate plane."""

        face = (
            float(self.plate_height_um) - float(self.depth_origin_um)
        ) / self.spacing_um("depth")
        rounded = int(round(face))
        if not np.isclose(face, rounded, rtol=0.0, atol=1e-6):
            raise ValueError(
                "plate_height_um must align with a depth voxel face; "
                f"the resolved face index was {face:.8g}."
            )
        return rounded

    def depth_coordinates_um(self, depth_indices: ArrayLike) -> NDArray[np.float64]:
        """Convert fractional voxel-center indices to physical coordinates."""

        indices = np.asarray(depth_indices, dtype=float)
        return np.asarray(
            float(self.depth_origin_um)
            + (indices + 0.5) * self.spacing_um("depth"),
            dtype=float,
        )

    def depth_below_plate_um(self, z_um: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(z_um, dtype=float)
        if self.depth_positive_into_plate:
            return np.asarray(z - float(self.plate_height_um), dtype=float)
        return np.asarray(float(self.plate_height_um) - z, dtype=float)

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        *,
        spatial_axes: tuple[AxisName, ...],
        voxel_size_um: float | Sequence[float],
        plate_height_um: float,
        depth_origin_um: float = 0.0,
        depth_positive_into_plate: bool = False,
        time_index: int = -1,
    ) -> "DatasetSpec":
        spacing = _normalize_voxel_size(
            voxel_size_um,
            spatial_dims=len(spatial_axes),
        )
        return cls(
            field_names=tuple(str(name) for name in dataset.field_names),
            n_steps=int(getattr(dataset, "n_steps", 1)),
            time_index=int(time_index),
            spatial_axes=spatial_axes,
            voxel_size_um=spacing,
            plate_height_um=float(plate_height_um),
            depth_origin_um=float(depth_origin_um),
            depth_positive_into_plate=bool(depth_positive_into_plate),
        )


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Rules used to obtain temperature- and liquid-defined regions."""

    melt_temperature_k: float = 1710.0
    liquid_field: str = "liqlabel"
    liquid_threshold: float = 0.5
    keep_largest_component: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.melt_temperature_k):
            raise ValueError("melt_temperature_k must be finite.")
        if not np.isfinite(self.liquid_threshold):
            raise ValueError("liquid_threshold must be finite.")


@dataclass(frozen=True, slots=True)
class KeyholeSpec:
    """Topology, contour, and wall-angle controls for keyhole analysis."""

    method: KeyholeMethod = "auto"
    material_field: str | None = None
    material_threshold: float = 0.5
    contour_level: float | None = None
    closing_radius_um: float = 0.0
    minimum_cavity_size: int = 1
    center_width_index: int | None = None
    leading_direction: LeadingDirection = "max_length"
    wall_angle_depth_range_um: tuple[float, float] | None = None
    minimum_wall_angle_points: int = 5

    def __post_init__(self) -> None:
        if self.method not in {"auto", "material_flood", "liquid_cap_fill"}:
            raise ValueError(f"Unknown keyhole method {self.method!r}.")
        if not np.isfinite(self.material_threshold):
            raise ValueError("material_threshold must be finite.")
        if self.contour_level is not None and not np.isfinite(self.contour_level):
            raise ValueError("contour_level must be finite when provided.")
        if not np.isfinite(self.closing_radius_um) or self.closing_radius_um < 0.0:
            raise ValueError("closing_radius_um must be finite and nonnegative.")
        if int(self.minimum_cavity_size) < 1:
            raise ValueError("minimum_cavity_size must be at least one.")
        if int(self.minimum_wall_angle_points) < 2:
            raise ValueError("minimum_wall_angle_points must be at least two.")
        if self.leading_direction not in {"min_length", "max_length"}:
            raise ValueError(f"Unknown leading_direction {self.leading_direction!r}.")
        if self.wall_angle_depth_range_um is not None:
            lower, upper = self.wall_angle_depth_range_um
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower < 0.0
                or upper <= lower
            ):
                raise ValueError(
                    "wall_angle_depth_range_um must be a finite increasing "
                    "interval below the plate."
                )


@dataclass(frozen=True, slots=True)
class KeyholeTopology:
    """Mouth-connected cavity, pores, and complete cavity--liquid interface."""

    cavity_mask: NDArray[np.bool_]
    pore_labels: NDArray[np.int32]
    liquid_wall_mask: NDArray[np.bool_]
    gas_wall_mask: NDArray[np.bool_]
    mouth_mask: NDArray[np.bool_]
    segmentation_method: str


@dataclass(frozen=True, slots=True)
class KeyholeSurface2D:
    """Ordered inner-wall contour segments in a depth--length plane."""

    topology: KeyholeTopology
    contour_segments_xz_um: tuple[NDArray[np.float64], ...]
    mouth_points_xz_um: NDArray[np.float64]
    width_index: int | None


@dataclass(frozen=True, slots=True)
class KeyholeSurface3D:
    """Optional mesh representation of a 3-D keyhole interface."""

    topology: KeyholeTopology
    vertices_xyz_um: NDArray[np.float64]
    faces: NDArray[np.int64]
    wall_face_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class LeadingWallAngle:
    """Leading-wall fit and enough diagnostics to audit the scalar angle."""

    angle_deg: float
    fit_points_xz_um: NDArray[np.float64]
    width_index: int | None
    valid: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MetricResult(Mapping[str, MetricValue]):
    """Mapping with attribute access for scalar or batched metric values."""

    data: Mapping[str, MetricValue]
    is_batch: bool

    def __getitem__(self, key: str) -> MetricValue:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name: str) -> MetricValue:
        try:
            return self.data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to_dict(self) -> dict[str, MetricValue]:
        return dict(self.data)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Scalar metric plus its dimensional and field applicability contract."""

    name: str
    function: MetricFunction
    required_axes: frozenset[AxisName]
    spatial_dims: frozenset[int]
    required_fields: frozenset[str] = frozenset()
    unavailable_hint: str | None = None


@dataclass(slots=True)
class MeltPoolContext:
    """One sample plus lazily cached fields, masks, topology, and surfaces."""

    sample: NDArray[np.generic]
    dataset: DatasetSpec
    regions: RegionSpec
    keyholes: KeyholeSpec
    basis: RegionBasis
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def field_volume(self, field_name: str) -> NDArray[np.generic]:
        """Return a field as a view in canonical semantic-axis order."""

        key = f"field:{field_name}"
        if key not in self._cache:
            channel = _field_channel_index(
                self.dataset.field_names,
                field_name,
                self.dataset.time_index,
                self.dataset.n_steps,
            )
            source_volume = self.sample[channel]
            source_positions = {
                name: index for index, name in enumerate(self.dataset.spatial_axes)
            }
            order = tuple(
                source_positions[name] for name in self.dataset.canonical_axes
            )
            self._cache[key] = np.transpose(source_volume, order)
        return self._cache[key]

    def raw_mask(self, basis: RegionBasis | None = None) -> NDArray[np.bool_]:
        """Return an unfiltered threshold mask in canonical order."""

        selected = basis or self.basis
        key = f"raw_mask:{selected}"
        if key not in self._cache:
            if selected == "liquid":
                volume = self.field_volume(self.regions.liquid_field)
                mask = np.isfinite(volume) & (
                    volume > float(self.regions.liquid_threshold)
                )
            elif selected == "temperature":
                volume = self.field_volume("temperature")
                mask = np.isfinite(volume) & (
                    volume >= float(self.regions.melt_temperature_k)
                )
            else:
                raise ValueError(f"Unknown region basis {selected!r}.")
            self._cache[key] = np.asarray(mask, dtype=bool)
        return self._cache[key]

    def mask(self, basis: RegionBasis | None = None) -> NDArray[np.bool_]:
        selected = basis or self.basis
        key = f"mask:{selected}"
        if key not in self._cache:
            raw = self.raw_mask(selected)
            self._cache[key] = (
                _largest_component(raw)
                if self.regions.keep_largest_component
                else raw
            )
        return self._cache[key]

    def keyhole_topology(self) -> KeyholeTopology:
        """Return the cached mouth-connected cavity and full inner interface."""

        key = "keyhole_topology"
        if key not in self._cache:
            liquid = np.asarray(self.mask("liquid"), dtype=bool)
            if self.keyholes.closing_radius_um > 0.0:
                liquid = _physical_binary_closing(
                    liquid,
                    radius_um=float(self.keyholes.closing_radius_um),
                    spacing_um=self.dataset.canonical_voxel_size_um,
                )
            method = self.keyholes.method
            if method == "auto":
                material_available = (
                    self.keyholes.material_field is not None
                    and self.keyholes.material_field in self.dataset.field_names
                )
                method = "material_flood" if material_available else "liquid_cap_fill"
            if method == "material_flood":
                field_name = self.keyholes.material_field
                if field_name is None:
                    raise ValueError(
                        "material_flood requires KeyholeSpec.material_field."
                    )
                material = self.field_volume(field_name)
                material_mask = np.isfinite(material) & (
                    material > float(self.keyholes.material_threshold)
                )
                topology = _material_flood_topology(
                    material_mask=np.asarray(material_mask, dtype=bool),
                    liquid_mask=liquid,
                    dataset=self.dataset,
                    minimum_cavity_size=int(self.keyholes.minimum_cavity_size),
                )
            elif method == "liquid_cap_fill":
                topology = _liquid_cap_fill_topology(
                    liquid_mask=liquid,
                    dataset=self.dataset,
                    minimum_cavity_size=int(self.keyholes.minimum_cavity_size),
                )
            else:  # pragma: no cover
                raise ValueError(f"Unknown keyhole method {method!r}.")
            self._cache[key] = topology
        return self._cache[key]

    def center_width_index(self) -> int:
        if "width" not in self.dataset.spatial_axes:
            raise MetricUnavailableError(
                "A center-width index requires a physical width axis."
            )
        width_axis = self.dataset.canonical_axis_index("width")
        width_count = self.mask("liquid").shape[width_axis]
        requested = self.keyholes.center_width_index
        index = width_count // 2 if requested is None else int(requested)
        if index < 0 or index >= width_count:
            raise ValueError(
                f"center_width_index={index} is outside width size {width_count}."
            )
        return index

    def center_keyhole_surface(self) -> KeyholeSurface2D:
        """Intersect the segmented cavity interface with the central y plane."""

        key = "center_keyhole_surface"
        if key in self._cache:
            return self._cache[key]
        if "length" not in self.dataset.spatial_axes:
            raise MetricUnavailableError(
                "A leading-wall surface requires a physical length axis."
            )
        topology = self.keyhole_topology()
        liquid_field = self.field_volume(self.regions.liquid_field)
        if self.dataset.spatial_dims == 2:
            cavity_slice = topology.cavity_mask
            liquid_wall_slice = topology.liquid_wall_mask
            mouth_slice = topology.mouth_mask
            field_slice = liquid_field
            width_index = None
        else:
            width_index = self.center_width_index()
            cavity_slice = topology.cavity_mask[:, width_index, :]
            liquid_wall_slice = topology.liquid_wall_mask[:, width_index, :]
            mouth_slice = topology.mouth_mask[:, width_index, :]
            field_slice = liquid_field[:, width_index, :]
        level = (
            float(self.keyholes.contour_level)
            if self.keyholes.contour_level is not None
            else float(self.regions.liquid_threshold)
        )
        segments = _extract_wall_contours_xz_um(
            scalar_field_dl=np.asarray(field_slice),
            contour_level=level,
            cavity_mask_dl=np.asarray(cavity_slice, dtype=bool),
            liquid_wall_mask_dl=np.asarray(liquid_wall_slice, dtype=bool),
            dataset=self.dataset,
        )
        surface = KeyholeSurface2D(
            topology=topology,
            contour_segments_xz_um=segments,
            mouth_points_xz_um=_mask_points_xz_um(mouth_slice, self.dataset),
            width_index=width_index,
        )
        self._cache[key] = surface
        return surface

    def leading_wall_angle(self) -> LeadingWallAngle:
        """Fit the leading inner wall in the center-width depth--length slice."""

        key = "leading_wall_angle"
        if key in self._cache:
            return self._cache[key]
        surface = self.center_keyhole_surface()
        segment = _select_leading_wall_segment(
            surface.contour_segments_xz_um,
            surface.mouth_points_xz_um,
            leading_direction=self.keyholes.leading_direction,
        )
        if segment is None:
            result = LeadingWallAngle(
                angle_deg=np.nan,
                fit_points_xz_um=np.empty((0, 2), dtype=float),
                width_index=surface.width_index,
                valid=False,
                reason="No leading cavity--liquid wall intersects the selected slice.",
            )
            self._cache[key] = result
            return result
        fit_points = np.asarray(segment, dtype=float)
        depth_range = self.keyholes.wall_angle_depth_range_um
        if depth_range is not None:
            depths = self.dataset.depth_below_plate_um(fit_points[:, 1])
            lower, upper = depth_range
            fit_points = fit_points[(depths >= lower) & (depths <= upper)]
        minimum = int(self.keyholes.minimum_wall_angle_points)
        if fit_points.shape[0] < minimum:
            result = LeadingWallAngle(
                angle_deg=np.nan,
                fit_points_xz_um=fit_points,
                width_index=surface.width_index,
                valid=False,
                reason=(
                    f"Only {fit_points.shape[0]} contour points were available; "
                    f"at least {minimum} are required."
                ),
            )
        else:
            angle = _orthogonal_line_angle_deg(fit_points)
            result = LeadingWallAngle(
                angle_deg=angle,
                fit_points_xz_um=fit_points,
                width_index=surface.width_index,
                valid=bool(np.isfinite(angle)),
                reason=None if np.isfinite(angle) else "Degenerate wall fit.",
            )
        self._cache[key] = result
        return result


class MeltPoolMetrics:
    """Configured scalar metric engine for one sample or a leading batch."""

    _ALIASES = {
        "width": "width_um",
        "length": "length_um",
        "depth": "depth_um",
        "area": "area_um2",
        "volume": "volume_um3",
        "cells": "occupied_cells",
        "molten_voxels": "occupied_cells",
        "keyhole_depth": "keyhole_depth_um",
        "leading_wall_angle": "leading_wall_angle_deg",
    }

    def __init__(
        self,
        dataset: DatasetSpec,
        regions: RegionSpec | None = None,
        keyholes: KeyholeSpec | None = None,
    ):
        self.dataset = dataset
        self.regions = regions or RegionSpec()
        self.keyholes = keyholes or KeyholeSpec()
        liquid_field = self.regions.liquid_field
        both_dims = frozenset({2, 3})
        self._registry: dict[str, MetricDefinition] = {
            "width_um": MetricDefinition(
                "width_um", self._width_um, frozenset({"width"}), both_dims,
                unavailable_hint="No in-plane dimension is relabeled as width.",
            ),
            "length_um": MetricDefinition(
                "length_um", self._length_um, frozenset({"length"}), both_dims,
            ),
            "depth_um": MetricDefinition(
                "depth_um", self._depth_um, frozenset({"depth"}), both_dims,
            ),
            "area_um2": MetricDefinition(
                "area_um2", self._area_um2, frozenset({"depth"}), frozenset({2}),
                unavailable_hint=(
                    "area_um2 is a 2-D slice area; request volume_um3 for 3-D data."
                ),
            ),
            "volume_um3": MetricDefinition(
                "volume_um3", self._volume_um3,
                frozenset({"depth", "width", "length"}), frozenset({3}),
                unavailable_hint=(
                    "Volume is undefined for 2-D data; request area_um2. "
                    "No out-of-plane thickness is assumed."
                ),
            ),
            "occupied_cells": MetricDefinition(
                "occupied_cells", self._occupied_cells, frozenset(), both_dims,
            ),
            "has_melt_pool": MetricDefinition(
                "has_melt_pool", self._has_melt_pool, frozenset(), both_dims,
            ),
            "keyhole_depth_um": MetricDefinition(
                "keyhole_depth_um", self._keyhole_depth_um,
                frozenset({"depth"}), both_dims,
                required_fields=frozenset({liquid_field}),
            ),
            "keyhole_area_um2": MetricDefinition(
                "keyhole_area_um2", self._keyhole_area_um2,
                frozenset({"depth"}), frozenset({2}),
                required_fields=frozenset({liquid_field}),
                unavailable_hint="Request keyhole_volume_um3 for a 3-D cavity.",
            ),
            "keyhole_volume_um3": MetricDefinition(
                "keyhole_volume_um3", self._keyhole_volume_um3,
                frozenset({"depth", "width", "length"}), frozenset({3}),
                required_fields=frozenset({liquid_field}),
                unavailable_hint="Request keyhole_area_um2 for a 2-D cavity.",
            ),
            "leading_wall_angle_deg": MetricDefinition(
                "leading_wall_angle_deg", self._leading_wall_angle_deg,
                frozenset({"depth", "length"}), both_dims,
                required_fields=frozenset({liquid_field}),
                unavailable_hint=(
                    "Leading-wall angle requires a depth--length plane. In 3-D "
                    "it is measured from the full cavity at center width."
                ),
            ),
            "leading_wall_angle_valid": MetricDefinition(
                "leading_wall_angle_valid", self._leading_wall_angle_valid,
                frozenset({"depth", "length"}), both_dims,
                required_fields=frozenset({liquid_field}),
            ),
            "leading_wall_angle_points": MetricDefinition(
                "leading_wall_angle_points", self._leading_wall_angle_points,
                frozenset({"depth", "length"}), both_dims,
                required_fields=frozenset({liquid_field}),
            ),
        }

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        *,
        spatial_axes: tuple[AxisName, ...],
        voxel_size_um: float | Sequence[float],
        plate_height_um: float,
        depth_origin_um: float = 0.0,
        depth_positive_into_plate: bool = False,
        time_index: int = -1,
        regions: RegionSpec | None = None,
        keyholes: KeyholeSpec | None = None,
    ) -> "MeltPoolMetrics":
        spec = DatasetSpec.from_dataset(
            dataset,
            spatial_axes=spatial_axes,
            voxel_size_um=voxel_size_um,
            plate_height_um=plate_height_um,
            depth_origin_um=depth_origin_um,
            depth_positive_into_plate=depth_positive_into_plate,
            time_index=time_index,
        )
        return cls(spec, regions=regions, keyholes=keyholes)

    @property
    def all_metrics(self) -> tuple[str, ...]:
        return tuple(self._registry)

    @property
    def available_metrics(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, definition in self._registry.items()
            if self._unavailable_reason(definition) is None
        )

    def register_metric(
        self,
        name: str,
        function: MetricFunction,
        *,
        required_axes: Sequence[AxisName] = (),
        spatial_dims: Sequence[int] = (2, 3),
        required_fields: Sequence[str] = (),
        unavailable_hint: str | None = None,
        overwrite: bool = False,
    ) -> None:
        if not name or name in self._ALIASES:
            raise ValueError("Metric names must be nonempty canonical output names.")
        if name in self._registry and not overwrite:
            raise ValueError(f"Metric {name!r} is already registered.")
        self._registry[name] = MetricDefinition(
            name=name,
            function=function,
            required_axes=frozenset(required_axes),
            spatial_dims=frozenset(int(value) for value in spatial_dims),
            required_fields=frozenset(str(value) for value in required_fields),
            unavailable_hint=unavailable_hint,
        )

    def __call__(
        self,
        samples: ArrayLike,
        *,
        metrics: str | Sequence[str] | None = None,
        basis: RegionBasis = "temperature",
    ) -> MetricResult:
        array = _as_numpy(samples)
        self._validate_basis_field(basis)
        names = self._normalize_metric_names(metrics)
        expected_single_ndim = self.dataset.spatial_dims + 1
        if array.ndim == expected_single_ndim:
            return MetricResult(self._evaluate_one(array, names, basis), is_batch=False)
        if array.ndim == expected_single_ndim + 1:
            rows = [self._evaluate_one(sample, names, basis) for sample in array]
            columns = {
                name: np.asarray([row[name] for row in rows]) for name in names
            }
            return MetricResult(columns, is_batch=True)
        raise ValueError(
            f"Expected a single {self.dataset.spatial_dims}-D sample with "
            f"{expected_single_ndim} dimensions or a leading batch with "
            f"{expected_single_ndim + 1}; got {array.shape}."
        )

    def calc_width(
        self, samples: ArrayLike, *, basis: RegionBasis = "temperature"
    ) -> MetricValue:
        return self(samples, metrics="width_um", basis=basis)["width_um"]

    def calc_length(
        self, samples: ArrayLike, *, basis: RegionBasis = "temperature"
    ) -> MetricValue:
        return self(samples, metrics="length_um", basis=basis)["length_um"]

    def calc_depth(
        self, samples: ArrayLike, *, basis: RegionBasis = "temperature"
    ) -> MetricValue:
        return self(samples, metrics="depth_um", basis=basis)["depth_um"]

    def calc_keyhole_depth(self, samples: ArrayLike) -> MetricValue:
        return self(samples, metrics="keyhole_depth_um", basis="liquid")[
            "keyhole_depth_um"
        ]

    def calc_leading_wall_angle(self, samples: ArrayLike) -> MetricValue:
        return self(samples, metrics="leading_wall_angle_deg", basis="liquid")[
            "leading_wall_angle_deg"
        ]

    def _validate_basis_field(self, basis: RegionBasis) -> None:
        if basis not in ("temperature", "liquid"):
            raise ValueError("basis must be 'temperature' or 'liquid'.")
        field_name = "temperature" if basis == "temperature" else self.regions.liquid_field
        if field_name not in self.dataset.field_names:
            raise MetricUnavailableError(
                f"basis={basis!r} requires field {field_name!r}, but fields are "
                f"{self.dataset.field_names!r}."
            )

    def _default_metric_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if "width" in self.dataset.spatial_axes:
            names.append("width_um")
        if "length" in self.dataset.spatial_axes:
            names.append("length_um")
        names.extend(
            (
                "depth_um",
                "area_um2" if self.dataset.spatial_dims == 2 else "volume_um3",
                "occupied_cells",
                "has_melt_pool",
            )
        )
        if self.regions.liquid_field in self.dataset.field_names:
            names.append("keyhole_depth_um")
            if "length" in self.dataset.spatial_axes:
                names.extend(("leading_wall_angle_deg", "leading_wall_angle_valid"))
        return tuple(names)

    def _unavailable_reason(self, definition: MetricDefinition) -> str | None:
        if self.dataset.spatial_dims not in definition.spatial_dims:
            return (
                f"{definition.name!r} is not defined for "
                f"{self.dataset.spatial_dims}-D data."
            )
        missing_axes = definition.required_axes - set(self.dataset.spatial_axes)
        if missing_axes:
            return (
                f"{definition.name!r} requires axes {sorted(missing_axes)}, "
                f"but the sample contains {self.dataset.spatial_axes!r}."
            )
        missing_fields = definition.required_fields - set(self.dataset.field_names)
        if missing_fields:
            return (
                f"{definition.name!r} requires fields {sorted(missing_fields)}, "
                f"but the dataset contains {self.dataset.field_names!r}."
            )
        return None

    def _normalize_metric_names(
        self, metrics: str | Sequence[str] | None
    ) -> tuple[str, ...]:
        if metrics is None:
            requested = self._default_metric_names()
        elif isinstance(metrics, str):
            requested = (metrics,)
        else:
            requested = tuple(metrics)
        names = tuple(self._ALIASES.get(name, name) for name in requested)
        unknown = sorted(set(names) - set(self._registry))
        if unknown:
            raise ValueError(
                f"Unknown metrics {unknown}; known metrics are {self.all_metrics}."
            )
        unavailable: list[str] = []
        for name in names:
            definition = self._registry[name]
            reason = self._unavailable_reason(definition)
            if reason is not None:
                hint = (
                    f" {definition.unavailable_hint}"
                    if definition.unavailable_hint
                    else ""
                )
                unavailable.append(reason + hint)
        if unavailable:
            raise MetricUnavailableError(
                "Some requested metrics are unavailable:\n- "
                + "\n- ".join(unavailable)
            )
        return tuple(dict.fromkeys(names))

    def _evaluate_one(
        self,
        sample: NDArray[np.generic],
        names: Sequence[str],
        basis: RegionBasis,
    ) -> dict[str, float | int | bool]:
        expected_channels = len(self.dataset.field_names) * self.dataset.n_steps
        if sample.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels, got sample shape {sample.shape}."
            )
        context = MeltPoolContext(
            sample=sample,
            dataset=self.dataset,
            regions=self.regions,
            keyholes=self.keyholes,
            basis=basis,
        )
        return {name: self._registry[name].function(context) for name in names}

    @staticmethod
    def _has_melt_pool(context: MeltPoolContext) -> bool:
        return bool(context.mask().any())

    @staticmethod
    def _occupied_cells(context: MeltPoolContext) -> int:
        return int(context.mask().sum())

    @staticmethod
    def _width_um(context: MeltPoolContext) -> float:
        return _physical_extent(
            context.mask(),
            context.dataset.canonical_axis_index("width"),
            context.dataset.spacing_um("width"),
        )

    @staticmethod
    def _length_um(context: MeltPoolContext) -> float:
        return _physical_extent(
            context.mask(),
            context.dataset.canonical_axis_index("length"),
            context.dataset.spacing_um("length"),
        )

    @staticmethod
    def _depth_um(context: MeltPoolContext) -> float:
        return _penetration_depth_from_plate(context.mask(), context.dataset)

    @staticmethod
    def _area_um2(context: MeltPoolContext) -> float:
        return float(
            context.mask().sum() * np.prod(context.dataset.canonical_voxel_size_um)
        )

    @staticmethod
    def _volume_um3(context: MeltPoolContext) -> float:
        return float(
            context.mask().sum() * np.prod(context.dataset.canonical_voxel_size_um)
        )

    @staticmethod
    def _keyhole_depth_um(context: MeltPoolContext) -> float:
        return _penetration_depth_from_plate(
            context.keyhole_topology().cavity_mask, context.dataset
        )

    @staticmethod
    def _keyhole_area_um2(context: MeltPoolContext) -> float:
        return float(
            context.keyhole_topology().cavity_mask.sum()
            * np.prod(context.dataset.canonical_voxel_size_um)
        )

    @staticmethod
    def _keyhole_volume_um3(context: MeltPoolContext) -> float:
        return float(
            context.keyhole_topology().cavity_mask.sum()
            * np.prod(context.dataset.canonical_voxel_size_um)
        )

    @staticmethod
    def _leading_wall_angle_deg(context: MeltPoolContext) -> float:
        return float(context.leading_wall_angle().angle_deg)

    @staticmethod
    def _leading_wall_angle_valid(context: MeltPoolContext) -> bool:
        return bool(context.leading_wall_angle().valid)

    @staticmethod
    def _leading_wall_angle_points(context: MeltPoolContext) -> int:
        return int(context.leading_wall_angle().fit_points_xz_um.shape[0])


def _normalize_voxel_size(
    voxel_size_um: float | Sequence[float], *, spatial_dims: int
) -> tuple[float, ...]:
    if np.isscalar(voxel_size_um):
        values = (float(voxel_size_um),) * int(spatial_dims)
    else:
        values = tuple(float(value) for value in voxel_size_um)
    if len(values) != int(spatial_dims):
        raise ValueError(
            f"voxel_size_um must be a scalar or contain {spatial_dims} values."
        )
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("voxel_size_um values must be positive and finite.")
    return values


def _resolve_time_index(time_index: int, n_steps: int) -> int:
    resolved = int(time_index) + int(n_steps) if int(time_index) < 0 else int(time_index)
    if resolved < 0 or resolved >= int(n_steps):
        raise ValueError(f"time_index={time_index} is outside n_steps={n_steps}.")
    return resolved


def _field_channel_index(
    field_names: Sequence[str], field_name: str, time_index: int, n_steps: int
) -> int:
    if field_name not in field_names:
        raise ValueError(f"Field {field_name!r} is absent from {tuple(field_names)!r}.")
    step = _resolve_time_index(time_index, n_steps)
    return step * len(field_names) + tuple(field_names).index(field_name)


def _as_numpy(values: ArrayLike) -> NDArray[np.generic]:
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    return np.asarray(values)


def _largest_component(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if not mask.any():
        return np.asarray(mask, dtype=bool)
    from scipy import ndimage

    structure = np.ones((3,) * mask.ndim, dtype=bool)
    labels, count = ndimage.label(mask, structure=structure)
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return np.asarray(labels == int(np.argmax(sizes)), dtype=bool)


def _physical_extent(
    mask: NDArray[np.bool_], axis: int, spacing_um: float
) -> float:
    reduction_axes = tuple(index for index in range(mask.ndim) if index != axis)
    occupied = mask.any(axis=reduction_axes)
    indices = np.flatnonzero(occupied)
    if not indices.size:
        return 0.0
    return float((indices[-1] - indices[0] + 1) * spacing_um)


def _penetration_depth_from_plate(mask: NDArray[np.bool_], spec: DatasetSpec) -> float:
    array = np.asarray(mask, dtype=bool)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D mask, got {array.shape}.")
    occupied = array.any(axis=tuple(range(1, array.ndim)))
    indices = np.flatnonzero(occupied)
    if not indices.size:
        return 0.0
    spacing = spec.spacing_um("depth")
    if spec.depth_positive_into_plate:
        deepest_face_um = spec.depth_origin_um + (indices[-1] + 1) * spacing
        depth_um = deepest_face_um - spec.plate_height_um
    else:
        deepest_face_um = spec.depth_origin_um + indices[0] * spacing
        depth_um = spec.plate_height_um - deepest_face_um
    return float(max(0.0, depth_um))


def _connectivity_structure(ndim: int) -> NDArray[np.bool_]:
    from scipy import ndimage

    return np.asarray(ndimage.generate_binary_structure(ndim, 1), dtype=bool)


def _physical_binary_closing(
    mask: NDArray[np.bool_], *, radius_um: float, spacing_um: Sequence[float]
) -> NDArray[np.bool_]:
    from scipy import ndimage

    if radius_um <= 0.0:
        return np.asarray(mask, dtype=bool)
    spacing = np.asarray(spacing_um, dtype=float)
    radii = np.ceil(float(radius_um) / spacing).astype(int)
    vectors = [np.arange(-radius, radius + 1) for radius in radii]
    grids = np.meshgrid(*vectors, indexing="ij")
    distance = sum(
        np.square(grid * axis_spacing / float(radius_um))
        for grid, axis_spacing in zip(grids, spacing)
    )
    return np.asarray(
        ndimage.binary_closing(mask, structure=distance <= 1.0), dtype=bool
    )


def _plate_material_cell(dataset: DatasetSpec, depth_count: int) -> tuple[int, int]:
    face = dataset.depth_face_index()
    if face < 0 or face > depth_count:
        raise ValueError(
            f"Plate face {face} lies outside a depth axis of size {depth_count}."
        )
    cell, inward_step = (face, 1) if dataset.depth_positive_into_plate else (face - 1, -1)
    if cell < 0 or cell >= depth_count:
        raise ValueError(
            "The plate plane has no adjacent in-domain cell on the material side."
        )
    return cell, inward_step


def _lateral_footprint(liquid_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    from scipy import ndimage

    footprint = np.asarray(liquid_mask.any(axis=0), dtype=bool)
    return np.asarray(
        ndimage.binary_fill_holes(
            footprint, structure=_connectivity_structure(footprint.ndim)
        ),
        dtype=bool,
    )


def _empty_topology(shape: Sequence[int], method: str) -> KeyholeTopology:
    empty = np.zeros(tuple(shape), dtype=bool)
    return KeyholeTopology(
        cavity_mask=empty,
        pore_labels=np.zeros(tuple(shape), dtype=np.int32),
        liquid_wall_mask=empty.copy(),
        gas_wall_mask=empty.copy(),
        mouth_mask=empty.copy(),
        segmentation_method=method,
    )


def _component_score(
    component: NDArray[np.bool_], dataset: DatasetSpec
) -> tuple[float, int]:
    depth_indices = np.flatnonzero(
        component.any(axis=tuple(range(1, component.ndim)))
    )
    if not depth_indices.size:
        return 0.0, 0
    z_um = dataset.depth_coordinates_um(depth_indices)
    deepest = float(np.max(dataset.depth_below_plate_um(z_um)))
    return deepest, int(component.sum())


def _select_touching_component(
    candidates: NDArray[np.bool_],
    mouth_neighbors: NDArray[np.bool_],
    *,
    dataset: DatasetSpec,
    minimum_cavity_size: int,
) -> tuple[NDArray[np.bool_], NDArray[np.int32]]:
    from scipy import ndimage

    structure = _connectivity_structure(candidates.ndim)
    labels, count = ndimage.label(candidates, structure=structure)
    if count == 0:
        return np.zeros_like(candidates, dtype=bool), labels.astype(np.int32)
    touching = np.unique(labels[mouth_neighbors & (labels > 0)])
    touching = [
        int(label)
        for label in touching
        if int(np.count_nonzero(labels == label)) >= int(minimum_cavity_size)
    ]
    if not touching:
        pores, _ = ndimage.label(candidates, structure=structure)
        return np.zeros_like(candidates, dtype=bool), pores.astype(np.int32)
    selected = max(
        touching, key=lambda label: _component_score(labels == label, dataset)
    )
    cavity = labels == selected
    pores, _ = ndimage.label(candidates & ~cavity, structure=structure)
    return np.asarray(cavity, dtype=bool), pores.astype(np.int32)


def _topology_from_cavity(
    *,
    cavity: NDArray[np.bool_],
    pores: NDArray[np.int32],
    liquid: NDArray[np.bool_],
    mouth: NDArray[np.bool_],
    method: str,
) -> KeyholeTopology:
    from scipy import ndimage

    structure = _connectivity_structure(cavity.ndim)
    liquid_wall = liquid & ndimage.binary_dilation(cavity, structure=structure)
    gas_wall = cavity & ndimage.binary_dilation(liquid, structure=structure)
    return KeyholeTopology(
        cavity_mask=np.asarray(cavity, dtype=bool),
        pore_labels=np.asarray(pores, dtype=np.int32),
        liquid_wall_mask=np.asarray(liquid_wall, dtype=bool),
        gas_wall_mask=np.asarray(gas_wall, dtype=bool),
        mouth_mask=np.asarray(mouth, dtype=bool),
        segmentation_method=method,
    )


def _liquid_cap_fill_topology(
    *,
    liquid_mask: NDArray[np.bool_],
    dataset: DatasetSpec,
    minimum_cavity_size: int,
) -> KeyholeTopology:
    from scipy import ndimage

    method = "liquid_cap_fill"
    liquid = np.asarray(liquid_mask, dtype=bool)
    footprint = _lateral_footprint(liquid)
    if not footprint.any():
        return _empty_topology(liquid.shape, method)
    cap_index, inward_step = _plate_material_cell(dataset, liquid.shape[0])
    sealed = liquid.copy()
    sealed[cap_index][footprint] = True
    structure = _connectivity_structure(liquid.ndim)
    envelope = ndimage.binary_fill_holes(sealed, structure=structure)
    candidates = np.asarray(envelope & ~sealed, dtype=bool)
    neighbor_index = cap_index + inward_step
    mouth_neighbors = np.zeros_like(liquid, dtype=bool)
    if 0 <= neighbor_index < liquid.shape[0]:
        mouth_neighbors[neighbor_index][footprint] = True
    cavity, pores = _select_touching_component(
        candidates,
        mouth_neighbors,
        dataset=dataset,
        minimum_cavity_size=minimum_cavity_size,
    )
    return _topology_from_cavity(
        cavity=cavity,
        pores=pores,
        liquid=liquid,
        mouth=cavity & mouth_neighbors,
        method=method,
    )


def _material_flood_topology(
    *,
    material_mask: NDArray[np.bool_],
    liquid_mask: NDArray[np.bool_],
    dataset: DatasetSpec,
    minimum_cavity_size: int,
) -> KeyholeTopology:
    from scipy import ndimage

    method = "material_flood"
    material = np.asarray(material_mask, dtype=bool)
    liquid = np.asarray(liquid_mask, dtype=bool)
    if material.shape != liquid.shape:
        raise ValueError(
            f"Material and liquid fields must match, got {material.shape} and "
            f"{liquid.shape}."
        )
    void = ~material
    structure = _connectivity_structure(void.ndim)
    labels, _ = ndimage.label(void, structure=structure)
    gas_face = 0 if dataset.depth_positive_into_plate else void.shape[0] - 1
    gas_labels = np.unique(labels[gas_face])
    gas_connected = np.isin(labels, gas_labels[gas_labels > 0])
    depth_indices = np.arange(void.shape[0], dtype=float)
    below_1d = dataset.depth_below_plate_um(
        dataset.depth_coordinates_um(depth_indices)
    ) >= 0.0
    below = below_1d.reshape((void.shape[0],) + (1,) * (void.ndim - 1))
    footprint = _lateral_footprint(liquid)
    inside = np.broadcast_to(footprint, void.shape)
    candidates = gas_connected & below & inside
    plate_cell, inward_step = _plate_material_cell(dataset, void.shape[0])
    mouth_neighbors = np.zeros_like(void, dtype=bool)
    for depth_index in (plate_cell, plate_cell + inward_step):
        if 0 <= depth_index < void.shape[0]:
            mouth_neighbors[depth_index][footprint] = True
    cavity, _ = _select_touching_component(
        candidates,
        mouth_neighbors,
        dataset=dataset,
        minimum_cavity_size=minimum_cavity_size,
    )
    enclosed = void & below & inside & ~gas_connected
    pore_labels, _ = ndimage.label(enclosed, structure=structure)
    return _topology_from_cavity(
        cavity=cavity,
        pores=pore_labels.astype(np.int32),
        liquid=liquid,
        mouth=cavity & mouth_neighbors,
        method=method,
    )


def _split_kept_contour(
    points: NDArray[np.float64], keep: NDArray[np.bool_]
) -> list[NDArray[np.float64]]:
    points = np.asarray(points, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    if points.shape[0] != keep.size or points.shape[0] < 2:
        return []
    if keep.all():
        return [points]
    closed = np.linalg.norm(points[0] - points[-1]) <= 1.5
    if closed and keep[0] and keep[-1]:
        false_indices = np.flatnonzero(~keep)
        if false_indices.size:
            shift = -(int(false_indices[0]) + 1)
            points = np.roll(points, shift, axis=0)
            keep = np.roll(keep, shift)
    padded = np.concatenate(([False], keep, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [
        np.asarray(points[start:stop], dtype=float)
        for start, stop in zip(starts, stops)
        if stop - start >= 2
    ]


def _extract_supported_contours(
    field: NDArray[np.generic], *, level: float, wall_mask: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], ...]:
    from scipy import ndimage
    from skimage.measure import find_contours

    array = np.asarray(field, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D contour field, got {array.shape}.")
    support = ndimage.binary_dilation(
        np.asarray(wall_mask, dtype=bool), structure=np.ones((3, 3), dtype=bool)
    )
    output: list[NDArray[np.float64]] = []
    for contour in find_contours(array, level=float(level)):
        row = np.clip(np.rint(contour[:, 0]).astype(int), 0, support.shape[0] - 1)
        column = np.clip(
            np.rint(contour[:, 1]).astype(int), 0, support.shape[1] - 1
        )
        output.extend(_split_kept_contour(contour, support[row, column]))
    return tuple(output)


def _extract_wall_contours_xz_um(
    *,
    scalar_field_dl: NDArray[np.generic],
    contour_level: float,
    cavity_mask_dl: NDArray[np.bool_],
    liquid_wall_mask_dl: NDArray[np.bool_],
    dataset: DatasetSpec,
) -> tuple[NDArray[np.float64], ...]:
    cavity = np.asarray(cavity_mask_dl, dtype=bool)
    wall = np.asarray(liquid_wall_mask_dl, dtype=bool)
    if not cavity.any() or not wall.any():
        return ()
    contours = _extract_supported_contours(
        scalar_field_dl, level=contour_level, wall_mask=wall
    )
    if not contours:
        contours = _extract_supported_contours(
            cavity.astype(float), level=0.5, wall_mask=wall
        )
    physical: list[NDArray[np.float64]] = []
    for contour in contours:
        z_um = dataset.depth_origin_um + (
            contour[:, 0] + 0.5
        ) * dataset.spacing_um("depth")
        x_um = (contour[:, 1] + 0.5) * dataset.spacing_um("length")
        physical.append(np.column_stack((x_um, z_um)).astype(float))
    return tuple(physical)


def _mask_points_xz_um(
    mask_dl: NDArray[np.bool_], dataset: DatasetSpec
) -> NDArray[np.float64]:
    indices = np.argwhere(np.asarray(mask_dl, dtype=bool))
    if not indices.size:
        return np.empty((0, 2), dtype=float)
    z_um = dataset.depth_coordinates_um(indices[:, 0])
    x_um = (indices[:, 1].astype(float) + 0.5) * dataset.spacing_um("length")
    return np.column_stack((x_um, z_um)).astype(float)


def _select_leading_wall_segment(
    contour_segments_xz_um: Sequence[NDArray[np.float64]],
    mouth_points_xz_um: NDArray[np.float64],
    *,
    leading_direction: LeadingDirection,
) -> NDArray[np.float64] | None:
    segments = [
        np.asarray(segment, dtype=float)
        for segment in contour_segments_xz_um
        if np.asarray(segment).ndim == 2 and np.asarray(segment).shape[0] >= 2
    ]
    if not segments:
        return None
    mouth = np.asarray(mouth_points_xz_um, dtype=float)
    if mouth.ndim == 2 and mouth.shape[0]:
        leading_mouth = (
            mouth[np.argmax(mouth[:, 0])]
            if leading_direction == "max_length"
            else mouth[np.argmin(mouth[:, 0])]
        )

        def score(segment: NDArray[np.float64]) -> float:
            return float(np.min(np.sum(np.square(segment - leading_mouth), axis=1)))

        return min(segments, key=score)
    if leading_direction == "max_length":
        return max(segments, key=lambda segment: float(np.mean(segment[:, 0])))
    return min(segments, key=lambda segment: float(np.mean(segment[:, 0])))


def _orthogonal_line_angle_deg(points_xz_um: NDArray[np.float64]) -> float:
    """Return acute dominant-line angle relative to the horizontal plate plane."""

    points = np.asarray(points_xz_um, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected wall points shaped (N, 2), got {points.shape}.")
    if points.shape[0] < 2:
        return np.nan
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    if not singular_values.size or singular_values[0] <= 0.0:
        return np.nan
    dx, dz = directions[0]
    return float(np.degrees(np.arctan2(abs(dz), abs(dx))))


__all__ = [
    "AxisName",
    "DatasetSpec",
    "KeyholeSpec",
    "KeyholeSurface2D",
    "KeyholeSurface3D",
    "KeyholeTopology",
    "LeadingWallAngle",
    "MeltPoolContext",
    "MeltPoolMetrics",
    "MetricDefinition",
    "MetricResult",
    "MetricUnavailableError",
    "RegionSpec",
]
