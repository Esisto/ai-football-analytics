"""Futsal ground-plane geometry for manually calibrated single-camera footage.

Coordinates are in metres: origin at one corner of the left goal line;
x runs along the touchline, y along the goal line. A camera may stand
anywhere (side, corner, or behind a goal); only labelled image-to-ground
correspondences matter. Other sports' court lines must NOT be guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

from calibration.minimap import MinimapRenderer

Point = Tuple[float, float]


@dataclass(frozen=True)
class FutsalPitchDimensions:
    length: float = 40.0
    width: float = 20.0

    def __post_init__(self) -> None:
        if (not math.isfinite(self.length) or not math.isfinite(self.width)
                or self.width <= 0 or self.length <= self.width):
            raise ValueError("Field dimensions must be finite, positive, and length > width")

    def is_regulation_size(self, international: bool = False) -> bool:
        """FIFA 2025/26 size ranges. False is allowed for a training court."""
        if international:
            return 38 <= self.length <= 42 and 20 <= self.width <= 25
        return 25 <= self.length <= 42 and 16 <= self.width <= 25

    def to_dict(self, camera_position: str = "unknown") -> dict:
        if camera_position not in CAMERA_POSITIONS:
            raise ValueError(f"Unsupported camera position: {camera_position}")
        return {"sport": "futsal", "length_m": self.length,
                "width_m": self.width, "camera_position": camera_position}


CAMERA_POSITIONS = ("unknown", "sideline", "behind_goal", "corner", "other")


class FutsalPitchModel:
    """Named *ground-plane* landmarks selected explicitly by the operator.

    For a multipurpose hall, only click landmarks from the active futsal
    layout. Each keyframe needs four DISTINCT non-collinear visible points.
    Calibration is impossible if the footage hides sufficient landmarks.
    """

    def __init__(self, dims: FutsalPitchDimensions = FutsalPitchDimensions()) -> None:
        self.dims = dims
        length, width = dims.length, dims.width
        cx, cy = length / 2, width / 2
        self.points: Dict[str, Point] = {
            "top_left_corner": (0, 0),
            "top_right_corner": (length, 0),
            "bottom_left_corner": (0, width),
            "bottom_right_corner": (length, width),
            "halfway_top_touchline": (cx, 0),
            "halfway_bottom_touchline": (cx, width),
            "center_spot": (cx, cy),
            "center_circle_top": (cx, cy - 3),
            "center_circle_bottom": (cx, cy + 3),
            "left_penalty_spot": (6, cy),
            "right_penalty_spot": (length - 6, cy),
            "left_second_penalty_spot": (10, cy),
            "right_second_penalty_spot": (length - 10, cy),
            "left_goalpost_top_base": (0, cy - 1.5),
            "left_goalpost_bottom_base": (0, cy + 1.5),
            "right_goalpost_top_base": (length, cy - 1.5),
            "right_goalpost_bottom_base": (length, cy + 1.5),
        }

    def field_point(self, name: str) -> Point:
        return self.points[name]

    def names(self) -> List[str]:
        return list(self.points)

    def contains(self, name: str) -> bool:
        return name in self.points


class FutsalMinimapRenderer(MinimapRenderer):
    """Futsal court renderer using the same projection API as football."""

    def __init__(self, dims: FutsalPitchDimensions = FutsalPitchDimensions(),
                 **kwargs) -> None:
        super().__init__(dims=dims, stripes=0, **kwargs)

    def _draw_pitch(self) -> np.ndarray:
        d = self.dims
        img = np.full((self.height_px, self.width_px, 3), (48, 124, 52), np.uint8)
        white = (244, 244, 244)
        line_width = 2
        def pt(x: float, y: float):
            return self.field_to_px(x, y)
        def segment(a: Point, b: Point):
            cv2.line(img, pt(*a), pt(*b), white, line_width, cv2.LINE_AA)
        def spot(x: float, y: float):
            cv2.circle(img, pt(x, y), 3, white, -1, cv2.LINE_AA)
        length, width = d.length, d.width
        cx, cy = length / 2, width / 2
        cv2.rectangle(img, pt(0, 0), pt(length, width), white, line_width, cv2.LINE_AA)
        segment((cx, 0), (cx, width))
        cv2.circle(img, pt(cx, cy), round(3 * self.scale), white, line_width, cv2.LINE_AA)
        spot(cx, cy)
        for x in (6, length - 6, 10, length - 10):
            spot(x, cy)
        # Simplified 6 m penalty-area outlines: arcs centred at goalpost bases,
        # joined by the 3 m segment parallel to the goal line.
        radius = round(6 * self.scale)
        for left in (True, False):
            x = 0 if left else length
            direction = 1 if left else -1
            segment((x + direction * 6, cy - 1.5),
                    (x + direction * 6, cy + 1.5))
            # Draw the two quarter-arcs using sampled points (keeps orientation clear).
            for sign in (-1, 1):
                arc = []
                for step in range(25):
                    theta = step * math.pi / 2 / 24
                    ax = x + direction * 6 * math.sin(theta)
                    ay = cy + sign * (1.5 + 6 * math.cos(theta))
                    arc.append(pt(ax, ay))
                cv2.polylines(img, [np.asarray(arc, dtype=np.int32)],
                              False, white, line_width, cv2.LINE_AA)
            # Short goal stubs behind the line (3 m wide).
            segment((x, cy - 1.5), (x - direction, cy - 1.5))
            segment((x - direction, cy - 1.5), (x - direction, cy + 1.5))
            segment((x - direction, cy + 1.5), (x, cy + 1.5))
        return img
