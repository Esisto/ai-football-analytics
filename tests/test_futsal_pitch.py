"""Tests for single-camera futsal ground-plane support (no video/model needed)."""
import json

import numpy as np
import pytest

from calibration.futsal_pitch import (
    FutsalPitchDimensions, FutsalPitchModel, FutsalMinimapRenderer,
)
from calibration.futsal_setup import profile_path_for_video, save_profile, load_profile
from calibration.homography import Homography


@pytest.mark.parametrize("length,width", [(25, 16), (38, 20), (40, 20), (42, 25)])
def test_regulation_courts(length, width):
    dims = FutsalPitchDimensions(length, width)
    assert dims.is_regulation_size()


def test_international_range_and_nonstandard_training_court():
    assert FutsalPitchDimensions(40, 20).is_regulation_size(international=True)
    assert not FutsalPitchDimensions(30, 18).is_regulation_size(international=True)
    assert not FutsalPitchDimensions(24, 15).is_regulation_size()
    assert FutsalPitchDimensions(24, 15).to_dict("behind_goal")["sport"] == "futsal"


@pytest.mark.parametrize("length,width", [(0, 20), (-1, 20), (20, 20),
                                           (20, 40), (float("nan"), 20)])
def test_bad_pitch_dimensions(length, width):
    with pytest.raises(ValueError):
        FutsalPitchDimensions(length, width)


def test_landmarks_scale_with_court_dimensions():
    pitch = FutsalPitchModel(FutsalPitchDimensions(38, 19))
    assert pitch.field_point("top_right_corner") == (38, 0)
    assert pitch.field_point("bottom_right_corner") == (38, 19)
    assert pitch.field_point("center_spot") == (19, 9.5)
    assert pitch.field_point("left_penalty_spot") == (6, 9.5)
    assert pitch.field_point("right_second_penalty_spot") == (28, 9.5)


@pytest.mark.parametrize("position", ["sideline", "behind_goal", "corner", "other"])
def test_camera_position_is_metadata_not_a_homography_assumption(position):
    dims = FutsalPitchDimensions()
    assert dims.to_dict(position)["camera_position"] == position
    field = [(0, 0), (40, 0), (40, 20), (0, 20)]
    # Synthetic oblique image correspondences, independent of named camera.
    img = [(140, 500), (440, 260), (920, 540), (160, 970)]
    H = Homography.from_correspondences(img, field)
    for image, expected in zip(img, field):
        assert H.project_image_to_field(image) == pytest.approx(expected, abs=1e-3)


def test_video_specific_profile_round_trip(tmp_path):
    video = tmp_path / "training.mov"
    dims = FutsalPitchDimensions(38, 19)
    saved = save_profile(video, dims, "corner", tmp_path)
    assert saved == profile_path_for_video(video, tmp_path)
    assert json.loads(saved.read_text())["camera_position"] == "corner"
    assert load_profile(video, tmp_path) == dims
    assert load_profile(tmp_path / "different.mov", tmp_path) is None


def test_invalid_profile_rejected(tmp_path):
    video = tmp_path / "training.mp4"
    path = profile_path_for_video(video, tmp_path)
    path.write_text(json.dumps({"sport": "football", "video": video.name,
                                "length_m": 40, "width_m": 20}))
    with pytest.raises(ValueError):
        load_profile(video, tmp_path)


def test_futsal_minimap_dimensions_and_lines():
    dims = FutsalPitchDimensions(38, 19)
    renderer = FutsalMinimapRenderer(dims=dims, px_per_meter=6)
    assert renderer.on_pitch(37.5, 18.5)
    assert not renderer.on_pitch(40, 20)
    image = renderer.blank()
    assert image.shape == (renderer.height_px, renderer.width_px, 3)
    assert np.any(np.all(image > 200, axis=2))
    assert renderer.field_to_px(19, 9.5) == (
        round(renderer.margin + 19 * renderer.scale),
        round(renderer.margin + 9.5 * renderer.scale),
    )
