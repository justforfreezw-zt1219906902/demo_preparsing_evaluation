import csv

import pytest

from openscan_eval.config import load_dotenv
from openscan_eval.dataset.metadata import read_positions_csv


def test_csv_parsing(tmp_path):
    path = tmp_path / "positions.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "position_index", "phi_deg", "theta_deg", "focus_index", "focus_value"])
        writer.writerow(["frame.jpg", "1", "-20", "30.5", "", ""])
    frame = read_positions_csv(path)[0]
    assert (frame.image, frame.position_index, frame.phi_deg, frame.theta_deg) == ("frame.jpg", 1, -20.0, 30.5)
    assert frame.pose_source == "openscan_commanded"


def test_required_angles_are_enforced(tmp_path):
    path = tmp_path / "positions.csv"
    path.write_text("image,position_index\nframe.jpg,1\n")
    with pytest.raises(ValueError, match="phi_deg, theta_deg"):
        read_positions_csv(path)


def test_dotenv_loads_paths_without_overwriting_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENSCAN_DATASET_DIR", raising=False)
    monkeypatch.setenv("OPENSCAN_REFERENCE_MESH", "/already/set.stl")
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OPENSCAN_DATASET_DIR=/data/images\nOPENSCAN_REFERENCE_MESH=/from/file.stl\n"
    )
    load_dotenv(dotenv)
    assert __import__("os").environ["OPENSCAN_DATASET_DIR"] == "/data/images"
    assert __import__("os").environ["OPENSCAN_REFERENCE_MESH"] == "/already/set.stl"
