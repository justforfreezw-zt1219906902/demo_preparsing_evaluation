import csv

from PIL import Image

from openscan_eval.config import load_config
from openscan_eval.dataset.loader import load_dataset
from openscan_eval.dataset.validation import validate_dataset


def make_dataset(tmp_path, rows):
    with (tmp_path / "positions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "position_index", "phi_deg", "theta_deg"])
        writer.writerows(rows)
    return tmp_path


def test_valid_dataset(tmp_path):
    images = make_dataset(tmp_path, [["a.jpg", 1, -20, 0], ["b.jpg", 2, -20, 30]])
    Image.new("RGB", (12, 8)).save(images / "a.jpg")
    Image.new("RGB", (12, 8)).save(images / "b.jpg")
    report = validate_dataset(load_dataset(load_config(), tmp_path), load_config())
    assert report["valid"] is True
    assert report["decoded_image_count"] == 2


def test_dataset_directory_comes_from_environment(tmp_path, monkeypatch):
    make_dataset(tmp_path, [])
    monkeypatch.setenv("OPENSCAN_DATASET_DIR", str(tmp_path))
    dataset = load_dataset(load_config())
    assert dataset.root == tmp_path.resolve()
    assert dataset.images_dir == tmp_path.resolve()


def test_missing_image_is_reported(tmp_path):
    make_dataset(tmp_path, [["missing.jpg", 1, 0, 0]])
    report = validate_dataset(load_dataset(load_config(), tmp_path), load_config())
    assert report["valid"] is False
    assert report["issues"][0]["code"] == "missing_image"


def test_duplicates_corruption_and_dimensions_are_reported(tmp_path):
    images = make_dataset(tmp_path, [["a.jpg", 1, 0, 0], ["a.jpg", 1, 0, 30], ["bad.jpg", 3, 0, 60], ["b.jpg", 4, 0, 90]])
    Image.new("RGB", (12, 8)).save(images / "a.jpg")
    Image.new("RGB", (10, 8)).save(images / "b.jpg")
    (images / "bad.jpg").write_text("not an image")
    report = validate_dataset(load_dataset(load_config(), tmp_path), load_config())
    codes = {issue["code"] for issue in report["issues"]}
    assert {"duplicate_image", "duplicate_position_index", "corrupted_image", "inconsistent_dimensions"} <= codes
