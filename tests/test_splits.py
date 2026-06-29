import sys
from pathlib import Path

# Allow imports from src/ regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from unittest.mock import MagicMock, patch
from data.wildfire_dataset import load_split_ids

BUCKET_NAME   = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_mock_blob(content=None, exists=True):
    blob = MagicMock()
    blob.exists.return_value = exists
    if content is not None:
        blob.download_as_text.return_value = content
    return blob


def mock_load_split_ids(content, branch="landsat", split="train", exists=True):
    """Call load_split_ids with a mocked GCS blob returning `content`."""
    with patch("data.wildfire_dataset.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob   = make_mock_blob(content=content, exists=exists)
        mock_bucket.blob.return_value = mock_blob
        mock_client.return_value.bucket.return_value = mock_bucket
        return load_split_ids(BUCKET_NAME, branch, split, OUTPUT_PREFIX)


# ==================================================================
# UNIT TESTS
# ==================================================================

class TestLoadSplitIds:

    @pytest.mark.unit
    def test_returns_set(self):
        result = mock_load_split_ids("id_001\nid_002\nid_003")
        assert isinstance(result, set)

    @pytest.mark.unit
    def test_correct_ids_loaded(self):
        result = mock_load_split_ids("id_001\nid_002\nid_003")
        assert result == {"id_001", "id_002", "id_003"}

    @pytest.mark.unit
    def test_strips_whitespace(self):
        """Leading/trailing whitespace and blank lines must be ignored."""
        result = mock_load_split_ids("  id_001  \n\nid_002\n  \nid_003\n")
        assert result == {"id_001", "id_002", "id_003"}

    @pytest.mark.unit
    def test_empty_file_returns_empty_set(self):
        result = mock_load_split_ids("")
        assert result == set()

    @pytest.mark.unit
    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            mock_load_split_ids(content=None, exists=False)
        assert "split_data.py" in str(exc_info.value)

    @pytest.mark.unit
    def test_no_duplicate_ids(self):
        """Duplicate lines in the file should collapse to one entry."""
        result = mock_load_split_ids("id_001\nid_001\nid_002")
        assert len(result) == 2
        assert "id_001" in result

    @pytest.mark.unit
    def test_correct_gcs_path_constructed(self):
        """load_split_ids must look up the right GCS path."""
        with patch("data.wildfire_dataset.storage.Client") as mock_client:
            mock_bucket = MagicMock()
            mock_blob   = make_mock_blob(content="id_001", exists=True)
            mock_bucket.blob.return_value = mock_blob
            mock_client.return_value.bucket.return_value = mock_bucket

            load_split_ids(BUCKET_NAME, "viirs", "val", OUTPUT_PREFIX)

            expected_path = f"{OUTPUT_PREFIX}/splits/viirs/val.txt"
            mock_bucket.blob.assert_called_once_with(expected_path)

    @pytest.mark.unit
    def test_works_for_all_split_names(self):
        for split in ["train", "val", "test"]:
            result = mock_load_split_ids("id_001\nid_002", split=split)
            assert len(result) == 2

    @pytest.mark.unit
    def test_works_for_both_branches(self):
        for branch in ["landsat", "viirs"]:
            result = mock_load_split_ids("id_001\nid_002", branch=branch)
            assert len(result) == 2


# ==================================================================
# INTEGRATION TESTS  (requires GCS credentials and real bucket)
# ==================================================================

@pytest.mark.integration
class TestSplitIntegrityLandsat:
    """
    Validate the real landsat split files in GCS.
    These tests confirm structural correctness of the splits produced
    by split_data.py — they do not re-run the split logic.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def splits(cls):
        return {
            split: load_split_ids(BUCKET_NAME, "landsat", split, OUTPUT_PREFIX)
            for split in ["train", "val", "test"]
        }

    def test_all_splits_non_empty(self, splits):
        for split, ids in splits.items():
            assert len(ids) > 0, f"Landsat {split} split is empty"

    def test_no_overlap_train_val(self, splits):
        overlap = splits["train"] & splits["val"]
        assert len(overlap) == 0, \
            f"Train/val overlap: {len(overlap)} shared IDs: {list(overlap)[:5]}"

    def test_no_overlap_train_test(self, splits):
        overlap = splits["train"] & splits["test"]
        assert len(overlap) == 0, \
            f"Train/test overlap: {len(overlap)} shared IDs: {list(overlap)[:5]}"

    def test_no_overlap_val_test(self, splits):
        overlap = splits["val"] & splits["test"]
        assert len(overlap) == 0, \
            f"Val/test overlap: {len(overlap)} shared IDs: {list(overlap)[:5]}"

    def test_approximate_split_ratios(self, splits):
        """
        Splits should be roughly 70/15/15. Allow ±10% absolute tolerance
        to account for rounding with small datasets.
        """
        total = sum(len(s) for s in splits.values())
        train_frac = len(splits["train"]) / total
        val_frac   = len(splits["val"])   / total
        test_frac  = len(splits["test"])  / total

        assert 0.60 <= train_frac <= 0.80, \
            f"Train fraction {train_frac:.2f} outside expected 0.60–0.80"
        assert 0.05 <= val_frac   <= 0.25, \
            f"Val fraction {val_frac:.2f} outside expected 0.05–0.25"
        assert 0.05 <= test_frac  <= 0.25, \
            f"Test fraction {test_frac:.2f} outside expected 0.05–0.25"

    def test_total_sample_count_plausible(self, splits):
        """Total IDs across all splits should be close to the known dataset size."""
        total = sum(len(s) for s in splits.values())
        # Landsat dataset has 1,214 complete pairs; allow some filtering headroom
        assert total >= 800, f"Total landsat split IDs suspiciously low: {total}"
        assert total <= 1300, f"Total landsat split IDs suspiciously high: {total}"

    def test_ids_are_non_empty_strings(self, splits):
        for split, ids in splits.items():
            for sid in ids:
                assert isinstance(sid, str) and len(sid) > 0, \
                    f"Landsat {split}: found empty or non-string ID: {repr(sid)}"


@pytest.mark.integration
class TestSplitIntegrityVIIRS:
    """Validate the real VIIRS split files in GCS."""

    @pytest.fixture(scope="class")
    @classmethod
    def splits(cls):
        return {
            split: load_split_ids(BUCKET_NAME, "viirs", split, OUTPUT_PREFIX)
            for split in ["train", "val", "test"]
        }

    def test_all_splits_non_empty(self, splits):
        for split, ids in splits.items():
            assert len(ids) > 0, f"VIIRS {split} split is empty"

    def test_no_overlap_train_val(self, splits):
        overlap = splits["train"] & splits["val"]
        assert len(overlap) == 0, \
            f"VIIRS train/val overlap: {len(overlap)} IDs"

    def test_no_overlap_train_test(self, splits):
        overlap = splits["train"] & splits["test"]
        assert len(overlap) == 0, \
            f"VIIRS train/test overlap: {len(overlap)} IDs"

    def test_total_sample_count_plausible(self, splits):
        total = sum(len(s) for s in splits.values())
        # VIIRS dataset has 4,117 complete pairs; allow filtering headroom
        assert total >= 2000, f"Total VIIRS split IDs suspiciously low: {total}"
        assert total <= 4200, f"Total VIIRS split IDs suspiciously high: {total}"


@pytest.mark.integration
class TestCrossDatasetConsistency:
    """
    Check that landsat and viirs splits don't accidentally share IDs.
    They shouldn't since IDs encode the sensor name, but worth verifying.
    """

    def test_landsat_viirs_ids_are_distinct(self):
        ls_train = load_split_ids(BUCKET_NAME, "landsat", "train", OUTPUT_PREFIX)
        vi_train = load_split_ids(BUCKET_NAME, "viirs",   "train", OUTPUT_PREFIX)
        overlap  = ls_train & vi_train
        assert len(overlap) == 0, \
            f"Landsat and VIIRS train splits share {len(overlap)} IDs"