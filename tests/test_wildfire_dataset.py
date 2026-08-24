import io
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
import rasterio
from rasterio.transform import from_bounds
from unittest.mock import MagicMock, patch
from data.wildfire_dataset import WildfireDataset

# ------------------------------------------------------------------
# Constants (updated to match current WildfireDataset)
# ------------------------------------------------------------------
LANDSAT_TARGET_SIZE = (256, 256)
LANDSAT_COV_SIZE    = (256, 256)
LANDSAT_GOES_SIZE   = (4, 4)          # was (15, 15)
LANDSAT_COV_CHANNELS = 11             # 1 DEM + 10 one-hot land cover
LANDSAT_INPUT_CHANNELS = 3 + LANDSAT_COV_CHANNELS  # goes_up + covariates

VIIRS_TARGET_SIZE   = (20, 20)
VIIRS_COV_SIZE      = (256, 256)       # DEFAULT_SIZES: viirs cov_size = (256, 256)
VIIRS_GOES_SIZE     = (4, 4)
VIIRS_COV_CHANNELS  = 11
VIIRS_INPUT_CHANNELS = 3 + VIIRS_COV_CHANNELS

BUCKET_NAME   = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset"

LANDCOVER_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]


# ------------------------------------------------------------------
# Helpers: build synthetic GeoTIFFs in memory
# ------------------------------------------------------------------

def make_tif_bytes(shape, dtype=np.float32, value=None, nodata=None):
    c, h, w = shape
    if value is not None:
        data = np.full((c, h, w), value, dtype=dtype)
    else:
        rng  = np.random.default_rng(42)
        data = rng.uniform(200, 330, size=(c, h, w)).astype(dtype)

    transform = from_bounds(0, 0, 1, 1, w, h)
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w", driver="GTiff", height=h, width=w, count=c,
        dtype=dtype, crs="EPSG:4326", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data)
    return buf.getvalue()


def make_covariate_tif_bytes(shape):
    """Raw 2-band covariate raster: band 0 = DEM (elevation, m), band 1 = land cover class code."""
    c, h, w = shape
    assert c == 2, "Raw covariate raster must have 2 bands (DEM, land cover code)"
    rng = np.random.default_rng(7)
    dem = rng.uniform(0, 1500, size=(h, w)).astype(np.float32)
    lc_code = rng.choice(LANDCOVER_CLASSES, size=(h, w)).astype(np.float32)
    data = np.stack([dem, lc_code], axis=0)

    transform = from_bounds(0, 0, 1, 1, w, h)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=h, width=w, count=c,
                       dtype=np.float32, crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)
    return buf.getvalue()


def make_binary_mask_bytes(shape):
    c, h, w = shape
    data = np.zeros((c, h, w), dtype=np.float32)
    rng  = np.random.default_rng(0)
    n_fire = max(1, h * w // 100)
    idxs = rng.choice(h * w, size=n_fire, replace=False)
    data[0].flat[idxs] = 1.0

    transform = from_bounds(0, 0, 1, 1, w, h)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff",
                       height=h, width=w, count=c,
                       dtype=np.float32, crs="EPSG:4326",
                       transform=transform) as dst:
        dst.write(data)
    return buf.getvalue()


def make_stripy_goes_bytes(shape=(3, 4, 4), zero_rows=1):
    c, h, w = shape
    rng  = np.random.default_rng(1)
    data = rng.uniform(200, 330, size=(c, h, w)).astype(np.float32)
    data[:, :zero_rows, :] = 0.0

    transform = from_bounds(0, 0, 1, 1, w, h)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff",
                       height=h, width=w, count=c,
                       dtype=np.float32, crs="EPSG:4326",
                       transform=transform) as dst:
        dst.write(data)
    return buf.getvalue()


def make_manifest(sample_id="test_sample_001", branch="landsat", is_valid=True):
    return {
        "id":   sample_id,
        "date": "2020-08-11",
        "is_valid": is_valid,
        "paths": {
            "goes_input":  f"processed_dataset/{branch}_pairs/goes_input/{sample_id}.tif",
            "target":      f"processed_dataset/{branch}_pairs/target/{sample_id}.tif",
            "covariates":  f"processed_dataset/{branch}_pairs/covariates/{sample_id}.tif",
        },
    }


# ------------------------------------------------------------------
# Mock bucket factory
# ------------------------------------------------------------------

def make_mock_bucket(branch="landsat"):
    bucket  = MagicMock()
    registry = {}

    def get_blob(path):
        if path not in registry:
            blob = MagicMock()
            blob.exists.return_value = True
            blob.name = path

            if "goes_input" in path:
                blob.download_as_bytes.return_value = make_tif_bytes((3, 4, 4))
            elif "target" in path:
                target_shape = (1, 256, 256) if branch == "landsat" else (1, 20, 20)
                blob.download_as_bytes.return_value = make_binary_mask_bytes(target_shape)
            elif "covariates" in path:
                blob.download_as_bytes.return_value = make_covariate_tif_bytes((2, 256, 256))
            else:
                blob.download_as_bytes.return_value = make_tif_bytes((1, 10, 10))

            registry[path] = blob
        return registry[path]

    bucket.blob.side_effect = get_blob
    bucket.list_blobs.return_value = []
    return bucket


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DEFAULT_GOES_MEAN = [305.98227561473846, 296.6176749324799, 9.364600682258606]
DEFAULT_GOES_STD  = [12.870277949475222, 9.583713562045117, 7.1649856621255585]
DEFAULT_DEM_MEAN  = [0.3390881556710402]
DEFAULT_DEM_STD   = [0.5048759609237181]

def _make_dataset(branch, mock_bucket):
    ds = WildfireDataset.__new__(WildfireDataset)

    # Private bucket cache (property reads these, not `bucket` directly)
    ds._bucket = mock_bucket
    ds._client_pid = os.getpid()
    ds.bucket_name = BUCKET_NAME

    ds.branch = branch
    ds.output_prefix = OUTPUT_PREFIX

    ds.goes_mean = torch.tensor(DEFAULT_GOES_MEAN).view(-1, 1, 1)
    ds.goes_std  = torch.tensor(DEFAULT_GOES_STD).view(-1, 1, 1)
    ds.dem_mean  = torch.tensor(DEFAULT_DEM_MEAN).view(-1, 1, 1)
    ds.dem_std   = torch.tensor(DEFAULT_DEM_STD).view(-1, 1, 1)

    ds.binarize_target = (branch == "viirs")
    ds.dilate_target = False
    ds.heatmap_target = False
    ds.heatmap_sigma = 1.0
    ds.dilation_kernel_size = 5 if branch == "landsat" else 3

    ds.drop_dem = False
    ds.drop_landcover = False
    ds.use_stage1_pred = False
    ds.stage1_pred_key = "stage1_pred"
    ds.augment = False

    if branch == "landsat":
        ds.target_size = LANDSAT_TARGET_SIZE
        ds.cov_size    = LANDSAT_COV_SIZE
        ds.goes_size   = LANDSAT_GOES_SIZE
    else:
        ds.target_size = VIIRS_TARGET_SIZE
        ds.cov_size    = VIIRS_COV_SIZE
        ds.goes_size   = VIIRS_GOES_SIZE

    ds.samples   = [make_manifest(f"sample_{branch}_001", branch)]
    ds.n_dropped = 0
    return ds


@pytest.fixture
def landsat_bucket():
    return make_mock_bucket("landsat")

@pytest.fixture
def viirs_bucket():
    return make_mock_bucket("viirs")

@pytest.fixture
def landsat_dataset(landsat_bucket):
    return _make_dataset("landsat", landsat_bucket)

@pytest.fixture
def viirs_dataset(viirs_bucket):
    return _make_dataset("viirs", viirs_bucket)


# ==================================================================
# UNIT TESTS
# ==================================================================

class TestOutputShapesLandsat:

    @pytest.mark.unit
    def test_input_shape(self, landsat_dataset):
        sample = landsat_dataset[0]
        expected = (LANDSAT_INPUT_CHANNELS, 256, 256)
        assert sample["input"].shape == expected, \
            f"Expected {expected}, got {sample['input'].shape}"

    @pytest.mark.unit
    def test_target_shape(self, landsat_dataset):
        sample = landsat_dataset[0]
        assert sample["target"].shape == (1, 256, 256)

    @pytest.mark.unit
    def test_goes_shape(self, landsat_dataset):
        """Raw GOES in the sample dict is at native low-res size, not upsampled."""
        sample = landsat_dataset[0]
        assert sample["goes"].shape == (3,) + LANDSAT_GOES_SIZE

    @pytest.mark.unit
    def test_covariates_shape(self, landsat_dataset):
        sample = landsat_dataset[0]
        expected = (LANDSAT_COV_CHANNELS, 256, 256)
        assert sample["covariates"].shape == expected, \
            f"Expected {expected}, got {sample['covariates'].shape}"

    @pytest.mark.unit
    def test_input_channels_are_goes_plus_covariates(self, landsat_dataset):
        """input[3:] must match the covariates tensor exactly (goes_up is first 3 channels)."""
        sample = landsat_dataset[0]
        assert torch.allclose(sample["input"][3:], sample["covariates"]), \
            "input[3:] does not match covariates tensor"


class TestOutputShapesVIIRS:

    @pytest.mark.unit
    def test_input_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        expected = (VIIRS_INPUT_CHANNELS,) + VIIRS_COV_SIZE
        assert sample["input"].shape == expected

    @pytest.mark.unit
    def test_target_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["target"].shape == (1,) + VIIRS_TARGET_SIZE

    @pytest.mark.unit
    def test_goes_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["goes"].shape == (3,) + VIIRS_GOES_SIZE

    @pytest.mark.unit
    def test_covariates_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["covariates"].shape == (VIIRS_COV_CHANNELS,) + VIIRS_COV_SIZE


class TestDtypes:

    @pytest.mark.unit
    def test_input_is_float32(self, landsat_dataset):
        assert landsat_dataset[0]["input"].dtype == torch.float32

    @pytest.mark.unit
    def test_target_is_float32(self, landsat_dataset):
        assert landsat_dataset[0]["target"].dtype == torch.float32

    @pytest.mark.unit
    def test_goes_is_float32(self, landsat_dataset):
        assert landsat_dataset[0]["goes"].dtype == torch.float32


class TestValueRanges:

    @pytest.mark.unit
    def test_target_is_binary_when_landsat_mask_source(self, landsat_dataset):
        """Landsat target source is already binary; output stays binary regardless of flag."""
        unique = landsat_dataset[0]["target"].unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique), \
            f"Target contains non-binary values: {unique}"

    @pytest.mark.unit
    def test_no_nan_in_input(self, landsat_dataset):
        assert not torch.isnan(landsat_dataset[0]["input"]).any()

    @pytest.mark.unit
    def test_no_nan_in_target(self, landsat_dataset):
        assert not torch.isnan(landsat_dataset[0]["target"]).any()

    @pytest.mark.unit
    def test_landcover_onehot_sums_to_at_most_one(self, landsat_dataset):
        """Each pixel's one-hot land cover channels should sum to 0 or 1."""
        cov = landsat_dataset[0]["covariates"]
        lc_onehot = cov[1:11]  # channels 1-10 are one-hot land cover
        totals = lc_onehot.sum(dim=0)
        assert torch.all((totals == 0) | (totals == 1)), \
            "One-hot land cover channels do not sum to 0 or 1 at every pixel"


class TestMetadata:

    @pytest.mark.unit
    def test_sample_id_is_string(self, landsat_dataset):
        sid = landsat_dataset[0]["sample_id"]
        assert isinstance(sid, str) and len(sid) > 0

    @pytest.mark.unit
    def test_date_is_string(self, landsat_dataset):
        date = landsat_dataset[0]["date"]
        assert isinstance(date, str) and len(date) > 0


class TestGoesValidation:
    """_goes_is_valid still exists as a standalone helper (used elsewhere in the
    pipeline, e.g. enrich_manifests.py), even though _filter_complete no longer
    calls it directly. Tested here in isolation."""

    @pytest.mark.unit
    def test_rejects_all_zero(self, landsat_dataset, landsat_bucket):
        path = "processed_dataset/landsat_pairs/goes_input/all_zero.tif"
        landsat_bucket.blob(path).download_as_bytes.return_value = \
            make_tif_bytes((3, 4, 4), value=0.0)
        assert not landsat_dataset._goes_is_valid(path)

    @pytest.mark.unit
    def test_accepts_valid_goes(self, landsat_dataset, landsat_bucket):
        path = "processed_dataset/landsat_pairs/goes_input/clean.tif"
        landsat_bucket.blob(path).download_as_bytes.return_value = \
            make_tif_bytes((3, 4, 4))
        assert landsat_dataset._goes_is_valid(path)

    @pytest.mark.unit
    def test_rejects_missing_file(self, landsat_dataset, landsat_bucket):
        path = "processed_dataset/landsat_pairs/goes_input/missing.tif"
        landsat_bucket.blob(path).exists.return_value = False
        assert not landsat_dataset._goes_is_valid(path)


class TestCentreCrop:
    """_centre_crop is retained as a utility (not currently called in the
    __getitem__ pipeline, which uses F.interpolate for resizing instead),
    but is tested standalone as it remains part of the public interface."""

    @pytest.mark.unit
    def test_crops_to_correct_size(self, landsat_dataset):
        t   = torch.ones(2, 300, 300)
        out = landsat_dataset._centre_crop(t, (256, 256))
        assert out.shape == (2, 256, 256)

    @pytest.mark.unit
    def test_crop_is_centred(self, landsat_dataset):
        t = torch.zeros(1, 300, 300)
        t[0, 150, 150] = 1.0
        out = landsat_dataset._centre_crop(t, (256, 256))
        assert out[0, 128, 128] == 1.0

    @pytest.mark.unit
    def test_no_op_when_already_correct_size(self, landsat_dataset):
        t   = torch.ones(2, 256, 256)
        out = landsat_dataset._centre_crop(t, (256, 256))
        assert torch.equal(t, out)


class TestFilterComplete:
    """_filter_complete now filters purely on the precomputed 'is_valid'
    manifest flag (set by enrich_manifests.py) and, optionally, on the
    presence of a stage-1 prediction path. It no longer performs any
    GCS blob validity checks itself."""

    @pytest.mark.unit
    def test_keeps_valid_sample(self, landsat_dataset):
        m = make_manifest("valid_sample", "landsat", is_valid=True)
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 0
        assert len(valid) == 1

    @pytest.mark.unit
    def test_drops_invalid_flagged_sample(self, landsat_dataset):
        m = make_manifest("invalid_sample", "landsat", is_valid=False)
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 1
        assert len(valid) == 0

    @pytest.mark.unit
    def test_drops_missing_is_valid_field(self, landsat_dataset):
        """A manifest with no 'is_valid' key at all should be dropped (not True)."""
        m = make_manifest("no_flag_sample", "landsat")
        del m["is_valid"]
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 1
        assert len(valid) == 0

    @pytest.mark.unit
    def test_drops_sample_missing_stage1_key_when_required(self, landsat_dataset):
        landsat_dataset.use_stage1_pred = True
        landsat_dataset.stage1_pred_key = "stage1_pred"
        m = make_manifest("no_stage1", "landsat", is_valid=True)
        # m["paths"] has no "stage1_pred" key
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 1
        assert len(valid) == 0

    @pytest.mark.unit
    def test_keeps_sample_with_stage1_key_when_required(self, landsat_dataset):
        landsat_dataset.use_stage1_pred = True
        landsat_dataset.stage1_pred_key = "stage1_pred"
        m = make_manifest("has_stage1", "landsat", is_valid=True)
        m["paths"]["stage1_pred"] = "some/path.tif"
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 0
        assert len(valid) == 1


# ==================================================================
# INTEGRATION TESTS  (requires GCS credentials and real bucket — currently
# unrunnable following GCP billing expiry; retained for documentation)
# ==================================================================

@pytest.mark.integration
class TestIntegrationLandsat:

    @pytest.fixture(scope="class")
    @classmethod
    def ds(cls):
        return WildfireDataset(
            bucket_name=BUCKET_NAME,
            output_prefix=OUTPUT_PREFIX,
            branch="landsat",
        )

    def test_dataset_non_empty(self, ds):
        assert len(ds) > 0

    def test_first_sample_shapes(self, ds):
        sample = ds[0]
        assert sample["input"].shape      == (LANDSAT_INPUT_CHANNELS, 256, 256)
        assert sample["target"].shape     == (1, 256, 256)
        assert sample["goes"].shape       == (3,) + LANDSAT_GOES_SIZE
        assert sample["covariates"].shape == (LANDSAT_COV_CHANNELS, 256, 256)

    def test_no_nan_in_first_sample(self, ds):
        sample = ds[0]
        assert not torch.isnan(sample["input"]).any()
        assert not torch.isnan(sample["target"]).any()

    def test_target_has_fire_pixels(self, ds):
        has_fire = any(ds[i]["target"].sum() > 0 for i in range(min(10, len(ds))))
        assert has_fire, "No fire pixels found in first 10 samples"


@pytest.mark.integration
class TestIntegrationVIIRS:

    @pytest.fixture(scope="class")
    @classmethod
    def ds(cls):
        return WildfireDataset(
            bucket_name=BUCKET_NAME,
            output_prefix=OUTPUT_PREFIX,
            branch="viirs",
        )

    def test_dataset_non_empty(self, ds):
        assert len(ds) > 0

    def test_first_sample_shapes(self, ds):
        sample = ds[0]
        assert sample["input"].shape      == (VIIRS_INPUT_CHANNELS,) + VIIRS_COV_SIZE
        assert sample["target"].shape     == (1,) + VIIRS_TARGET_SIZE
        assert sample["goes"].shape       == (3,) + VIIRS_GOES_SIZE
        assert sample["covariates"].shape == (VIIRS_COV_CHANNELS,) + VIIRS_COV_SIZE