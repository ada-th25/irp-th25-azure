import io
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
# Constants
# ------------------------------------------------------------------
LANDSAT_TARGET_SIZE = (256, 256)
LANDSAT_COV_SIZE    = (256, 256)
LANDSAT_GOES_SIZE   = (15, 15)
VIIRS_TARGET_SIZE   = (20, 20)
VIIRS_COV_SIZE      = (20, 20)
VIIRS_GOES_SIZE     = (15, 15)
 
BUCKET_NAME   = "wildfire-irp-data-th25"
OUTPUT_PREFIX = "processed_dataset"
 
 
# ------------------------------------------------------------------
# Helpers: build synthetic GeoTIFFs in memory
# ------------------------------------------------------------------
 
def make_tif_bytes(shape, dtype=np.float32, value=None, nodata=None):
    """
    Return bytes of a valid GeoTIFF with the given (C, H, W) shape.
    Fills with `value` if given, otherwise random data in [200, 330]
    (plausible GOES brightness temperature range in Kelvin).
    """
    c, h, w = shape
    if value is not None:
        data = np.full((c, h, w), value, dtype=dtype)
    else:
        rng  = np.random.default_rng(42)
        data = rng.uniform(200, 330, size=(c, h, w)).astype(dtype)
 
    transform = from_bounds(0, 0, 1, 1, w, h)
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=h, width=w, count=c,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return buf.getvalue()
 
 
def make_binary_mask_bytes(shape):
    """Binary fire mask: mostly 0 with a small number of 1s."""
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
 
 
def make_stripy_goes_bytes(shape=(3, 15, 15), zero_rows=4):
    """GOES tile with `zero_rows` fully-zeroed rows — simulates scan dropout."""
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
 
 
def make_manifest(sample_id="test_sample_001", branch="landsat"):
    return {
        "id":   sample_id,
        "date": "2020-08-11",
        "paths": {
            "goes_input":  f"processed_dataset/{branch}_pairs/goes_input/{sample_id}.tif",
            "target":      f"processed_dataset/{branch}_pairs/target/{sample_id}.tif",
            "covariates":  f"processed_dataset/{branch}_pairs/covariates/{sample_id}.tif",
        },
        "n_active_pixels": 100,
    }
 
 
# ------------------------------------------------------------------
# Mock bucket factory
# ------------------------------------------------------------------
 
def make_mock_bucket(branch="landsat"):
    """
    Returns a mock GCS bucket with a stable blob registry.
 
    The key design: bucket.blob(path) always returns the SAME mock object
    for a given path (stored in a dict), so tests can configure a blob
    once and have _goes_is_valid / _filter_complete see the same object.
    """
    bucket  = MagicMock()
    registry = {}  # path → blob mock
 
    if branch == "landsat":
        target_shape = (1, 256, 256)
        cov_shape    = (2, 256, 256)
    else:
        target_shape = (1, 20, 20)
        cov_shape    = (2, 20, 20)
 
    def get_blob(path):
        if path not in registry:
            blob = MagicMock()
            blob.exists.return_value = True
            blob.name = path
 
            if "goes_input" in path:
                blob.download_as_bytes.return_value = make_tif_bytes((3, 16, 15))
            elif "target" in path:
                blob.download_as_bytes.return_value = make_binary_mask_bytes(target_shape)
            elif "covariates" in path:
                blob.download_as_bytes.return_value = make_tif_bytes(cov_shape)
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
 
def _make_dataset(branch, mock_bucket):
    ds = WildfireDataset.__new__(WildfireDataset)
    ds.bucket        = mock_bucket
    ds.branch        = branch
    ds.output_prefix = OUTPUT_PREFIX
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
        assert sample["input"].shape == (5, 256, 256), \
            f"Expected (5,256,256), got {sample['input'].shape}"
 
    @pytest.mark.unit
    def test_target_shape(self, landsat_dataset):
        sample = landsat_dataset[0]
        assert sample["target"].shape == (1, 256, 256), \
            f"Expected (1,256,256), got {sample['target'].shape}"
 
    @pytest.mark.unit
    def test_goes_shape(self, landsat_dataset):
        """Raw GOES in the sample dict is at native low-res size, not upsampled."""
        sample = landsat_dataset[0]
        assert sample["goes"].shape == (3, 15, 15), \
            f"Expected (3,15,15), got {sample['goes'].shape}"
 
    @pytest.mark.unit
    def test_covariates_shape(self, landsat_dataset):
        sample = landsat_dataset[0]
        assert sample["covariates"].shape == (2, 256, 256), \
            f"Expected (2,256,256), got {sample['covariates'].shape}"
 
    @pytest.mark.unit
    def test_input_channels_are_goes_plus_covariates(self, landsat_dataset):
        """input[3:] must match the covariates tensor exactly."""
        sample = landsat_dataset[0]
        assert torch.allclose(sample["input"][3:], sample["covariates"]), \
            "input[3:] does not match covariates tensor"
 
 
class TestOutputShapesVIIRS:
 
    @pytest.mark.unit
    def test_input_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["input"].shape == (5, 20, 20), \
            f"Expected (5,20,20), got {sample['input'].shape}"
 
    @pytest.mark.unit
    def test_target_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["target"].shape == (1, 20, 20), \
            f"Expected (1,20,20), got {sample['target'].shape}"
 
    @pytest.mark.unit
    def test_goes_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["goes"].shape == (3, 15, 15), \
            f"Expected (3,15,15), got {sample['goes'].shape}"
 
    @pytest.mark.unit
    def test_covariates_shape(self, viirs_dataset):
        sample = viirs_dataset[0]
        assert sample["covariates"].shape == (2, 20, 20), \
            f"Expected (2,20,20), got {sample['covariates'].shape}"
 
 
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
    def test_target_is_binary(self, landsat_dataset):
        """Fire mask must contain only 0 and 1."""
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
    def test_goes_brightness_temperature_plausible(self, landsat_dataset):
        """
        Non-zero GOES values should be in a physically plausible BT range.
        3.9µm channel: ~200 K (cold cloud tops) to ~330 K (hot surfaces).
        """
        goes    = landsat_dataset[0]["goes"]
        nonzero = goes[goes != 0]
        if nonzero.numel() > 0:
            assert nonzero.min() >= 150, \
                f"GOES BT below 150 K — likely unphysical sentinel: {nonzero.min()}"
            assert nonzero.max() <= 400, \
                f"GOES BT above 400 K — likely calibration error: {nonzero.max()}"
 
 
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
    """_goes_is_valid should correctly reject bad GOES tiles."""
 
    @pytest.mark.unit
    def test_rejects_all_zero(self, landsat_dataset, landsat_bucket):
        path = "processed_dataset/landsat_pairs/goes_input/all_zero.tif"
        landsat_bucket.blob(path).download_as_bytes.return_value = \
            make_tif_bytes((3, 15, 15), value=0.0)
        assert not landsat_dataset._goes_is_valid(path)
 
    @pytest.mark.unit
    def test_rejects_high_zero_fraction(self, landsat_dataset, landsat_bucket):
        """GOES with >5% exact zeros (scan-line dropout) must be rejected."""
        path = "processed_dataset/landsat_pairs/goes_input/stripy.tif"
        landsat_bucket.blob(path).download_as_bytes.return_value = \
            make_stripy_goes_bytes((3, 15, 15), zero_rows=4)
        assert not landsat_dataset._goes_is_valid(path)
 
    @pytest.mark.unit
    def test_accepts_valid_goes(self, landsat_dataset, landsat_bucket):
        path = "processed_dataset/landsat_pairs/goes_input/clean.tif"
        landsat_bucket.blob(path).download_as_bytes.return_value = \
            make_tif_bytes((3, 15, 15))   # random 200–330 K, no zeros
        assert landsat_dataset._goes_is_valid(path)
 
    @pytest.mark.unit
    def test_rejects_missing_file(self, landsat_dataset, landsat_bucket):
        path = "processed_dataset/landsat_pairs/goes_input/missing.tif"
        landsat_bucket.blob(path).exists.return_value = False
        assert not landsat_dataset._goes_is_valid(path)
 
 
class TestCentreCrop:
 
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
 
    @pytest.mark.unit
    def test_drops_missing_target(self, landsat_dataset, landsat_bucket):
        m = make_manifest("missing_target", "landsat")
        landsat_bucket.blob(m["paths"]["target"]).exists.return_value = False
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 1
        assert len(valid) == 0
 
    @pytest.mark.unit
    def test_drops_all_zero_target(self, landsat_dataset, landsat_bucket):
        m = make_manifest("zero_target", "landsat")
        landsat_bucket.blob(m["paths"]["target"]).download_as_bytes.return_value = \
            make_tif_bytes((1, 256, 256), value=0.0)
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 1
        assert len(valid) == 0
 
    @pytest.mark.unit
    def test_keeps_valid_sample(self, landsat_dataset, landsat_bucket):
        m = make_manifest("valid_sample", "landsat")
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 0
        assert len(valid) == 1
 
    @pytest.mark.unit
    def test_drops_stripy_goes(self, landsat_dataset, landsat_bucket):
        m = make_manifest("stripy_goes", "landsat")
        landsat_bucket.blob(m["paths"]["goes_input"]).download_as_bytes.return_value = \
            make_stripy_goes_bytes((3, 15, 15), zero_rows=4)
        valid, dropped = landsat_dataset._filter_complete([m])
        assert dropped == 1
        assert len(valid) == 0
 
 
# ==================================================================
# INTEGRATION TESTS  (requires GCS credentials and real bucket)
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
        assert sample["input"].shape      == (5, 256, 256)
        assert sample["target"].shape     == (1, 256, 256)
        assert sample["goes"].shape       == (3, 15, 15)
        assert sample["covariates"].shape == (2, 256, 256)
 
    def test_no_nan_in_first_sample(self, ds):
        sample = ds[0]
        assert not torch.isnan(sample["input"]).any()
        assert not torch.isnan(sample["target"]).any()
 
    def test_target_has_fire_pixels(self, ds):
        has_fire = any(ds[i]["target"].sum() > 0 for i in range(min(10, len(ds))))
        assert has_fire, "No fire pixels found in first 10 samples"
 
    def test_goes_values_are_plausible(self, ds):
        for i in range(min(5, len(ds))):
            goes = ds[i]["goes"]
            # Channel 0 = C07 (3.9µm BT), Channel 1 = C14 (11.2µm BT)
            # Channel 2 = TFI (C07 - C14), different value range
            bt_channels = goes[:2]   # (2, 15, 15)
            nonzero = bt_channels[bt_channels != 0]
            if nonzero.numel() > 0:
                assert nonzero.min() >= 150, \
                    f"Sample {i}: BT channel below 150 K ({nonzero.min():.1f})"
                assert nonzero.max() <= 400, \
                    f"Sample {i}: BT channel above 400 K ({nonzero.max():.1f})"
    
            # TFI (C07 - C14) should be small, large negative values or
            # values above ~50 K difference would indicate a calibration issue
            tfi = goes[2]
            nonzero_tfi = tfi[tfi != 0]
            if nonzero_tfi.numel() > 0:
                assert nonzero_tfi.min() >= -50, \
                    f"Sample {i}: TFI below -50 K ({nonzero_tfi.min():.1f})"
                assert nonzero_tfi.max() <= 50, \
                    f"Sample {i}: TFI above 50 K ({nonzero_tfi.max():.1f})"
 
 
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
        assert sample["input"].shape      == (5, 20, 20)
        assert sample["target"].shape     == (1, 20, 20)
        assert sample["goes"].shape       == (3, 15, 15)
        assert sample["covariates"].shape == (2, 20, 20)