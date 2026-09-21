# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest
import tifffile

from hmsm.io import ArraySource, TiffSource, open_source


@pytest.fixture
def scan():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (523, 137, 3), dtype=np.uint8)


def write(tmp_path, scan, name="scan.tif", **kwargs):
    path = tmp_path / name
    tifffile.imwrite(path, scan, **kwargs)
    return str(path)


@pytest.mark.parametrize(
    "options",
    [
        {"compression": "lzw", "rowsperstrip": 1},
        {"compression": "lzw", "rowsperstrip": 25},
        {"compression": "deflate", "rowsperstrip": 84, "predictor": True},
        {"tile": (64, 64)},
        {},
    ],
    ids=["strips-of-1", "strips-of-25", "deflate-predictor", "tiled", "default"],
)
def test_streaming_reproduces_the_scan_exactly(tmp_path, scan, options):
    source = open_source(write(tmp_path, scan, **options))
    assert source.shape == scan.shape
    np.testing.assert_array_equal(source.read_rows(0, source.height), scan)
    source.close()


@pytest.mark.parametrize("band_height", [1, 7, 84, 523, 10_000])
def test_bands_cover_the_scan_in_order(tmp_path, scan, band_height):
    with open_source(write(tmp_path, scan, rowsperstrip=25)) as source:
        bands = list(source.bands(band_height))
    assert [bounds[0] for bounds in (b for b, _ in bands)] == list(
        range(0, 523, band_height)
    )
    np.testing.assert_array_equal(np.vstack([pixels for _, pixels in bands]), scan)


def test_random_access_matches_the_scan(tmp_path, scan):
    with open_source(write(tmp_path, scan, rowsperstrip=25)) as source:
        for start, stop in [(0, 1), (24, 26), (100, 100), (500, 600), (7, 333)]:
            np.testing.assert_array_equal(
                source.read_rows(start, stop), scan[start:stop]
            )


def test_reading_past_the_end_is_clipped(tmp_path, scan):
    with open_source(write(tmp_path, scan)) as source:
        assert len(source.read_rows(1000, 2000)) == 0


def test_partial_ranges(tmp_path, scan):
    with open_source(write(tmp_path, scan, rowsperstrip=25)) as source:
        bands = list(source.bands(50, start=100, stop=300))
    np.testing.assert_array_equal(
        np.vstack([pixels for _, pixels in bands]), scan[100:300]
    )


def test_resolution_is_read_from_the_scan(tmp_path, scan):
    path = write(tmp_path, scan, resolution=(300, 300), resolutionunit="INCH")
    with open_source(path) as source:
        assert source.dpi == pytest.approx(300)


def test_centimetre_resolution_is_converted(tmp_path, scan):
    path = write(tmp_path, scan, resolution=(100, 100), resolutionunit="CENTIMETER")
    with open_source(path) as source:
        assert source.dpi == pytest.approx(254)


def test_alpha_channels_are_dropped(tmp_path):
    rgba = np.zeros((40, 40, 4), np.uint8)
    rgba[..., 3] = 255
    with open_source(write(tmp_path, rgba, name="rgba.tif")) as source:
        assert source.read_rows(0, 40).shape[2] == 3


def test_non_tiff_formats_fall_back_to_reading_in_memory(tmp_path, scan):
    import skimage.io

    path = tmp_path / "scan.png"
    skimage.io.imsave(str(path), scan)
    source = open_source(str(path))
    assert isinstance(source, ArraySource)
    np.testing.assert_array_equal(source.read_rows(0, source.height), scan)


def test_a_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "absent.tif"))


def test_array_source_exposes_the_same_interface(scan):
    source = ArraySource(scan, dpi=300)
    assert source.shape == scan.shape and source.dpi == 300
    np.testing.assert_array_equal(
        np.vstack([pixels for _, pixels in source.bands(100)]), scan
    )
