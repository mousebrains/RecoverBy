import json

import numpy as np
import pytest
import xarray as xr

from SlocumBatteryPercentageDuration import main


def make_linear_nc(path, start="2025-01-01", n_days=51, batt_start=100.0, batt_end=50.0):
    """Create a NetCDF file with perfectly linear battery decay."""
    times = np.datetime64(start) + np.arange(n_days).astype("timedelta64[D]")
    battery = np.linspace(batt_start, batt_end, n_days)
    ds = xr.Dataset(
        {"m_lithium_battery_relative_charge": ("time", battery)},
        coords={"time": times},
    )
    ds.to_netcdf(path)


def test_linear_decay(tmp_path, capsys):
    """Perfectly linear decay should produce exact recovery date and R²=1."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc)

    # slope = (50 - 100) / 50 = -1 %/day
    # d_recovery = (15 - 100) / -1 = 85 days
    # 2025-01-01 + 85 days = 2025-03-27
    rc = main(["--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "2025-03-27" in out
    assert "R-squared:         1.0000" in out


def test_ndays_filter(tmp_path, capsys):
    """Using --ndays should restrict the fit window."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc, n_days=100)

    rc = main(["--ndays", "10", "--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Slope (95%, /day):" in out
    assert "Intercept (95%):" in out
    assert "R-squared:" in out
    assert "Pvalue:" in out
    assert "Recovery By (95%):" in out


def test_start_stop_filter(tmp_path, capsys):
    """Using --start/--stop should restrict the fit window."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc, n_days=100)

    rc = main(["--start", "2025-01-10", "--stop", "2025-02-10", "--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Slope (95%, /day):" in out
    assert "R-squared:" in out
    assert "Recovery By (95%):" in out


def test_missing_variable(tmp_path, capsys):
    """Missing sensor variable should skip the file and return failure."""
    nc = tmp_path / "test.nc"
    times = np.datetime64("2025-01-01") + np.arange(10).astype("timedelta64[D]")
    ds = xr.Dataset(
        {"wrong_sensor": ("time", np.linspace(100, 90, 10))},
        coords={"time": times},
    )
    ds.to_netcdf(nc)

    rc = main([str(nc)])
    assert rc == 1


def test_multiple_files(tmp_path, capsys):
    """Processing multiple files should report results for each."""
    nc1 = tmp_path / "a.nc"
    nc2 = tmp_path / "b.nc"
    make_linear_nc(nc1)
    make_linear_nc(nc2, batt_start=90, batt_end=45)

    rc = main(["--threshold", "15", str(nc1), str(nc2)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "a.nc" in out
    assert "b.nc" in out
    assert out.count("Recovery By") == 2


def test_too_few_points(tmp_path):
    """Fewer than 3 data points should skip the file and return failure."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc, n_days=2)

    rc = main([str(nc)])
    assert rc == 1


def test_exactly_three_points(tmp_path, capsys):
    """Exactly 3 data points should succeed (minimum valid for linear fit)."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc, n_days=3)

    rc = main(["--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Recovery By" in out


def test_ndays_stop_conflict(tmp_path):
    """--ndays and --stop together should be rejected."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc)

    with pytest.raises(SystemExit, match="2"):
        main(["--ndays", "7", "--stop", "2025-02-01", str(nc)])


def test_plot_all_files_fail(tmp_path):
    """--output with all files failing should not crash."""
    plot_path = tmp_path / "plot.png"
    rc = main(["--output", str(plot_path), str(tmp_path / "nonexistent.nc")])
    assert rc == 1


def test_output_saves_plot(tmp_path):
    """--output should save a plot file."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc)
    plot_path = tmp_path / "plot.png"

    rc = main(["--output", str(plot_path), str(nc)])
    assert rc == 0
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_posixtime_float(tmp_path, capsys):
    """Float epoch seconds should be auto-converted to datetime64."""
    nc = tmp_path / "test.nc"
    # Create epoch seconds without CF metadata so xarray won't decode
    epoch_start = int(
        (np.datetime64("2025-01-01") - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
    )
    times = epoch_start + np.arange(51, dtype=np.float64) * 86400
    battery = np.linspace(100, 50, 51)
    ds = xr.Dataset(
        {"m_lithium_battery_relative_charge": ("time", battery)},
        coords={"time": times},
    )
    ds.to_netcdf(nc)

    rc = main(["--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "2025-03-27" in out


def test_positive_slope(tmp_path, capsys):
    """Increasing battery should warn about past recovery date but succeed."""
    nc = tmp_path / "test.nc"
    # Battery increasing: 50% -> 100%
    make_linear_nc(nc, batt_start=50, batt_end=100)

    rc = main(["--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Recovery By" in out


def test_missing_file(tmp_path):
    """Nonexistent file should be logged and return failure."""
    rc = main([str(tmp_path / "nonexistent.nc")])
    assert rc == 1


def test_constant_data(tmp_path):
    """Constant battery values produce near-zero slope and should fail."""
    nc = tmp_path / "test.nc"
    times = np.datetime64("2025-01-01") + np.arange(20).astype("timedelta64[D]")
    battery = np.full(20, 80.0)
    ds = xr.Dataset(
        {"m_lithium_battery_relative_charge": ("time", battery)},
        coords={"time": times},
    )
    ds.to_netcdf(nc)

    rc = main([str(nc)])
    assert rc == 1


def test_nan_sensor_data(tmp_path, capsys):
    """NaN values in sensor data should be filtered out before fitting."""
    nc = tmp_path / "test.nc"
    times = np.datetime64("2025-01-01") + np.arange(51).astype("timedelta64[D]")
    battery = np.linspace(100, 50, 51)
    # Insert NaN values at known positions
    battery[10] = np.nan
    battery[20] = np.nan
    battery[30] = np.nan
    ds = xr.Dataset(
        {"m_lithium_battery_relative_charge": ("time", battery)},
        coords={"time": times},
    )
    ds.to_netcdf(nc)

    rc = main(["--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Recovery By" in out
    assert "R-squared:" in out


def test_json_output(tmp_path, capsys):
    """--json should produce valid JSON with expected fields."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc)

    rc = main(["--json", "--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    data = json.loads(out)
    assert len(data) == 1
    r = data[0]
    assert r["sensor"] == "m_lithium_battery_relative_charge"
    assert r["threshold"] == 15.0
    assert r["confidence"] == 0.95
    assert "2025-03-27" in r["recovery_date"]
    assert r["r_squared"] == pytest.approx(1.0)
    assert r["slope"] == pytest.approx(-1.0)
    assert r["intercept"] == pytest.approx(100.0)
    assert r["n_points"] == 51


def test_custom_confidence(tmp_path, capsys):
    """Custom confidence level should change output labels and CI width."""
    nc = tmp_path / "test.nc"
    make_linear_nc(nc)

    rc = main(["--confidence", "0.99", "--threshold", "15", str(nc)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Intercept (99%):" in out
    assert "Slope (99%, /day):" in out
    assert "Recovery By (99%):" in out
