# Estimate when a Slocum glider will need to be recovered by

The main method uses the estimated percentage of battery remaining to predict
when the percentage left will be a given value.

The input is a time series NetCDF with at least two columns:
 - time a CF compliant time of the observation
 - m\_lithium\_battery\_relative\_charge as defined by Slocum masterdata file

A compatible input NetCDF file is generated from my [Slocum_DataHarvester](https://github.com/mousebrains/ARCTERX_Slocum_DataHarvester) in the sensors files.

A linear regression is done to:

m\_lithium\_battery\_relative\_charge = a + b (time - min(time))/86400

so the slope is in days.

## Installation

```bash
pip install .
```

This also installs a `recover-by` console command.

For development (includes ruff, mypy, pytest, pre-commit):

```bash
pip install ".[dev]"
pre-commit install
pre-commit install --hook-type pre-push
```

## Usage

Basic usage:

```bash
recover-by sensor_data.nc
```

With plotting enabled:

```bash
recover-by --plot sensor_data.nc
```

Save plot to a file (useful on headless/remote servers):

```bash
recover-by --output recovery.png sensor_data.nc
```

Machine-readable JSON output:

```bash
recover-by --json sensor_data.nc
```

Using only the last N days of data:

```bash
recover-by --ndays 7 sensor_data.nc
```

Specifying a custom threshold:

```bash
recover-by --threshold 10 sensor_data.nc
```

Using a custom confidence level (default is 0.95):

```bash
recover-by --confidence 0.99 sensor_data.nc
```

Using a specific time range:

```bash
recover-by --start "2025-01-01" --stop "2025-01-15" sensor_data.nc
```

For verbose output:

```bash
recover-by --verbose sensor_data.nc
```

The script can also be run directly:

```bash
./SlocumBatteryPercentageDuration.py sensor_data.nc
```
