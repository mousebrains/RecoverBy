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

Install required dependencies:

```bash
pip install numpy xarray scipy matplotlib
```

## Usage

Basic usage:

```bash
./SlocumBatteryPercentageDuration.py sensor_data.nc
```

With plotting enabled:

```bash
./SlocumBatteryPercentageDuration.py --plot sensor_data.nc
```

Using only the last N days of data:

```bash
./SlocumBatteryPercentageDuration.py --ndays 7 sensor_data.nc
```

Specifying a custom threshold:

```bash
./SlocumBatteryPercentageDuration.py --threshold 10 sensor_data.nc
```

Using a specific time range:

```bash
./SlocumBatteryPercentageDuration.py --start "2025-01-01" --stop "2025-01-15" sensor_data.nc
```

For verbose output:

```bash
./SlocumBatteryPercentageDuration.py --verbose sensor_data.nc
```
