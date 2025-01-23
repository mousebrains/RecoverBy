# Estimate when a Slocum glider will need to be recovered by

The main method uses the estimated percentage of battery remaining to predict
when the percentage left will be a given value.

The input is a time series NetCDF with at least two columns:
 - time a CF compliant time of the observation
 - m\_lithium\_battery\_relative\_charge as defined by Slocum masterdata file

A compatible input NetCDF file is generated from my (Slocum_DataHarvester)[https://github.com/mousebrains/ARCTERX_Slocum_DataHarvester] in the sensors files.

A linear regression is done to:

m\_lithium\_battery\_relative\_charge = a + b (time - min(time))/86400

so the slope is in days.
