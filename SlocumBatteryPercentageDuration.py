#! /usr/bin/env python3
#
# Calculate when a Slocum glider will need to be recovered by 
# from the battery percentage used.
#
# There are a lot of assumptions here, such as constant in time usage.
# If your operation is changing modes, please don't use this, or
# set the start/end times appropriately.
#
# Jan-2025, Pat Welch

from argparse import ArgumentParser
import numpy as np
import xarray as xr
import logging
from scipy.stats import linregress
from matplotlib import pyplot as plt
import os

parser = ArgumentParser(
        description="Slocum recover by estimates",
        epilog="""
Calculate when a Slocum glider will need to be recovered by 
from the battery percentage used.
There are a lot of assumptions here, such as constant in time usage.
If your operation is changing modes, please don't use this, or
set the start/end times appropriately.
""",
        )
parser.add_argument("--verbose", action="store_true", help="Enable debugging messages")
parser.add_argument("filename", type=str, nargs="+", 
                    help="Input filename(s) containing the" 
                    + " m_present_time and sensor variables")
parser.add_argument("--sensor", type=str, default="m_lithium_battery_relative_charge",
                    help="Sensor name to fit to")
parser.add_argument("--start", type=str, help="Only use data after this UTC time.")
parser.add_argument("--stop", type=str, help="Only use data before this UTC time.")
parser.add_argument("--threshold", type=float, default=15,
                    help="Percentage at which recovery should happen")
parser.add_argument("--plot", action="store_true", help="Generate a plot")
args = parser.parse_args()

logging.basicConfig(
        level = logging.DEBUG if args.verbose else logging.INFO,
        format = "%(asctime)s %(levelname)s: %(message)s",
        )

if args.plot:
    [fig, axs] = plt.subplots(len(args.filename), 1, sharex=True)
    fig.subplots_adjust(hspace=0)

for index in range(len(args.filename)):
    fn = args.filename[index]
    with xr.open_dataset(fn) as ds:
        varNames = ("time", args.sensor)
        for name in varNames:
            if name not in ds:
                logging.error("time variable not present in %s", fn)
                continue
        ds = ds.drop_vars(filter(lambda x: x not in varNames, ds))
        ds = ds.drop_duplicates("time", keep="first")
        ds = ds.sel(time=ds.time[np.logical_not(ds[args.sensor].isnull())])
        ds = ds.sortby("time") # here before slicing
        if args.start is not None or args.stop is not None:
            if args.start is not None:
                stime = np.datetime64(args.start)
            else:
                stime = ds.time[0]
            if args.stop is not None:
                etime = np.datetime64(args.stop)
            else:
                etime = ds.time[-1]
            ds = ds.sel(time=slice(stime, etime))

        if ds.time.size == 0:
            logging.error("No data to fit in %s", fn)
            continue

        ds["dDays"] = (ds.time.data - ds.time.data[0]).astype(float) / 86400e9

        mdl = linregress(ds.dDays, ds[args.sensor])

        dRecovery = (args.threshold - mdl.intercept) / mdl.slope
        tRecoverBy = ds.time[0].data + dRecovery.astype("timedelta64[D]")
        tRecoverBy = tRecoverBy.astype("datetime64[s]")

        print("\n", fn)
        print(f"Intercept: {mdl.intercept:.4f}+-{mdl.intercept_stderr:.4f}")
        print(f"Slope:     {mdl.slope:.4f}+-{mdl.stderr:.4f}")
        print(f"Rvalue:    {mdl.rvalue:.4f}")
        print(f"Pvalue:    {mdl.pvalue:.4f}")
        print(f"Recovery By: {tRecoverBy}")

        if args.plot:
            slope = -mdl.slope
            iTit = os.path.basename(fn) + " " + args.sensor
            fitTit = f"{mdl.intercept:.1f}-{slope:.2f} * days"
            fitTit+= f"\nRecovery by {tRecoverBy}"
            ax = axs[index]
            ax.plot(ds.time, ds[args.sensor], "o", label=iTit)
            ax.plot(ds.time, mdl.intercept + mdl.slope * ds.dDays, "r", label=fitTit)
            ax.set_ylabel(args.sensor)
            ax.legend()
            ax.grid()

if args.plot:
    ax = axs[-1]
    ax.set_xlabel("Time (UTC)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
