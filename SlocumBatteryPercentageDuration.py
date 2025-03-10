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
from scipy.stats import linregress, t
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
grp = parser.add_mutually_exclusive_group()
grp.add_argument("--ndays", type=float, help="Number of days from last date to include")
grp.add_argument("--start", type=str, help="Only use data after this UTC time.")

parser.add_argument("--stop", type=str, help="Only use data before this UTC time.")
parser.add_argument("--threshold", type=float, default=15,
                    help="Percentage at which recovery should happen")
parser.add_argument("--time", type=str, default="time", help="Name of time sensor")
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
        varNames = (args.time, args.sensor)
        for name in varNames:
            if name not in ds:
                logging.error("%s variable not present in %s", name, fn)

        ds = ds.drop_vars(filter(lambda x: x not in varNames, ds))

        if ds[args.time].dtype.kind == "f":
            # Assume posixtime
            for name in ds.dims:
                print("DIM", name)
                ds[name] = ds[args.time].astype("datetime64[s]")
                if name != "time":
                    ds = ds.rename_dims({name: "time"})
                    ds = ds.rename({name: "time"})
                break

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
        elif args.ndays is not None:
            etime = ds.time[-1]
            stime = etime - np.timedelta64(int(args.ndays * 86400), "s")
            ds = ds.sel(time=slice(stime, etime))

        if ds.time.size == 0:
            logging.error("No data to fit in %s", fn)
            continue

        ds["dDays"] = (ds.time.data - ds.time.data[0]).astype(float) / 86400e9

        mdl = linregress(ds.dDays, ds[args.sensor])

        dRecovery = (args.threshold - mdl.intercept) / mdl.slope
        tRecoverBy = ds.time[0].data + dRecovery.astype("timedelta64[D]")
        tRecoverBy = tRecoverBy.astype("datetime64[s]")

        sigmaIntercept = mdl.intercept_stderr
        sigmaSlope = mdl.stderr
        sigmaRecoverBy = np.sqrt(sigmaIntercept**2 + dRecovery**2 * sigmaSlope**2)

        R2 = mdl.rvalue**2

        tinv = lambda p, df: abs(t.ppf(p/2, df)); # Multipler from sigma to 95% confidence

        ts = tinv(0.05, ds.dDays.size - 2)
        ciIntercept = sigmaIntercept * ts
        ciSlope = sigmaSlope * ts
        ciRecoverBy = sigmaRecoverBy * ts

        print("\n", fn)
        print(f"Sensor:            {args.sensor}")
        print(f"Sensor threshold:  {args.threshold}")
        print(f"Intercept (95%):   {mdl.intercept:.4f}+-{ciIntercept:.4f}")
        print(f"Slope (95%):       {mdl.slope:.4f}+-{ciSlope:.4f}")
        print(f"R-squared:         {R2:.4f}")
        print(f"Pvalue:            {mdl.pvalue:.4f}")
        print(f"Recovery By (95%): {tRecoverBy}+-{ciRecoverBy:.2f} (days)")

        if args.plot:
            slope = mdl.slope
            iTit = os.path.basename(fn)
            if slope < 0:
                slope = -slope
                fitTit = f"{mdl.intercept:.1f}-{slope:.2f} * days"
            else:
                fitTit = f"{mdl.intercept:.1f}+{slope:.2f} * days"
            fitTit+= f"\nRecovery by {tRecoverBy}"
            ax = axs[index] if isinstance(axs, np.ndarray) else axs
            print("pre first plot", ax, type(axs))
            ax.plot(ds.time, ds[args.sensor], "o", label=iTit)
            print("pre second plot")
            ax.plot(ds.time, mdl.intercept + mdl.slope * ds.dDays, "r", label=fitTit)
            print("post second plot")
            ax.set_ylabel(args.sensor)
            ax.legend()
            ax.grid()

if args.plot:
    ax = axs[-1] if isinstance(axs, np.ndarray) else axs
    ax.set_xlabel("Time (UTC)")
    plt.title(f"{args.sensor} threshold {args.threshold}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
