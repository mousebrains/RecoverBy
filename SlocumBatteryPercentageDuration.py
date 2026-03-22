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
import json
import logging
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt
import xarray as xr

S_PER_DAY = 86400  # seconds per day
ONE_DAY = np.timedelta64(1, "D")


def _safe_sqrt(x):
    """sqrt that propagates NaN and clamps negative values to 0."""
    if np.isnan(x):
        return float("nan")
    return float(np.sqrt(max(0, x)))


def main(argv=None):
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
    parser.add_argument(
        "filename",
        type=str,
        nargs="+",
        help="Input filename(s) containing the" + " time and sensor variables",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        default="m_lithium_battery_relative_charge",
        help="Sensor name to fit to",
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--ndays", type=float, help="Number of days from last date to include")
    grp.add_argument("--start", type=str, help="Only use data after this UTC time.")
    parser.add_argument(
        "--stop", type=str, help="Only use data before this UTC time (cannot be used with --ndays)"
    )
    parser.add_argument(
        "--threshold", type=float, default=15, help="Percentage at which recovery should happen"
    )
    parser.add_argument("--time", type=str, default="time", help="Name of time sensor")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true", help="Output results as JSON"
    )
    parser.add_argument("--plot", action="store_true", help="Generate a plot")
    parser.add_argument("--output", type=str, help="Save plot to file instead of displaying")
    args = parser.parse_args(argv)

    if args.ndays is not None and args.stop is not None:
        parser.error("--ndays and --stop cannot be used together")

    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between 0 and 1")

    alpha = 1 - args.confidence
    ci_pct = f"{args.confidence * 100:g}"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if args.plot or args.output:
        fig, axs = plt.subplots(len(args.filename), 1, sharex=True, squeeze=False)
        fig.subplots_adjust(hspace=0)

    success = False
    results = []
    plotted_indices = set()

    for index, fn in enumerate(args.filename):
        try:
            with xr.open_dataset(fn) as ds:
                var_names = (args.time, args.sensor)
                missing = [name for name in var_names if name not in ds]
                if missing:
                    for name in missing:
                        logging.error("%s variable not present in %s", name, fn)
                    continue

                ds = ds.drop_vars(set(ds) - set(var_names))

                if ds[args.time].dtype.kind == "f":
                    # Assume posixtime
                    for name in ds.dims:
                        logging.debug("Processing dimension: %s", name)
                        ds[name] = ds[args.time].astype("datetime64[s]")
                        if name != "time":
                            ds = ds.rename_dims({name: "time"})
                            ds = ds.rename({name: "time"})
                        break

                ds = ds.drop_duplicates("time", keep="first")
                ds = ds.sel(time=ds.time[np.logical_not(ds[args.sensor].isnull())])
                ds = ds.sortby("time")

                if args.start is not None or args.stop is not None:
                    stime = np.datetime64(args.start) if args.start is not None else ds.time[0]
                    etime = np.datetime64(args.stop) if args.stop is not None else ds.time[-1]
                    ds = ds.sel(time=slice(stime, etime))
                elif args.ndays is not None:
                    etime = ds.time[-1]
                    stime = etime - np.timedelta64(int(args.ndays * S_PER_DAY), "s")
                    ds = ds.sel(time=slice(stime, etime))

                if ds.time.size < 3:
                    logging.error(
                        "Not enough data to fit in %s (%d points, need >= 3)", fn, ds.time.size
                    )
                    continue

                ds["dDays"] = ("time", (ds.time.data - ds.time.data[0]) / ONE_DAY)

                # Linear fit: sensor = intercept + slope * dDays
                coeffs, cov = np.polyfit(ds.dDays, ds[args.sensor], 1, cov=True)
                slope, intercept = coeffs
                # cov[0,0]=Var(slope), cov[1,1]=Var(intercept), cov[0,1]=Cov(slope, intercept)

                if abs(slope) < 1e-10:
                    logging.error("Near-zero slope in %s — cannot estimate recovery date", fn)
                    continue

                d_recovery = (args.threshold - intercept) / slope
                t_recover_by = ds.time[0].data + np.timedelta64(round(d_recovery * S_PER_DAY), "s")
                t_recover_by = t_recover_by.astype("datetime64[s]")

                if d_recovery < 0:
                    logging.warning(
                        "Recovery date is in the past for %s "
                        "(positive slope — battery increasing?)",
                        fn,
                    )

                # Validate covariance matrix
                if not np.all(np.isfinite(cov)):
                    logging.warning(
                        "Unstable fit for %s — confidence intervals may be unreliable", fn
                    )

                # Propagate uncertainty including covariance between slope and intercept
                # d_recovery = (threshold - intercept) / slope
                # ∂d/∂intercept = -1/slope, ∂d/∂slope = -d_recovery/slope
                var_recovery = (
                    cov[1, 1] + d_recovery**2 * cov[0, 0] + 2 * d_recovery * cov[0, 1]
                ) / slope**2
                sigma_recovery = _safe_sqrt(var_recovery)

                sigma_intercept = _safe_sqrt(cov[1, 1])
                sigma_slope = _safe_sqrt(cov[0, 0])

                # R-squared
                y_pred = intercept + slope * ds.dDays
                ss_res = np.sum((ds[args.sensor] - y_pred) ** 2).item()
                ss_tot = np.sum((ds[args.sensor] - ds[args.sensor].mean()) ** 2).item()
                if ss_tot == 0:
                    r_squared = float("nan")
                    logging.warning("Constant sensor values in %s — R² undefined", fn)
                else:
                    r_squared = 1 - ss_res / ss_tot

                # p-value for slope
                n = ds.dDays.size
                df = n - 2
                if sigma_slope > 0:
                    t_stat = slope / sigma_slope
                    pvalue = 2 * (1 - t.cdf(abs(t_stat), df))
                else:
                    pvalue = float("nan")

                # Confidence intervals
                ts = abs(t.ppf(alpha / 2, df))
                ci_intercept = sigma_intercept * ts
                ci_slope = sigma_slope * ts
                ci_recovery = sigma_recovery * ts

                if not args.json_output:
                    print(f"\n{fn}")
                    print(f"Sensor:            {args.sensor}")
                    print(f"Sensor threshold:  {args.threshold}")
                    print(f"Intercept ({ci_pct}%):   {intercept:.4f}+-{ci_intercept:.4f}")
                    print(f"Slope ({ci_pct}%, /day):  {slope:.4f}+-{ci_slope:.4f}")
                    print(f"R-squared:         {r_squared:.4f}")
                    print(f"Pvalue:            {pvalue:.4f}")
                    print(f"Recovery By ({ci_pct}%): {t_recover_by}+-{ci_recovery:.2f} (days)")

                results.append(
                    {
                        "file": fn,
                        "sensor": args.sensor,
                        "threshold": args.threshold,
                        "confidence": args.confidence,
                        "n_points": int(n),
                        "intercept": float(intercept),
                        "intercept_ci": float(ci_intercept) if np.isfinite(ci_intercept) else None,
                        "slope": float(slope),
                        "slope_ci": float(ci_slope) if np.isfinite(ci_slope) else None,
                        "r_squared": float(r_squared) if np.isfinite(r_squared) else None,
                        "pvalue": float(pvalue) if np.isfinite(pvalue) else None,
                        "recovery_date": str(t_recover_by),
                        "recovery_ci_days": float(ci_recovery)
                        if np.isfinite(ci_recovery)
                        else None,
                    }
                )

                success = True

                if args.plot or args.output:
                    abs_slope = abs(slope)
                    input_title = Path(fn).name
                    if slope < 0:
                        fit_title = f"{intercept:.1f}-{abs_slope:.2f} * days"
                    else:
                        fit_title = f"{intercept:.1f}+{abs_slope:.2f} * days"
                    fit_title += f"\nRecovery by {t_recover_by}"
                    ax = axs[index, 0]
                    plotted_indices.add(index)
                    logging.debug("Plotting: ax=%s, axs type=%s", ax, type(axs))
                    ax.plot(ds.time, ds[args.sensor], "o", label=input_title)
                    ax.plot(ds.time, intercept + slope * ds.dDays, "r", label=fit_title)
                    # Extend fit line to recovery date
                    if t_recover_by > ds.time[-1].values:
                        last_fit_val = float(intercept + slope * ds.dDays[-1].item())
                        ax.plot(
                            [ds.time[-1].values, t_recover_by],
                            [last_fit_val, args.threshold],
                            "r--",
                            alpha=0.5,
                        )
                    ax.axhline(y=args.threshold, color="gray", linestyle="--", alpha=0.5)
                    ax.set_ylabel(args.sensor)
                    ax.legend()
                    ax.grid()

        except Exception as e:
            logging.error("Failed to read %s: %s", fn, e)
            continue

    if args.json_output:
        print(json.dumps(results, indent=2))

    if args.plot or args.output:
        # Remove blank subplots for files that failed
        for i in range(len(args.filename)):
            if i not in plotted_indices:
                fig.delaxes(axs[i, 0])
        if plotted_indices:
            ax = fig.axes[-1]
            ax.set_xlabel("Time (UTC)")
            plt.title(f"{args.sensor} threshold {args.threshold}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            if args.output:
                plt.savefig(args.output)
                logging.info("Plot saved to %s", args.output)
            else:
                plt.show()
        plt.close(fig)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
