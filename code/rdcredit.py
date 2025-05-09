# imports
import multiprocessing
from distributed import Client
import os, argparse
import json
import time
import importlib.resources
from importlib.resources import files
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from taxcalc import Calculator
from ogusa.calibrate import Calibration
from ogcore.parameters import Specifications
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.utils import safe_read_pickle

# Use a custom matplotlib style file for plots
style_file_url = (
    "https://raw.githubusercontent.com/PSLmodels/OG-Core/"
    + "master/ogcore/OGcorePlots.mplstyle"
)
plt.style.use(style_file_url)

def main():
    # Define parameters to use for multiprocessing
    num_workers = min(multiprocessing.cpu_count() - 2, 10)
    client = Client(n_workers=num_workers, threads_per_worker=1)
    print("Number of workers = ", num_workers)

    # Directories to save data
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "RDcreditOutput")
    base_dir = os.path.join(save_dir, "OUTPUT_BASELINE")
    reform_dir = os.path.join(save_dir, "OUTPUT_REFORM")
    figtab_dir = os.path.join(save_dir, "tables_figures")
    if not os.path.exists(figtab_dir):
        os.makedirs(figtab_dir)

    # """
    # ---------------------------------------------------------------------------
    # Run baseline policy
    # ---------------------------------------------------------------------------
    # """
    # # Set up baseline parameterization
    # p = Specifications(
    #     baseline=True,
    #     num_workers=num_workers,
    #     baseline_dir=base_dir,
    #     output_base=base_dir,
    # )
    # # Update parameters for baseline from default json file
    # with importlib.resources.open_text(
    #     "ogusa", "ogusa_default_parameters.json"
    # ) as file:
    #     defaults = json.load(file)
    # p.update_specifications(defaults)
    # p.tax_func_type = "HSV"
    # p.age_specific = True

    # # c = Calibration(p, estimate_tax_functions=True, iit_baseline=iit_baseline, data='tmd', client=client)
    # # c = Calibration(p, estimate_tax_functions=True, data='tmd', client=client)
    # tmd_dir = "/Users/richardevans/Docs/Economics/OSE/microsim/tax-microdata-benchmarking/tmd/storage/output"
    # c = Calibration(
    #     p,
    #     estimate_tax_functions=True,
    #     client=client,
    #     data=Path(os.path.join(tmd_dir, "tmd_jason2.csv.gz")),
    #     weights=Path(os.path.join(tmd_dir, "tmd_weights_jason2.csv.gz")),
    #     gfactors=Path(os.path.join(tmd_dir, "tmd_growfactors_jason2.csv")),
    #     records_start_year=2021,
    # )
    # client.close()
    # d = c.get_dict()
    # # # additional parameters to change
    # updated_params = {
    #     "start_year": 2026,
    #     "RC_TPI": 100*1e-4,
    #     "etr_params": d["etr_params"],
    #     "mtrx_params": d["mtrx_params"],
    #     "mtry_params": d["mtry_params"],
    #     "mean_income_data": d["mean_income_data"],
    #     "frac_tax_payroll": d["frac_tax_payroll"],
    # }
    # p.update_specifications(updated_params)

    # # Run model
    # start_time = time.time()
    # client = Client(n_workers=num_workers, threads_per_worker=1)
    # runner(p, time_path=True, client=client)
    # print("run time = ", time.time() - start_time)
    # client.close()

    """
    ---------------------------------------------------------------------------
    Run reform policy
    ---------------------------------------------------------------------------
    """
    client = Client(n_workers=num_workers, threads_per_worker=1)
    # Set up baseline parameterization
    p2 = Specifications(
        baseline=False,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=reform_dir,
    )
    # Update parameters for baseline from default json file
    with importlib.resources.open_text(
        "ogusa", "ogusa_default_parameters.json"
    ) as file:
        defaults = json.load(file)
    p2.update_specifications(defaults)
    p2.tax_func_type = "HSV"
    p2.age_specific = True

    # c = Calibration(p, estimate_tax_functions=True, iit_baseline=iit_baseline, data='tmd', client=client)
    # c = Calibration(p, estimate_tax_functions=True, data='tmd', client=client)
    tmd_dir = "/Users/richardevans/Docs/Economics/OSE/microsim/tax-microdata-benchmarking/tmd/storage/output"
    c2 = Calibration(
        p2,
        estimate_tax_functions=True,
        client=client,
        data=Path(os.path.join(tmd_dir, "tmd_jason2.csv.gz")),
        weights=Path(os.path.join(tmd_dir, "tmd_weights_jason2.csv.gz")),
        gfactors=Path(os.path.join(tmd_dir, "tmd_growfactors_jason2.csv")),
        records_start_year=2021,
    )
    client.close()
    d2 = c2.get_dict()
    # # additional parameters to change
    updated_params2 = {
        "start_year": 2026,
        "RC_TPI": 100*1e-4,
        "baseline_spending": True,
        "inv_tax_credit": [[0.01]],
        "etr_params": d2["etr_params"],
        "mtrx_params": d2["mtrx_params"],
        "mtry_params": d2["mtry_params"],
        "mean_income_data": d2["mean_income_data"],
        "frac_tax_payroll": d2["frac_tax_payroll"],
    }
    p2.update_specifications(updated_params2)

    # Run model
    start_time = time.time()
    client = Client(n_workers=num_workers, threads_per_worker=1)
    runner(p2, time_path=True, client=client)
    print("run time = ", time.time() - start_time)
    client.close()

    """
    ---------------------------------------------------------------------------
    Save some results of simulations
    ---------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )

    # create plots of output
    op.plot_all(
        base_dir,
        reform_dir,
        os.path.join(figtab_dir),
    )
    # Create CSV file with output
    ot.time_series_table(
        base_params,
        base_tpi,
        reform_params,
        reform_tpi,
        table_format="csv",
        path=os.path.join(figtab_dir, "macro_time_series_output.csv"),
    )

    print("Percentage changes in aggregates:", ans)
    # save percentage change output to csv file
    ans.to_csv(
        os.path.join(figtab_dir, "output.csv")
    )


if __name__ == "__main__":
    main()
