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
from ogcore.constants import (
    VAR_LABELS,
    ToGDP_LABELS,
)

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
    reform_retro_dir = os.path.join(save_dir, "OUTPUT_REFORM_RETRO")
    reform_noretro_dir = os.path.join(save_dir, "OUTPUT_REFORM_NORETRO")
    figtab_retro_dir = os.path.join(
        save_dir, "OUTPUT_REFORM_RETRO", "tables_figures"
    )
    figtab_noretro_dir = os.path.join(
        save_dir, "OUTPUT_REFORM_NORETRO", "tables_figures"
    )
    figdoc_dir = os.path.join(save_dir, "figs_for_doc")
    if not os.path.exists(figtab_retro_dir):
        os.makedirs(figtab_retro_dir)
    if not os.path.exists(figtab_noretro_dir):
        os.makedirs(figtab_noretro_dir)
    if not os.path.exists(figdoc_dir):
        os.makedirs(figdoc_dir)


    """
    ---------------------------------------------------------------------------
    Run baseline policy
    ---------------------------------------------------------------------------
    """
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
    #     "inv_tax_credit": [[0.015]],
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
    Run reform policy with retroactivity
    ---------------------------------------------------------------------------
    """
    # client = Client(n_workers=num_workers, threads_per_worker=1)
    # # Set up baseline parameterization
    # p2 = Specifications(
    #     baseline=False,
    #     num_workers=num_workers,
    #     baseline_dir=base_dir,
    #     output_base=reform_retro_dir,
    # )
    # # Update parameters for baseline from default json file
    # with importlib.resources.open_text(
    #     "ogusa", "ogusa_default_parameters.json"
    # ) as file:
    #     defaults = json.load(file)
    # p2.update_specifications(defaults)
    # p2.tax_func_type = "HSV"
    # p2.age_specific = True

    # # c = Calibration(p, estimate_tax_functions=True, iit_baseline=iit_baseline, data='tmd', client=client)
    # # c = Calibration(p, estimate_tax_functions=True, data='tmd', client=client)
    # tmd_dir = "/Users/richardevans/Docs/Economics/OSE/microsim/tax-microdata-benchmarking/tmd/storage/output"
    # c2 = Calibration(
    #     p2,
    #     estimate_tax_functions=True,
    #     client=client,
    #     data=Path(os.path.join(tmd_dir, "tmd_jason2.csv.gz")),
    #     weights=Path(os.path.join(tmd_dir, "tmd_weights_jason2.csv.gz")),
    #     gfactors=Path(os.path.join(tmd_dir, "tmd_growfactors_jason2.csv")),
    #     records_start_year=2021,
    # )
    # client.close()
    # d2 = c2.get_dict()
    # # # additional parameters to change
    # updated_params2 = {
    #     "start_year": 2026,
    #     "RC_TPI": 100*1e-4,
    #     "baseline_spending": True,
    #     "inv_tax_credit": [[0.03333], [0.02295]],
    #     "etr_params": d2["etr_params"],
    #     "mtrx_params": d2["mtrx_params"],
    #     "mtry_params": d2["mtry_params"],
    #     "mean_income_data": d2["mean_income_data"],
    #     "frac_tax_payroll": d2["frac_tax_payroll"],
    # }
    # p2.update_specifications(updated_params2)

    # # Run model
    # start_time = time.time()
    # client = Client(n_workers=num_workers, threads_per_worker=1)
    # runner(p2, time_path=True, client=client)
    # print("run time = ", time.time() - start_time)
    # client.close()

    """
    ---------------------------------------------------------------------------
    Run reform policy with no retroactivity
    ---------------------------------------------------------------------------
    """
    # client = Client(n_workers=num_workers, threads_per_worker=1)
    # # Set up baseline parameterization
    # p3 = Specifications(
    #     baseline=False,
    #     num_workers=num_workers,
    #     baseline_dir=base_dir,
    #     output_base=reform_noretro_dir,
    # )
    # # Update parameters for baseline from default json file
    # with importlib.resources.open_text(
    #     "ogusa", "ogusa_default_parameters.json"
    # ) as file:
    #     defaults = json.load(file)
    # p3.update_specifications(defaults)
    # p3.tax_func_type = "HSV"
    # p3.age_specific = True

    # # c = Calibration(p, estimate_tax_functions=True, iit_baseline=iit_baseline, data='tmd', client=client)
    # # c = Calibration(p, estimate_tax_functions=True, data='tmd', client=client)
    # tmd_dir = "/Users/richardevans/Docs/Economics/OSE/microsim/tax-microdata-benchmarking/tmd/storage/output"
    # c3 = Calibration(
    #     p3,
    #     estimate_tax_functions=True,
    #     client=client,
    #     data=Path(os.path.join(tmd_dir, "tmd_jason2.csv.gz")),
    #     weights=Path(os.path.join(tmd_dir, "tmd_weights_jason2.csv.gz")),
    #     gfactors=Path(os.path.join(tmd_dir, "tmd_growfactors_jason2.csv")),
    #     records_start_year=2021,
    # )
    # client.close()
    # d3 = c3.get_dict()
    # # # additional parameters to change
    # updated_params3 = {
    #     "start_year": 2026,
    #     "RC_TPI": 100*1e-4,
    #     "baseline_spending": True,
    #     "inv_tax_credit": [[0.02295]],
    #     "etr_params": d3["etr_params"],
    #     "mtrx_params": d3["mtrx_params"],
    #     "mtry_params": d3["mtry_params"],
    #     "mean_income_data": d3["mean_income_data"],
    #     "frac_tax_payroll": d3["frac_tax_payroll"],
    # }
    # p3.update_specifications(updated_params3)

    # # Run model
    # start_time = time.time()
    # client = Client(n_workers=num_workers, threads_per_worker=1)
    # runner(p3, time_path=True, client=client)
    # print("run time = ", time.time() - start_time)
    # client.close()

    """
    ---------------------------------------------------------------------------
    Save some results of simulations
    ---------------------------------------------------------------------------
    """
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_retro_tpi = safe_read_pickle(
        os.path.join(reform_retro_dir, "TPI", "TPI_vars.pkl")
    )
    reform_retro_params = safe_read_pickle(
        os.path.join(reform_retro_dir, "model_params.pkl")
    )
    reform_noretro_tpi = safe_read_pickle(
        os.path.join(reform_noretro_dir, "TPI", "TPI_vars.pkl")
    )
    reform_noretro_params = safe_read_pickle(
        os.path.join(reform_noretro_dir, "model_params.pkl")
    )
    ans_retro = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_retro_tpi,
        reform_params=reform_retro_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )
    ans_noretro = ot.macro_table(
        base_tpi,
        base_params,
        reform_tpi=reform_noretro_tpi,
        reform_params=reform_noretro_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=base_params.start_year,
    )

    # create plots of output
    op.plot_all(
        base_dir,
        reform_retro_dir,
        os.path.join(figtab_retro_dir),
    )
    op.plot_all(
        base_dir,
        reform_noretro_dir,
        os.path.join(figtab_noretro_dir),
    )
    # Create CSV file with output
    ot.time_series_table(
        base_params,
        base_tpi,
        reform_retro_params,
        reform_retro_tpi,
        table_format="csv",
        path=os.path.join(figtab_retro_dir, "macro_time_series_output.csv"),
    )
    ot.time_series_table(
        base_params,
        base_tpi,
        reform_noretro_params,
        reform_noretro_tpi,
        table_format="csv",
        path=os.path.join(figtab_noretro_dir, "macro_time_series_output.csv"),
    )

    print("")
    print("Percentage changes in aggregates (retro):")
    print(ans_retro)
    print("")
    print("Percentage changes in aggregates (noretro):")
    print(ans_noretro)
    # save percentage change output to csv file
    ans_retro.to_csv(
        os.path.join(figtab_retro_dir, "output.csv")
    )
    ans_noretro.to_csv(
        os.path.join(figtab_noretro_dir, "output.csv")
    )

    # Create plots for document: macro aggregates
    plot_title1 = (
        "Figure 1. Percent change in macroeconomic variables from Section " +
        "174 reform allowing full R&D expensing deductibility: 2026-2045"
    )
    var_list1=["Y", "K", "L"]
    color_list1 = ["blue", "red", "green"]
    start_year=base_params.start_year
    num_years_to_plot=11
    year_vec = np.arange(
        start_year-1, start_year + num_years_to_plot - 1
    ).astype(int)
    start_index = start_year - base_params.start_year
    fig1, ax1 = plt.subplots()
    for i, v in enumerate(var_list1):
        plot_retro_var = (reform_retro_tpi[v] - base_tpi[v]) / base_tpi[v]
        plot_noretro_var = (reform_noretro_tpi[v] - base_tpi[v]) / base_tpi[v]
        plt.plot(
            year_vec,
            np.concatenate(
                (
                    [0.0],
                    plot_retro_var[
                        start_index : start_index + num_years_to_plot - 1
                    ]
                )
            ),
            color=color_list1[i],
            linestyle="-",
            label=VAR_LABELS[v] + ", backdate"
        )
        plt.plot(
            year_vec,
            np.concatenate(
                (
                    [0.0],
                    plot_noretro_var[
                        start_index : start_index + num_years_to_plot - 1
                    ]
                )
            ),
            color=color_list1[i],
            linestyle="--",
            label=VAR_LABELS[v] + ", no backdate"
        )
    plt.xlabel(r"Year $t$")
    plt.ylabel( r"Pct. change")
    # plt.title(plot_title1, fontsize=15)
    ax1.set_yticks(ax1.get_yticks().tolist())
    vals = ax1.get_yticks()
    ax1.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year - 1,
            base_params.start_year + num_years_to_plot - 2
        )
    )
    plt.xticks(
        [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig1_path = os.path.join(figdoc_dir, "fig1_macro.png")
    plt.savefig(fig1_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Create plots for document: fiscal variables
    plot_title2 = (
        "Figure 2. Percent change in fiscal variables from Section " +
        "174 reform allowing full R&D expensing deductibility: 2026-2045"
    )
    var_list2=["D", "total_tax_revenue"]
    color_list2 = ["blue", "red"]
    start_year=base_params.start_year
    num_years_to_plot=11
    year_vec = np.arange(
        start_year-1, start_year + num_years_to_plot-1
    ).astype(int)
    start_index = start_year - base_params.start_year
    fig2, ax2 = plt.subplots()
    for i, v in enumerate(var_list2):
        plot_retro_var = (reform_retro_tpi[v] - base_tpi[v]) / base_tpi[v]
        plot_noretro_var = (reform_noretro_tpi[v] - base_tpi[v]) / base_tpi[v]
        plt.plot(
            year_vec,
            np.concatenate(
                (
                    [0.0],
                    plot_retro_var[
                        start_index : start_index + num_years_to_plot -1
                    ]
                )
            ),
            color=color_list2[i],
            linestyle="-",
            label=VAR_LABELS[v] + ", backdate"
        )
        plt.plot(
            year_vec,
            np.concatenate(
                (
                    [0.0],
                    plot_noretro_var[
                        start_index : start_index + num_years_to_plot -1
                    ]
                )
            ),
            color=color_list2[i],
            linestyle="--",
            label=VAR_LABELS[v] + ", no backdate"
        )
    plt.xlabel(r"Year $t$")
    plt.ylabel( r"Pct. change")
    # plt.title(plot_title2, fontsize=15)
    ax2.set_yticks(ax2.get_yticks().tolist())
    vals = ax2.get_yticks()
    ax2.set_yticklabels(["{:,.2%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year - 1,
            base_params.start_year + num_years_to_plot - 2
        )
    )
    plt.xticks(
        [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig2_path = os.path.join(figdoc_dir, "fig2_fiscal.png")
    plt.savefig(fig2_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Create plots for document: fiscal variables
    plot_title3 = (
        "Change in government debt-to-GDP from Section 174 reform allowing " +
        "full R&D expensing deductibility: 2026-2045"
    )
    start_year=base_params.start_year
    num_years_to_plot=10
    year_vec = np.arange(
        start_year, start_year + num_years_to_plot
    ).astype(int)
    start_index = start_year - base_params.start_year
    fig3, ax3 = plt.subplots()
    plot_var_base = (
        base_tpi["D"][: base_params.T] / base_tpi["Y"][: base_params.T]
    )
    plot_var_reform_retro = (
        reform_retro_tpi["D"][: base_params.T] /
        reform_retro_tpi["Y"][: base_params.T]
    )
    plot_var_reform_noretro = (
        reform_noretro_tpi["D"][: base_params.T] /
        reform_noretro_tpi["Y"][: base_params.T]
    )
    plt.plot(
        year_vec, plot_var_base[start_index: start_index + num_years_to_plot],
        color="blue", linestyle="-", label="Baseline " + ToGDP_LABELS["D"],
    )
    plt.plot(
        year_vec,
        plot_var_reform_retro[start_index: start_index + num_years_to_plot],
        color="red", linestyle="-",
        label="Reform " + ToGDP_LABELS["D"] + ", backdate"
    )
    plt.plot(
        year_vec,
        plot_var_reform_noretro[start_index: start_index + num_years_to_plot],
        color="red", linestyle="--",
        label="Reform " + ToGDP_LABELS["D"] + ", no backdate"
    )
    plt.xlabel(r"Year $t$")
    plt.ylabel( r"Percent of GDP")
    # plt.title(plot_title3, fontsize=15)
    ax3.set_yticks(ax3.get_yticks().tolist())
    vals = ax3.get_yticks()
    ax3.set_yticklabels(["{:,.0%}".format(x) for x in vals])
    plt.xlim(
        (
            base_params.start_year-1,
            base_params.start_year + num_years_to_plot-1)
    )
    plt.xticks(
        [2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]
    )
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig3_path = os.path.join(figdoc_dir, "fig3_debt_gdp.png")
    plt.savefig(fig3_path, bbox_inches="tight", dpi=300)
    plt.close()

    # Create plots for document: individual savings
    op.ability_bar(
        base_tpi,
        base_params,
        reform_retro_tpi,
        reform_retro_params,
        var="b_sp1",
        num_years=10,
        start_year=base_params.start_year,
        path=os.path.join(figdoc_dir, "fig4_indiv_save.png"),
    )


if __name__ == "__main__":
    main()
