# main.py
import os
import re
import pandas as pd
from gams import transfer as gt

from helpers_priv import (
    Extrapolation,
    net_cashflow_full_pipeline,
    plot_cumulative_net_cashflow_all_sites,
)

# -------------------------
# utils function
# -------------------------
def reorder_and_separate(wide, sites, commodities):
    def is_cost_col(name: str) -> bool:
        if name.startswith("Revenue "):
            return False
        if name in ("Net Cash Flow [MCHF]", "Cumulative Net Cash Flow [MCHF]"):
            return False
        if name in ("Total Revenue [MCHF]", "Total Costs [MCHF]"):
            return False
        return any(k in name for k in ["Cost", "Costs", "Investment", "Decomm", "Uranium", "O&M", "tax", "subsid"])

    new_cols = []
    for i, s in enumerate(sites):
        site_cols = [c for c in wide.columns if c[0] == s]
        names = [c[1] for c in site_cols]

        disc = [(s, "Discount Factor")] if "Discount Factor" in names else []
        cost_cols = [c for c in site_cols if is_cost_col(c[1])]

        rev = []
        for com in commodities:
            col = (s, f"Revenue {com} [MCHF]")
            if col in site_cols:
                rev.append(col)
        if (s, "Total Revenue [MCHF]") in site_cols:
            rev.append((s, "Total Revenue [MCHF]"))

        tot_cost = [(s, "Total Costs [MCHF]")] if "Total Costs [MCHF]" in names else []
        net = [(s, "Net Cash Flow [MCHF]")] if "Net Cash Flow [MCHF]" in names else []
        cum = [(s, "Cumulative Net Cash Flow [MCHF]")] if "Cumulative Net Cash Flow [MCHF]" in names else []

        chosen = set(disc + cost_cols + rev + tot_cost + net + cum)
        remaining = [c for c in site_cols if c not in chosen]

        new_cols.extend(disc + cost_cols + rev + tot_cost + net + cum + remaining)
        if i < len(sites) - 1:
            sep = ("  ", f" ")
            wide[sep] = "  "
            new_cols.append(sep)
    return wide[new_cols]





#--------------------------------------------------------------------------------------------------
# Main function
def run_profitability_analysis(
    gdx_unprocessed_path,
    gdx_processed_path,
    prices_xlsx_path,
    output_dir,
    sites=("BMGL", "GOE", "BEZ"),
    lifetimes={"BMGL": 40, "GOE": 40, "BEZ": 40},
    commodities=("ELC", "LTH", "HTH"),
    milestone_years=(2040, 2045, 2050, 2055, 2060, 2065, 2070),
    reactors=("LRG", "SMR", "AMR"),
    urn_processes=("LRG", "SMR", "AMR", "HPLNUC"),
    mapping= {2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},
    end_year=2200,
    discount=True,
    discount_rate=0.025,
    base_year=2040,
    decom_rate=0.10,
    revenue_factors={"ELC": 1.0, "LTH": 1.0, "HTH": 1.0},
    sheet="out_prices_avg",
    region="CH",
    uranium_commodity="URN",
    system_directory=r"C:\GAMS\51",
):
    # ---- defaults ----
    sites = list(sites)
    commodities = list(commodities)
    milestone_years = list(map(int, milestone_years))

    # ---- output folder naming ----
    run_name = os.path.splitext(os.path.basename(prices_xlsx_path))[0]
    out_run_dir = os.path.join(output_dir, run_name)
    os.makedirs(out_run_dir, exist_ok=True)

    com_tag = "-".join(commodities)
    disc_tag = "disc" if discount else "nodisc"
    suffix = f"{run_name}_{com_tag}_{disc_tag}_r{discount_rate:g}_de{decom_rate:g}_by{base_year}"

    xlsx_path = os.path.join(out_run_dir, f"profitability_{suffix}.xlsx")
    wide_csv  = os.path.join(out_run_dir, f"profitability_wide_{suffix}.csv")
    fig_path  = os.path.join(out_run_dir, f"cumulative_net_cashflow_all_sites_{suffix}.png")

    # ---- load GDX ----
    c = gt.Container(load_from=gdx_unprocessed_path, system_directory=system_directory)
    c_out = gt.Container(load_from=gdx_processed_path, system_directory=system_directory)

    pat = re.compile("|".join(map(re.escape, reactors)))

    def load_filtered(name, pcols=("P_3", "P")):
        df = c[name].records
        pcol = next((col for col in pcols if col in df.columns), None)
        if pcol is None:
            raise KeyError(f"{name}: none of {pcols} found")
        return df[df[pcol].astype(str).str.contains(pat, na=False)]
    
    # EXTRACT THE DATA ON THE COSTS AND ACTIVITY FROM THE UNPROCESSED .GDX FILE
    # ---- unprocessed parameters/ variables ----
    VAR_FLO  = load_filtered("VAR_FLO")            #Flow of a commodity in or out of a process
    CST_ACTC = load_filtered("CST_ACTC", ("P",))   #Activity Variable Costs [MCHF]
    CST_FIXC = load_filtered("CST_FIXC", ("P",))   #Fixed O&M Costs [MCHF]  
    CST_FLOX = load_filtered("CST_FLOX")           #Flow taxes/ subsidies related Costs [MCHF]
    CST_INVC = load_filtered("CST_INVC")           #Investment Costs [MCHF]
    CST_FLOC = c["CST_FLOC"].records               #Cost of imported Uranium each milestone year [MCHF]

    # ---- processed: veda_vdd var_fin for uranium fractions ----
    dfv = c_out["veda_vdd"].records.copy()
    pat_urn = re.compile("|".join(map(re.escape, urn_processes)))
    t = pd.to_numeric(dfv["t"], errors="coerce").astype("Int64")
    veda_vdd_varfin = dfv.loc[
        dfv["Attribute"].astype(str).str.strip().str.lower().eq("var_fin")
        & dfv["Process"].astype(str).str.contains(pat_urn, na=False)
        & t.isin(milestone_years)
    ].reset_index(drop=True)


    """
    # ANOTHER, SIMILAR, WAY OF EXTRACTING THE DATA ON THE PROCESSES THAT USE URANIUM AS AN INPUT COMMODITY FROM THE PROCESSED .GDX FILE (i.e. with veda_vdd, from the Fuel_Prices_STEM routine)
    urn_processes = ["LRG", "SMR", "AMR", "HPLNUC"] #HPLNUC used to compute the total consumption of uranium by all processes
    pat_urn = re.compile("|".join(map(re.escape, urn_processes)))
    path_file_out = gt.Container(load_from=r"C:\EEG-Users\Pilar\Updated\ProcessedRuns\93_1.2.gdx",
                    system_directory=r"C:\GAMS\51")
    df = path_file_out["veda_vdd"].records.copy()
    t = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
    m = (
        df["Attribute"].astype(str).str.strip().str.lower().eq("var_fin")   # case-insensitive
        & df["Process"].astype(str).str.contains(pat_urn, na=False)             # any reactor
        & t.isin(milestone_years)                                           # milestone years
    )
    veda_vdd_varfin = df.loc[m].reset_index(drop=True)
    """



    # -------------------------
    # per-site cashflow tables (FULL: revenues + costs + inv + decomm + totals + cumulative)
    # -------------------------
    site_tables = {}
    for s in sites:
        lt = int(lifetimes.get(s, 40))

        # 1) revenues + “normal” costs (no inv/decomm) yearly + extrapolated (and discounted if requested)
        df_parts = Extrapolation(
            VAR_FLO, prices_xlsx_path,
            site=s,
            commodities=tuple(commodities),
            sheet=sheet,
            CST_ACTC=CST_ACTC, CST_FIXC=CST_FIXC, CST_FLOX=CST_FLOX,
            veda_vdd_varfin=veda_vdd_varfin, CST_FLOC=CST_FLOC,
            region=region, uranium_commodity=uranium_commodity,
            mapping=mapping,
            end_year=end_year,
            discount=discount, discount_rate=discount_rate, base_year=base_year,
            revenue_factors=revenue_factors,
        )

        # 2) add inv + decomm + totals + cumulative (discounting happens inside the helper functions consistently)
        full = net_cashflow_full_pipeline(
            df_extrapolated=df_parts,
            CST_INVC=CST_INVC,
            site=s,
            lifetime=lt,
            milestone_years=milestone_years,
            mapping=mapping,
            end_year=end_year,
            decom_rate=decom_rate,
            discount=discount,
            discount_rate=discount_rate,
            base_year=base_year,
            keep_components=True,
        )
        site_tables[s] = full
        # optional: per-site csv
        full.to_csv(os.path.join(out_run_dir, f"{s}_cashflow_{suffix}.csv"), index=False)

    # -------------------------
    # build wide MultiIndex table
    # -------------------------
    parts = []
    for s in sites:
        df = site_tables[s].copy().set_index("Calendar Year")
        df.columns = pd.MultiIndex.from_product([[s], df.columns])
        parts.append(df)

    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "Calendar Year"
    wide = reorder_and_separate(wide, sites, commodities)

    # wide csv (flattened headers)
    flat = wide.copy()
    flat.columns = [(b if a == "" else f"{a} {b}") for (a, b) in flat.columns]
    flat = flat.reset_index()
    flat.to_csv(wide_csv, index=False)

    # -------------------------
    # plot + save figure (uses your updated plotter that calls the full pipeline internally)
    # -------------------------
    plot_cumulative_net_cashflow_all_sites(
        VAR_FLO, prices_xlsx_path,
        sites=tuple(sites),
        lifetimes=lifetimes,
        commodities=tuple(commodities),
        sheet=sheet,
        CST_ACTC=CST_ACTC, CST_FIXC=CST_FIXC, CST_FLOX=CST_FLOX, CST_INVC=CST_INVC,
        veda_vdd_varfin=veda_vdd_varfin, CST_FLOC=CST_FLOC,
        region=region, uranium_commodity=uranium_commodity,
        mapping=mapping, end_year=end_year,
        discount=discount, discount_rate=discount_rate, base_year=base_year,
        revenue_factors=revenue_factors,
        decom_rate=decom_rate,
        save_path=fig_path,
    )

    # -------------------------
    # write Excel (one sheet per site + a wide sheet)
    # -------------------------
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="ALL_SITES_WIDE", index=True, merge_cells=True)
        for s in sites:
            site_tables[s].to_excel(writer, sheet_name=s[:31], index=False)

    return {
        "output_dir": out_run_dir,
        "xlsx": xlsx_path,
        "wide_csv": wide_csv,
        "figure": fig_path,
        "per_site_csvs": {s: os.path.join(out_run_dir, f"{s}_cashflow_{suffix}.csv") for s in sites},
    }


if __name__ == "__main__":
    res = run_profitability_analysis(
        gdx_unprocessed_path=r"C:\EEG-Users\Pilar\Updated\93_1.gdx",
        gdx_processed_path=r"C:\EEG-Users\Pilar\Updated\ProcessedRuns\93_1.2.gdx",
        prices_xlsx_path=r"C:\EEG-Users\Pilar\Updated\ProcessedRuns\93_1.2.xlsx",
        output_dir=r"C:\EEG-Users\Pilar\Updated\AutomatedProfitability\260129_outputs",
        sites=("BMGL", "GOE", "BEZ"),
        lifetimes={"BMGL": 40, "GOE": 40, "BEZ": 40},
        commodities=("ELC", "LTH"),
        discount=True, discount_rate=0.025, base_year=2038,
        decom_rate=0.10,
        revenue_factors={"ELC": 1.0, "LTH": 1.0, "HTH": 1.0},
        end_year=2200,
    )
    print(res)
