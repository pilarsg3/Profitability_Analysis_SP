from gams import transfer as gt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

milestone_years = [2040,2045,2050,2055,2060,2065,2070]
reactors = ["LRG", "SMR", "AMR"]
com = ["ELC", "LTH", "HTH"]

mapping = { 2040: list(range(2038, 2043)), 
           2045: list(range(2043, 2049)), 
           2050: list(range(2049, 2052)), 
           2055: list(range(2052, 2059)), 
           2060: list(range(2059, 2063)), 
           2065: list(range(2063, 2069)), 
           2070: list(range(2069, 2072)), }





#--------------------------------------------------------------------------------------------------
#COMPUTE THE INVESTMENT COSTS THROUGHOUT THE LIFETIME OF THE REACTOR
def apply_lifetime_inv_cost(df, p_like="BMGL", lifetime=40,milestone_years=(2040,2045,2050,2055,2060,2065,2070),p_col="P", t_col="T", year_col="ALLYEAR", value_col="value",start_T=2040, max_T=2200, step=5, last_real_year=2070,):
    df = df.copy()

    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[t_col, year_col, value_col])
    df[t_col] = df[t_col].astype(int)
    df[year_col] = df[year_col].astype(int)
    milestone_years = [int(y) for y in milestone_years]

    sub = df[df[p_col].astype(str).str.contains(p_like, na=False, regex=False)]
    # ---- capture the TRUE first value for each ALLYEAR (T0, v0) ----
    first = {}
    for y in milestone_years:
        s = sub[sub[year_col] == y]
        if s.empty:
            continue
        t0 = int(s[t_col].min())
        v0 = float(s.loc[s[t_col] == t0, value_col].sum())
        first[y] = (t0, v0)
    # Pivot (sum over duplicates)
    table = (
        sub.groupby([t_col, year_col])[value_col].sum()
           .unstack(year_col, fill_value=0.0)
           .reindex(columns=milestone_years, fill_value=0.0)
           .sort_index())
    # full T grid starting at 2040
    table = table.reindex(list(range(start_T, max_T + step, step)), fill_value=0.0)
    # ---- lifetime rule ----
    for y in table.columns:
        col = table[y]
        nonzero = col[col != 0]
        if nonzero.empty:
            continue
        T0 = int(nonzero.index.min())
        T_end = T0 + int(lifetime)
        T_prev = T_end - step

        last_real = col.loc[col.index <= last_real_year]
        last_val = float(last_real.iloc[-1]) if not last_real.empty else 0.0
        start_val = float(col.loc[T0])

        for T in col.index:
            if T < T0:
                table.loc[T, y] = 0.0
            elif T0 < T <= T_end:
                if (not last_real.empty) and (T > int(last_real.index.max())):
                    table.loc[T, y] = last_val
            else:
                table.loc[T, y] = 0.0
        if T_end in table.index:
            table.loc[T_end, y] = float(table.loc[T_prev, y]) - start_val
    # restore the first (T0, v0) cells so they don't get lost
    for y, (t0, v0) in first.items():
        if t0 in table.index and y in table.columns:
            table.loc[t0, y] = v0
    return table





#--------------------------------------------------------------------------------------------------
def inv_cost_table_from_lifetime(
    table,                      # output of apply_lifetime_inv_cost (index=T, cols=vintages)
    mapping = {2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},          
    step=5,
    start_year=None,
    end_year=None,
    discount=False,
    discount_rate=0.0,
    base_year=None,
    out_col="Investment Costs [MCHF]",
):
    """
    1) Expand ROWS (payment year T) to annual years:
       - use explicit mapping for milestone T
       - for other T use block [T-(step-2) .. T+1]
    2) Discount each ANNUAL year row (by Calendar Year)
    3) Sum DOWN each vintage column
    Returns: DataFrame with rows = vintage years (original columns)
    """

    tab = table.copy()

    # ensure clean int index/cols
    tab.index = pd.to_numeric(tab.index, errors="coerce")
    tab = tab.dropna(axis=0)
    tab.index = tab.index.astype(int)
    tab.columns = [int(c) for c in tab.columns]

    # build year -> row-vector mapping (annualization of rows)
    year_to_row = {}
    mapped_keys = set()

    # 1) explicit mapping first
    for ms, yrs in mapping.items():
        ms = int(ms)
        mapped_keys.add(ms)
        row_vec = tab.loc[ms].astype(float).values if ms in tab.index else np.zeros(len(tab.columns), dtype=float)
        for y in yrs:
            year_to_row[int(y)] = row_vec.copy()

    # 2) block rule for remaining T values
    block_start_offset = step - 2  # step=5 -> 3
    for T in tab.index:
        T = int(T)
        if T in mapped_keys:
            continue
        row_vec = tab.loc[T].astype(float).values
        start = T - block_start_offset
        end = T + 1
        for y in range(start, end + 1):
            if y not in year_to_row:  # don't overwrite explicit mapping years already given by STEM
                year_to_row[y] = row_vec.copy()

    # decide output year range
    if start_year is None:
        start_year = min(year_to_row) if year_to_row else int(tab.index.min())
    if end_year is None:
        end_year = max(year_to_row) if year_to_row else int(tab.index.max())

    years_full = list(range(int(start_year), int(end_year) + 1))

    # annualized table: index=Calendar Year, cols=vintages
    annual = pd.DataFrame(
        data=[year_to_row.get(y, np.zeros(len(tab.columns))) for y in years_full],
        index=pd.Index(years_full, name="Calendar Year"),
        columns=tab.columns
    )

    # discount each annual row (by Calendar Year)
    if discount:
        by = int(start_year) if base_year is None else int(base_year)
        r = float(discount_rate)
        disc = 1.0 / ((1.0 + r) ** (annual.index.astype(int) - by))
        annual = annual.mul(disc, axis=0)

    # sum DOWN each vintage column
    pv = annual.sum(axis=0)

    out = pd.DataFrame({
        "Calendar Year": pv.index.astype(int),  # vintage year
        out_col: pv.values
    }).sort_values("Calendar Year").reset_index(drop=True)

    return out





#--------------------------------------------------------------------------------------------------
#OBTAIN THE DECOMMISSIONING COSTS
def decomm_cost_table_from_lifetime(
    table,
    decom_rate=0.10,
    mapping={2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},
    step=5,
    start_year=None,
    end_year=None,
    discount=False,
    discount_rate=0.0,
    base_year=None,
):
    """
    table: output of apply_lifetime_inv_cost (index=T payment years, columns=vintages)
    Returns: yearly Calendar Year decomm costs (one row per year)

    - Uses `mapping` first (e.g. 2040..2070 mapping)
    - For other T values, assigns a contiguous block of length `step`
      defined as [T-(step-2) .. T+1]
      Example step=5:
        2075 -> 2072..2076
        2080 -> 2077..2081
        2085 -> 2082..2086
    """

    # 1) decomm per payment year T (rows of apply_lifetime_inv_cost output)
    decomm_T = (table.sum(axis=1) * decom_rate)
    decomm_T.index = decomm_T.index.astype(int)

    year_to_val = {}

    # 2a) apply explicit mapping first (these years should win)
    mapped_keys = set()
    if mapping is not None:
        for ms, yrs in mapping.items():
            ms = int(ms)
            mapped_keys.add(ms)
            val = float(decomm_T.get(ms, 0.0))
            for y in yrs:
                year_to_val[int(y)] = val

    # 2b) for remaining T values, use block [T-(step-2) .. T+1]
    block_start_offset = step - 2   # step=5 -> 3
    for T, val in decomm_T.items():
        T = int(T)
        if T in mapped_keys:
            continue
        start = T - block_start_offset
        end = T + 1
        for y in range(start, end + 1):
            if y not in year_to_val:
                year_to_val[y] = float(val)

    # 3) decide output year range
    if start_year is None:
        start_year = min(year_to_val) if year_to_val else int(decomm_T.index.min())
    if end_year is None:
        end_year = max(year_to_val) if year_to_val else int(decomm_T.index.max())

    # 4) create full yearly series, fill missing with 0
    years_full = list(range(int(start_year), int(end_year) + 1))
    out = pd.DataFrame({"Calendar Year": years_full})
    out["Decomm Costs [MCHF]"] = [year_to_val.get(y, 0.0) for y in years_full]

    # 5) discounting (by Calendar Year)
    if discount:
        by = int(out["Calendar Year"].min()) if base_year is None else int(base_year)
        r = float(discount_rate)
        disc = 1.0 / ((1.0 + r) ** (out["Calendar Year"] - by))
        out["Decomm Costs [MCHF]"] = pd.to_numeric(out["Decomm Costs [MCHF]"], errors="coerce").fillna(0.0) * disc

    return out





#--------------------------------------------------------------------------------------------------
#OBTAIN ALL THE OTHER COSTS (i.e. all the costs except investment and decommissioning costs)
def actc_cost_table(CST_ACTC, years, site="BMGL"):
    v = "value" if "value" in CST_ACTC.columns else "Value"
    years = list(map(int, years))
    t = pd.to_numeric(CST_ACTC["T"], errors="coerce")
    m = t.isin(years) & CST_ACTC["P"].astype(str).str.contains(site, na=False, regex=False) & \
        CST_ACTC["RPM"].astype(str).str.strip().eq("-")
    out = (pd.to_numeric(CST_ACTC.loc[m, v], errors="coerce").abs()
           .groupby(t[m]).sum().reindex(years, fill_value=0.0).reset_index())
    out.columns = ["Calendar Year", "Activity Variable Costs [MCHF]"]
    return out


def fixc_cost_table(CST_FIXC, years, site="BMGL"):
    v = "value" if "value" in CST_FIXC.columns else "Value"
    years = list(map(int, years))
    t = pd.to_numeric(CST_FIXC["T"], errors="coerce")
    m = t.isin(years) & CST_FIXC["P"].astype(str).str.contains(site, na=False, regex=False)
    out = (pd.to_numeric(CST_FIXC.loc[m, v], errors="coerce")
           .groupby(t[m]).sum().reindex(years, fill_value=0.0).reset_index())
    out.columns = ["Calendar Year", "Fixed O&M Costs [MCHF]"]
    return out


def flox_cost_table(CST_FLOX, years, site="BMGL", c_vals=("ELC","LTH")):
    v = "value" if "value" in CST_FLOX.columns else "Value"
    years = list(map(int, years))
    t = pd.to_numeric(CST_FLOX["T"], errors="coerce")
    c_vals = [str(x) for x in (c_vals if isinstance(c_vals, (list, tuple, set)) else [c_vals])]
    m = (
        t.isin(years)
        & CST_FLOX["P"].astype(str).str.contains(site, na=False, regex=False)
        & CST_FLOX["C"].astype(str).str.strip().isin(c_vals))
    # only commodities that actually appear in the filtered data
    present = sorted(CST_FLOX.loc[m, "C"].astype(str).str.strip().unique().tolist())
    tag = "+".join(present) if present else "+".join(c_vals)  # fallback if empty
    out = (pd.to_numeric(CST_FLOX.loc[m, v], errors="coerce").abs()
           .groupby(t[m]).sum().reindex(years, fill_value=0.0).reset_index())
    out.columns = ["Calendar Year", f"Flow taxes/ subsidies Costs [MCHF] ({tag})"]
    return out


def uranium_cost_and_fractions_from_varfin(veda_vdd_varfin, CST_FLOC, years,
                                           sites=("BMGL","GOE","BEZ"),
                                           commodity="URN",
                                           cost_year_col="T", cost_val_col="value",
                                           region="CH"):
    years = list(map(int, years))
    df = veda_vdd_varfin.copy()
    # URN activity only
    t = pd.to_numeric(df["t"], errors="coerce").astype("Int64")
    v = pd.to_numeric(df["value"], errors="coerce")
    m = t.isin(years) & df["Commodity"].astype(str).str.strip().eq(commodity)
    total = v[m].groupby(t[m]).sum().reindex(years, fill_value=0.0).values
    denom = np.where(total == 0, np.nan, total)
    out = pd.DataFrame({"Calendar Year": years})
    # fractions per site
    for s in sites:
        ms = m & df["Process"].astype(str).str.contains(s, na=False, regex=False)
        site_sum = v[ms].groupby(t[ms]).sum().reindex(years, fill_value=0.0).values
        out[f"{s} Fraction use of uranium"] = np.nan_to_num(site_sum / denom)
    # uranium cost from CST_FLOC
    dfc = CST_FLOC.copy()
    r_col = next((c for c in dfc.columns if c.upper().startswith("R")), None)
    c_col = next((c for c in dfc.columns if c.upper().startswith("C")), None)
    tc = pd.to_numeric(dfc[cost_year_col], errors="coerce")
    cv = pd.to_numeric(dfc[cost_val_col], errors="coerce")
    mc = tc.isin(years)
    if r_col is not None:
        mc &= dfc[r_col].astype(str).str.strip().eq(region)
    if c_col is not None:
        mc &= dfc[c_col].astype(str).str.strip().eq(commodity)
    out["Cost U [MCHF]"] = cv[mc].groupby(tc[mc]).sum().reindex(years, fill_value=0.0).values
    # cost allocation per site
    for s in sites:
        out[f"Cost {s} Uranium [MCHF]"] = out["Cost U [MCHF]"] * out[f"{s} Fraction use of uranium"]
    return out





#--------------------------------------------------------------------------------------------------
#Combine all the costs (except investment and decommissioning costs into one table, only for the milestone years)
def cost_cashflow_table_no_inv_decomm(
    CST_ACTC, CST_FIXC, CST_FLOX,
    years, site="BMGL", c_vals=("ELC","LTH"),
    veda_vdd_varfin=None, CST_FLOC=None, region="CH", uranium_commodity="URN"):
    years = list(map(int, years))
    # --- Uranium inputs MUST be provided ---
    if veda_vdd_varfin is None or CST_FLOC is None:
        raise ValueError("Uranium costs are mandatory: provide both veda_vdd_varfin and CST_FLOC.")

    out = pd.DataFrame({"Calendar Year": years})
    a = actc_cost_table(CST_ACTC, years, site=site)
    f = fixc_cost_table(CST_FIXC, years, site=site)
    # FLOX: compute then force a generic title (no commodities)
    x = flox_cost_table(CST_FLOX, years, site=site, c_vals=c_vals)
    flox_cols = [c for c in x.columns if c != "Calendar Year"]
    if len(flox_cols) == 1:
        x = x.rename(columns={flox_cols[0]: "Flow taxes/ subsidies Costs [MCHF]"})
    else:
        x["Flow taxes/ subsidies Costs [MCHF]"] = x[flox_cols].sum(axis=1)
        x = x[["Calendar Year", "Flow taxes/ subsidies Costs [MCHF]"]]
    # Uranium cost
    u = uranium_cost_and_fractions_from_varfin(
        veda_vdd_varfin, CST_FLOC, years,
        sites=("BMGL","GOE","BEZ"),
        commodity=uranium_commodity,
        region=region
    )[["Calendar Year", f"Cost {site} Uranium [MCHF]"]]
    out = (out.merge(a, on="Calendar Year", how="left")
              .merge(f, on="Calendar Year", how="left")
              .merge(x, on="Calendar Year", how="left")
              .merge(u, on="Calendar Year", how="left"))
    cost_cols = [c for c in out.columns if c != "Calendar Year"]
    out[cost_cols] = out[cost_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out





#--------------------------------------------------------------------------------------------------
#Obtain the activity for each commodity
def activity_table_multi_com(VAR_FLO, years, site="BMGL",
                             com=("ELC", "LTH", "HTH"),
                             year_col="ALLYEAR_2",
                             p_col="P_3", c_col="C_4",
                             level_col="level",
                             add_total=False):
    #Sum VAR_FLO 'level' by milestone year and commodity, filtered by site in P.
    #Output columns:
    #  - Calendar Year
    #  - Activity [ELC] (sum level)
    #  - Activity [LTH] (sum level)
    #  - Activity [HTH] (sum level)
    #  - (optional) Total Activity (sum level)
    df = VAR_FLO.copy()
    years = list(map(int, years))
    com = list(com)
    # sanity checks
    for col in (year_col, p_col, c_col, level_col):
        if col not in df.columns:
            raise KeyError(f"'{col}' not found in VAR_FLO columns: {df.columns.tolist()}")
    t = pd.to_numeric(df[year_col], errors="coerce")
    lvl = pd.to_numeric(df[level_col], errors="coerce")
    # base filter: year + site
    m_base = (
        t.isin(years)
        & df[p_col].astype(str).str.contains(site, na=False, regex=False)
    )
    # keep only needed rows/cols
    tmp = df.loc[m_base, [c_col]].copy()
    tmp["Calendar Year"] = t[m_base].astype(int)
    tmp["level"] = lvl[m_base]
    # restrict to commodities list
    tmp = tmp[tmp[c_col].astype(str).str.strip().isin(com)]
    # pivot: rows = years, cols = commodities, values = sum(level)
    wide = (tmp.groupby(["Calendar Year", c_col], observed=False)["level"].sum()
          .unstack(c_col, fill_value=0.0)
          .reindex(index=years, fill_value=0.0)
          .reindex(columns=com, fill_value=0.0)
          .reset_index())
    # rename columns
    rename_map = {c: f"Activity {c} [PJ]" for c in com}
    wide = wide.rename(columns=rename_map)
    if add_total:
        activity_cols = [rename_map[c] for c in com]
        wide["Total Activity"] = wide[activity_cols].sum(axis=1)
    return wide





#--------------------------------------------------------------------------------------------------
#Obtain the prices of each commodity in each milestone year from the output excel of the GAMS routine Fuel_Prices_STEM
def price_table_from_excel(path, commodities, milestone_years, sheet="out_prices_avg"):
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    # find header row with years
    year_row = None
    for i in range(len(df)):
        yrs = pd.to_numeric(df.iloc[i, 1:], errors="coerce")
        if yrs.notna().sum() > 0:
            year_row = i
            break
    if year_row is None:
        raise ValueError("Couldn't find a row with year values.")
    years_all = pd.to_numeric(df.iloc[year_row, 1:], errors="coerce").dropna().astype(int).tolist()
    ms = list(map(int, milestone_years))

    pos = {y: i for i, y in enumerate(years_all)}  # year -> index in row values
    out = pd.DataFrame({"Calendar Year": ms}).set_index("Calendar Year")

    labels = df.iloc[:, 0].astype(str).str.strip()
    for com in commodities:
        hit = df[labels.eq(com)]
        if hit.empty:
            out[f"Price {com} [MCHF/PJ]"] = 0.0
        else:
            vals = pd.to_numeric(hit.iloc[0, 1:1+len(years_all)], errors="coerce").fillna(0.0).values
            out[f"Price {com} [MCHF/PJ]"] = [vals[pos[y]] if y in pos else 0.0 for y in ms]
    return out.reset_index()





#--------------------------------------------------------------------------------------------------
# Combines into one table the revenues from each commodity and the costs (except inv and decom costs) for the milestone years
def revenue_table_no_inv_decomm(
    VAR_FLO, price_xlsx_path, years,
    site="BMGL", commodities=("ELC","LTH"),
    sheet="out_prices_avg",
    CST_ACTC=None, CST_FIXC=None, CST_FLOX=None,
    veda_vdd_varfin=None, CST_FLOC=None,
    region="CH", uranium_commodity="URN",
    revenue_factors=None
):
    import pandas as pd

    years = list(map(int, years))
    revenue_factors = revenue_factors or {}

    # --- keep only commodities present in VAR_FLO for this site + these years ---
    df_tmp = VAR_FLO.copy()
    t = pd.to_numeric(df_tmp["ALLYEAR_2"], errors="coerce")
    m = (
        t.isin(years)
        & df_tmp["P_3"].astype(str).str.contains(site, na=False, regex=False)
    )
    present = (
        df_tmp.loc[m, "C_4"]
        .astype(str).str.strip()
        .unique()
        .tolist()
    )
    commodities = [str(c).strip() for c in commodities]
    commodities = [c for c in commodities if c in present]
    # --- prices + activity (milestone years only) ---
    prices = price_table_from_excel(price_xlsx_path, commodities, years, sheet=sheet)
    act = activity_table_multi_com(VAR_FLO, years, site=site, com=commodities, add_total=False)
    out = prices.merge(act, on="Calendar Year", how="left").fillna(0.0)
    # --- revenues (per commodity) ---
    for com in commodities:
        f = float(revenue_factors.get(com, 1.0))
        out[f"Revenue {com} [MCHF]"] = (
            f
            * out[f"Price {com} [MCHF/PJ]"]
            * out[f"Activity {com} [PJ]"]
        )
    # --- costs (NO inv + NO decomm) ---
    if CST_ACTC is None or CST_FIXC is None or CST_FLOX is None:
        raise ValueError("Provide CST_ACTC, CST_FIXC, CST_FLOX to compute costs.")
    if veda_vdd_varfin is None or CST_FLOC is None:
        raise ValueError("Uranium costs are mandatory: provide veda_vdd_varfin and CST_FLOC.")
    costs = cost_cashflow_table_no_inv_decomm(
        CST_ACTC, CST_FIXC, CST_FLOX,
        years,
        site=site, c_vals=commodities,
        veda_vdd_varfin=veda_vdd_varfin, CST_FLOC=CST_FLOC,
        region=region, uranium_commodity=uranium_commodity
    )
    out = out.merge(costs, on="Calendar Year", how="left").fillna(0.0)
    # keep only Calendar Year + Revenues + cost components
    rev_cols = [f"Revenue {c} [MCHF]" for c in commodities]
    cost_cols = [c for c in costs.columns if c != "Calendar Year"]
    keep = ["Calendar Year"] + rev_cols + cost_cols
    return out[keep]





#--------------------------------------------------------------------------------------------------
#Expand the values using the milestone years to all years mapping
def final_yearly_table(
    VAR_FLO, price_xlsx_path,
    site="BMGL",
    commodities=("ELC","LTH"),
    sheet="out_prices_avg",
    CST_ACTC=None, CST_FIXC=None, CST_FLOX=None,
    veda_vdd_varfin=None, CST_FLOC=None,
    region="CH", uranium_commodity="URN",
    mapping= {2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},
    revenue_factors=None,
    end_year=2200
):
    import pandas as pd
    milestone_years = sorted(map(int, mapping.keys()))
    # --- keep only commodities that actually appear in VAR_FLO for this site + milestone years ---
    df_tmp = VAR_FLO.copy()
    t = pd.to_numeric(df_tmp["ALLYEAR_2"], errors="coerce")
    m = (
        t.isin(milestone_years)
        & df_tmp["P_3"].astype(str).str.contains(site, na=False, regex=False)
    )
    present = (
        df_tmp.loc[m, "C_4"]
        .astype(str).str.strip()
        .unique()
        .tolist()
    )
    # keep only requested commodities that are present
    commodities = [str(c).strip() for c in commodities]
    commodities = [c for c in commodities if c in present]
    # 1) milestone table: revenues + cost components (NO inv, NO decomm, NO net cashflow)
    src = revenue_table_no_inv_decomm(
        VAR_FLO, price_xlsx_path, milestone_years,
        site=site, commodities=commodities, sheet=sheet,
        CST_ACTC=CST_ACTC, CST_FIXC=CST_FIXC, CST_FLOX=CST_FLOX,
        veda_vdd_varfin=veda_vdd_varfin, CST_FLOC=CST_FLOC,
        region=region, uranium_commodity=uranium_commodity,
        revenue_factors=revenue_factors
    ).copy()
    src["Calendar Year"] = pd.to_numeric(src["Calendar Year"], errors="coerce").astype(int)
    src = src.set_index("Calendar Year")
    # 2) yearly skeleton
    start_year = int(min(min(v) for v in mapping.values()))
    out = pd.DataFrame({"Calendar Year": range(start_year, int(end_year) + 1)})
    # 3) expand ALL columns via mapping
    cols = src.columns.tolist()
    rows = []
    for ms, yrs in mapping.items():
        ms = int(ms)
        if ms not in src.index:
            continue
        vals = src.loc[ms, cols].to_dict()
        for y in yrs:
            rows.append({"Calendar Year": int(y), **vals})
    mapped = pd.DataFrame(rows)
    out = out.merge(mapped, on="Calendar Year", how="left")
    # 4) fill outside mapping with zeros
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out.sort_values("Calendar Year").reset_index(drop=True)





#--------------------------------------------------------------------------------------------------
#Extrapolate the values from 2070 until end_year (by default 2200) approximated as the average of the values in 2065 and 2070
def Extrapolation(
    VAR_FLO, price_xlsx_path,
    site="BMGL",
    commodities=("ELC","LTH"),
    sheet="out_prices_avg",
    CST_ACTC=None, CST_FIXC=None, CST_FLOX=None,
    veda_vdd_varfin=None, CST_FLOC=None,
    region="CH", uranium_commodity="URN",
    mapping = {2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},
    end_year=2200,
    discount=False, discount_rate=0.0, base_year=None,
    revenue_factors=None,
    # which years define your “steady-state” extrapolation
    anchor_years=(2065, 2070),
    # last year that contains “real” mapped values (typically max year in mapping)
    last_real_year=None,
):
    import pandas as pd
    # 1) Get yearly table (already expanded by mapping, filled with 0 outside mapping)
    df = final_yearly_table(
        VAR_FLO, price_xlsx_path,
        site=site,
        commodities=commodities,
        sheet=sheet,
        CST_ACTC=CST_ACTC, CST_FIXC=CST_FIXC, CST_FLOX=CST_FLOX,
        veda_vdd_varfin=veda_vdd_varfin, CST_FLOC=CST_FLOC,
        region=region, uranium_commodity=uranium_commodity,
        mapping=mapping,
        revenue_factors=revenue_factors,
        end_year=end_year
    ).copy()
    df["Calendar Year"] = pd.to_numeric(df["Calendar Year"], errors="coerce").astype(int)
    df = df.sort_values("Calendar Year").reset_index(drop=True)
    # 2) Decide last_real_year (end of mapping coverage)
    if last_real_year is None:
        if mapping is None:
            # fallback: assume last "real" year is last year with any non-zero values
            num_cols = [c for c in df.columns if c != "Calendar Year"]
            nonzero_years = df.loc[df[num_cols].abs().sum(axis=1) > 0, "Calendar Year"]
            last_real_year = int(nonzero_years.max()) if not nonzero_years.empty else int(df["Calendar Year"].min())
        else:
            last_real_year = int(max(max(v) for v in mapping.values()))
    # 3) Compute fill values from anchor years (avg of 2065&2070 by default)
    num_cols = [c for c in df.columns if c != "Calendar Year" and pd.api.types.is_numeric_dtype(df[c])]
    a1, a2 = map(int, anchor_years)
    if (a1 in df["Calendar Year"].values) and (a2 in df["Calendar Year"].values):
        v1 = df.loc[df["Calendar Year"] == a1, num_cols].iloc[0]
        v2 = df.loc[df["Calendar Year"] == a2, num_cols].iloc[0]
        fill_vals = (v1 + v2) / 2.0
    else:
        # fallback: use last real year row
        fill_vals = df.loc[df["Calendar Year"] == last_real_year, num_cols].iloc[0]
    # 4) Extrapolate AFTER last_real_year (overwrite the zeros that mapping created)
    mask_future = df["Calendar Year"] > last_real_year
    if mask_future.any():
        df.loc[mask_future, num_cols] = fill_vals.values
    # 5) Discount (in-place) WITHOUT keeping a discount factor column
    if discount:
        by = int(df["Calendar Year"].min()) if base_year is None else int(base_year)
        r = float(discount_rate)
        disc = 1.0 / ((1.0 + r) ** (df["Calendar Year"] - by))  # correct: <1 after base year
        df[num_cols] = df[num_cols].mul(disc, axis=0)
    return df





#--------------------------------------------------------------------------------------------------
#Compute the net cashflow from the revenues and all costs (incl. inv and decom costs), for every individual year from 2038 to end_year
def net_cashflow_full_pipeline(
    df_extrapolated,          # output of Extrapolation() (already discounted or not)
    CST_INVC,                 # original CST_INVC (filtered) dataframe
    site="BMGL",
    lifetime=40,
    milestone_years=(2040,2045,2050,2055,2060,2065,2070),
    mapping= {2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},
    end_year=2200,
    decom_rate=0.10,
    # pass-through to inv_cost_table_from_lifetime (discounting happens THERE, not here)
    discount=False,
    discount_rate=0.0,
    base_year=None,
    # apply_lifetime_inv_cost params you may want to override
    p_col="P",
    t_col="T",
    year_col="ALLYEAR",
    value_col="value",
    start_T=2040,
    step=5,
    last_real_year=2070,
    keep_components=False,
):
    """
    Computes Investment + Decomm costs internally (calling your functions),
    merges them with df_extrapolated, then computes:
      Net Cash Flow = Total Revenue - Total Costs
    NO discounting is applied in this function. It assumes consistency:
      - df_extrapolated discounted? => discount=True here too (so inv gets discounted inside its function)
      - df_extrapolated undiscounted? => discount=False here too
    """
    import pandas as pd
    # -------------------------
    # 1) Build lifetime table (payment-year T x vintage)
    # -------------------------
    inv_lifetime_table = apply_lifetime_inv_cost(
        CST_INVC,
        p_like=site,
        lifetime=lifetime,
        milestone_years=milestone_years,
        p_col=p_col,
        t_col=t_col,
        year_col=year_col,
        value_col=value_col,
        start_T=start_T,
        max_T=end_year,
        step=step,
        last_real_year=last_real_year,
    )
    # -------------------------
    # 2) Investment costs (YEARLY via mapping; discounting happens inside inv_cost_table_from_lifetime)
    # -------------------------
    inv_yearly = inv_cost_table_from_lifetime(
        inv_lifetime_table,
        mapping=mapping,
        end_year=end_year,
        start_year=min(min(v) for v in mapping.values()),
        discount=discount,
        discount_rate=discount_rate,
        base_year=base_year,
    )[["Calendar Year", "Investment Costs [MCHF]"]].copy()
    # drop any Discount Factor column if your inv function still returns it
    if "Discount Factor" in inv_yearly.columns:
        inv_yearly = inv_yearly.drop(columns=["Discount Factor"])
    # -------------------------
    # 3) Decommissioning costs (YEARLY; mapping/windows; NO discounting here)
    # -------------------------
    decomm_yearly = decomm_cost_table_from_lifetime(
        inv_lifetime_table,
        decom_rate=decom_rate,
        mapping=mapping,
        step=step,
        start_year=min(min(v) for v in mapping.values()),
        end_year=end_year, discount=discount,
        discount_rate=discount_rate,
        base_year=base_year
        )[["Calendar Year", "Decomm Costs [MCHF]"]].copy()
    # -------------------------
    # 4) Merge with extrapolated table
    # -------------------------
    out = df_extrapolated.copy()
    out["Calendar Year"] = pd.to_numeric(out["Calendar Year"], errors="coerce").astype(int)

    inv_yearly["Calendar Year"] = pd.to_numeric(inv_yearly["Calendar Year"], errors="coerce").astype(int)
    decomm_yearly["Calendar Year"] = pd.to_numeric(decomm_yearly["Calendar Year"], errors="coerce").astype(int)

    out = (out.merge(inv_yearly, on="Calendar Year", how="left")
              .merge(decomm_yearly, on="Calendar Year", how="left"))
    # numeric cleanup
    for c in out.columns:
        if c != "Calendar Year":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    # -------------------------
    # 5) Compute totals + net cash flow (NO discounting here)
    # -------------------------
    revenue_cols = [c for c in out.columns if c.startswith("Revenue ")]
    # costs = everything numeric except revenues, activities, prices, calendar year, discount factor
    def is_cost_col(c: str) -> bool:
        if c == "Calendar Year":
            return False
        if c.startswith("Revenue "):
            return False
        if c.startswith("Price "):
            return False
        if c.startswith("Activity ") and c.endswith("[PJ]"):
            return False
        if c == "Discount Factor":
            return False
        return True
    cost_cols = [c for c in out.columns if is_cost_col(c)]

    out["Total Revenue [MCHF]"] = out[revenue_cols].sum(axis=1) if revenue_cols else 0.0
    out["Total Costs [MCHF]"] = out[cost_cols].sum(axis=1) if cost_cols else 0.0
    out["Net Cash Flow [MCHF]"] = out["Total Revenue [MCHF]"] - out["Total Costs [MCHF]"]
    out["Cumulative Net Cash Flow [MCHF]"] = out["Net Cash Flow [MCHF]"].cumsum()

    out = out.sort_values("Calendar Year").reset_index(drop=True)
    if keep_components:
        return out
    return out[[
        "Calendar Year",
        "Total Revenue [MCHF]",
        "Total Costs [MCHF]",
        "Net Cash Flow [MCHF]",
        "Cumulative Net Cash Flow [MCHF]",
    ]]





#--------------------------------------------------------------------------------------------------
#Plot the cumulative net cash flow for all sites
def plot_cumulative_net_cashflow_all_sites(
    VAR_FLO, price_xlsx_path,
    sites=("BMGL","GOE","BEZ"),
    lifetimes=None,
    commodities=("ELC","LTH"),
    sheet="out_prices_avg",
    CST_ACTC=None, CST_FIXC=None, CST_FLOX=None, CST_INVC=None,
    veda_vdd_varfin=None, CST_FLOC=None, region="CH", uranium_commodity="URN",
    mapping={2040: list(range(2038, 2043)),
            2045: list(range(2043, 2049)),
            2050: list(range(2049, 2052)),
            2055: list(range(2052, 2059)),
            2060: list(range(2059, 2063)),
            2065: list(range(2063, 2069)),
            2070: list(range(2069, 2072)),},
    end_year=2200,
    discount=False, discount_rate=0.025, base_year=2040,
    save_path=None, dpi=300,
    revenue_factors=None,
    decom_rate=0.10
):
    if lifetimes is None:
        lifetimes = {s: 40 for s in sites}

    fig, ax = plt.subplots()
    msg_lines = []

    for s in sites:
        lt = lifetimes.get(s, 40)
        # 1) revenues + operational costs (excl. inv/decomm), extrapolated (and discounted if requested)
        df_parts = Extrapolation(
            VAR_FLO, price_xlsx_path,
            site=s,
            commodities=commodities,
            sheet=sheet,
            CST_ACTC=CST_ACTC, CST_FIXC=CST_FIXC, CST_FLOX=CST_FLOX,
            veda_vdd_varfin=veda_vdd_varfin, CST_FLOC=CST_FLOC,
            region=region, uranium_commodity=uranium_commodity,
            mapping=mapping,
            end_year=end_year,
            discount=discount, discount_rate=discount_rate, base_year=base_year,
            revenue_factors=revenue_factors
        )
        # 2) add inv + decomm internally + compute net + cumulative
        cashflow = net_cashflow_full_pipeline(
            df_extrapolated=df_parts,
            CST_INVC=CST_INVC,
            site=s,
            lifetime=lt,
            milestone_years=sorted(map(int, mapping.keys())),
            mapping=mapping,
            end_year=end_year,
            decom_rate=decom_rate,
            discount=discount,
            discount_rate=discount_rate,
            base_year=base_year,
            keep_components=False
        )
        x = cashflow["Calendar Year"].astype(int).values
        y = cashflow["Cumulative Net Cash Flow [MCHF]"].astype(float).values

        ax.plot(x, y, label=f"{s} ({lt}y)")

        #idx0 = next((i for i, val in enumerate(y) if val >= 0), None)
        #msg_lines.append(f"{s} in {x[idx0]}" if idx0 is not None else f"{s}: never ≥ 0")
        # --- BOL = first year with non-zero activity ---
        ms_years = sorted(map(int, mapping.keys()))
        act_ms = activity_table_multi_com(
            VAR_FLO, ms_years,
            site=s,
            com=commodities,
            add_total=True
        )
        act_ms = act_ms.set_index("Calendar Year")

        # expand milestone activity to annual years using mapping
        year_to_act = {}
        for ms, yrs in mapping.items():
            ms = int(ms)
            tot = float(act_ms.loc[ms, "Total Activity"]) if ms in act_ms.index else 0.0
            for yy in yrs:
                year_to_act[int(yy)] = tot
        bol_year = None
        for yy in sorted(year_to_act):
            if year_to_act[yy] > 0:
                bol_year = yy
                break
        # --- profitability = first time cumulative >= 0 AFTER we have been negative, and after BOL ---
        idx0 = None
        if bol_year is not None:
            mask = x >= bol_year
            x2 = x[mask]
            y2 = y[mask]
            # only declare "profitable" once it actually crosses from negative to >= 0
            was_negative = False
            for i in range(len(y2)):
                if y2[i] < 0:
                    was_negative = True
                if was_negative and y2[i] >= 0:
                    idx0 = np.where(mask)[0][i]   # index back into original x/y
                    break
        # fallback: if it never went negative after BOL, then it's "profitable from BOL"
        if idx0 is None and bol_year is not None:
            i0 = np.where((x >= bol_year) & (y >= 0))[0]
            idx0 = int(i0[0]) if len(i0) else None
        msg_lines.append(
            f"{s} in {x[idx0]} (BOL={bol_year})" if idx0 is not None
            else f"{s}: never ≥ 0 after BOL={bol_year}"
        )
    ax.set_ylim(-20000, 20000)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")

    com_str = "+".join(commodities)
    title = f"Cumulative Net Cash Flow ({com_str})"
    title += f" discounted @{discount_rate:.2%}" if discount else " undiscounted"
    ax.set_title(title)

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Net Cash Flow [MCHF]")
    ax.grid(True)
    ax.legend()

    ax.text(
        0.02, 0.98,
        "Cumulative Net Cash Flow ≥ 0:\n" + "\n".join(msg_lines),
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", alpha=0.25)
    )
    if save_path is None:
        save_path = "cumulative_net_cashflow_all_sites.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    return save_path





