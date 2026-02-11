The entire profitability workflow is triggered by calling `run_profitability_analysis(...)` in `main.py`. This function runs the full pipeline: it loads the necessary data from the raw and processed GDX files, filters it to the reactor processes you want to analyze, and then uses the helper functions in `helpers.py` to build all revenue, cost, and cash‑flow results for each site.

--- Required Inputs ---
- Paths to three input files
  - `gdx_unprocessed_path` 
  - `gdx_processed_path`
  - `prices_xlsx_path`
  - plus an `output_dir` where results will be saved
- Which sites to analyze and their lifetimes
- Which commodities are traded, and optional `revenue_factors`
- Model milestone years and how they map to actual years
- The analysis horizon, defined by `end_year`
- Discounting settings: `discount`, `discount_rate`, `base_year`
- Decommissioning assumption: `decom_rate` (fraction of investment cost used as a proxy for decommissioning)

Optional adjustments include:
- selecting which reactor processes to include (`reactors`, `urn_processes`)
- specifying the worksheet name in the prices Excel file (`sheet`)


--- What the pipeline does ---

Once configured, the function:
1. Builds yearly revenue and operating‑cost tables for each site  
2. Extrapolates them to the chosen end year  
3. Adds investment and decommissioning costs  
4. Computes net and cumulative net cash flows  


--- Outputs ---
- one detailed cash‑flow table per site  
- a combined table comparing all sites  
- a plot of cumulative net cash‑flow trajectories  
