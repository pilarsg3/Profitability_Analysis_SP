The full profitability workflow is executed by calling run_profitability_analysis(...) in main.py.  This function handles the entire pipeline: it loads the required data from the raw and processed GDX files, filters it to the reactor processes you want to analyze,  and uses the helper functions in helpers.py to build all revenue, cost, and cash‑flow outputs for each site.

--- Required Inputs ---
To run the pipeline, you need to specify:
•  Paths to the input files
  ⁠◦  gdx_unprocessed_path
  ⁠◦  gdx_processed_path
  ⁠◦  prices_xlsx_path
  ⁠◦  plus an output_dir for saving results
•  Sites to analyze and their lifetimes
•  Traded commodities and optional revenue_factors
•  Model milestone years and their mapping to actual years
•  Analysis horizon, defined by end_year
•  Discounting settings (discount, discount_rate, base_year)
•  Decommissioning assumption (decom_rate)

Optional configuration includes selecting which reactor processes to include (reactors, urn_processes) and specifying the worksheet name in the prices Excel file (sheet).


--- What the Pipeline Does ---
Once configured, the function:
1.  Builds yearly revenue and operating‑cost tables for each site
2.  Extrapolates these values to the chosen end year
3.  Adds investment and decommissioning costs
4.  Computes net and cumulative net cash flows


--- Outputs --
The script generates:
•  A detailed cash‑flow table for each site
•  A combined table comparing all sites
•  A figure showing cumulative net cash‑flow trajectories
