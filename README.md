# HDI 2050 Projection Console

An open, interactive research system for exploring national and subnational human-development scenarios through 2050. The project combines the 193-country UNDP Human Development Report 2025 release, demographic and institutional factors, future-oriented technology variables, scenario controls, historical backtesting, and a 1,775-row subnational extension.

> **Research status:** independent scenario projection. This is not a UNDP forecast and should not be read as a deterministic prediction. The HDR 2025 baseline uses the latest values published in that report cycle, which are primarily observations for 2023.

## Research Questions

1. Which countries are most likely to converge toward the human-development frontier by 2050 under plausible structural conditions?
2. Where can national progress coexist with persistent or widening subnational inequality?
3. How sensitive are projections to technology diffusion, demographic structure, institutional capacity, climate/resource pressure, and inclusive governance?
4. Which lagging subdivisions warrant priority health, education, infrastructure, or institutional intervention?

## What the Project Contains

- **193-country projection table:** baseline HDI, 2050 scenario HDI, component indices, P10/P90 heuristic bounds, ranks, demographics, readiness factors, and driver attribution.
- **Subnational extension:** 1,775 subdivision records covering 164 countries, reconciled to each country's population-weighted national HDI target.
- **Interactive console:** maps, tables, scenario tuning, comparison tools, country briefs, validation diagnostics, demographic mosaics, and HDI-GPT.
- **ML research layer:** live policy counterfactuals, TreeSHAP attribution, Isolation Forest audit flags, and cross-border subdivision similarity search.
- **Historical validation:** 2,170 held-out country-year observations from 2014-2023.
- **Reproducible documentation:** exact equations, data provenance, assumptions, limitations, and literature framing in [METHODOLOGY.md](METHODOLOGY.md).
- **Policy narrative:** research findings and appropriate interpretation in [RESEARCH_REPORT.md](RESEARCH_REPORT.md).

## Run Locally

```powershell
python hdi_gpt_server.py
```

Open `http://localhost:8765/`. API-backed HDI-GPT requires `OPENAI_API_KEY` or `HDI_GPT_API_KEY` in the server process environment. Never place a key in client-side JavaScript or commit it to the repository.

### Enable HDI-GPT on Vercel

The browser calls the server-side `/api/hdi-gpt` function; the API key must therefore be configured in the Vercel project, not only in a local PowerShell session.

1. Open **Vercel Project Settings -> Environment Variables**.
2. Add `OPENAI_API_KEY` for Production, Preview, and Development.
3. Optionally add `HDI_GPT_MODEL`; the default is `gpt-4.1-mini`.
4. Redeploy the latest commit so the function receives the new environment variables.

If `/api/hdi-gpt` returns HTTP `503`, the deployed function cannot see either `OPENAI_API_KEY` or `HDI_GPT_API_KEY`.

## Rebuild Outputs

```powershell
python run_2050.py
python build_subdivision_hdi.py
python build_ml_research_artifacts.py
```

Core outputs are written to `data/output/`. The dashboard reads these files directly, so generated outputs and UI calculations can be audited independently.

`build_ml_research_artifacts.py` exports a browser-readable Extra Trees surrogate and precomputed explainability, anomaly, cluster, and nearest-neighbor records to `data/output/ml_research_artifacts.json`. Its cross-validation measures fidelity to the existing scenario table, not observed 2050 forecast accuracy.

## Key Sources

- [UNDP Human Development Report 2025, Table 1](https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Statistical_Annex_HDI_Table.pdf)
- [UNDP HDR 2025 technical notes](https://hdr.undp.org/sites/default/files/2025_HDR/hdr2025_technical_notes.pdf)
- [UNDP data documentation and downloads](https://hdr.undp.org/data-center/documentation-and-downloads)

## Citation

HDI 2050 Projection Console. (2026). *Global and subnational human development scenario projections, version 1.2*. https://hdi-2050-console-f9gr.vercel.app/

## Limitations

The model does not predict geopolitical border changes, black-swan shocks, or exact future policy choices. P10/P90 ranges are structured scenario bounds, not empirically calibrated UN confidence intervals. Subnational population weights are modeled proxies where official projections are unavailable. Ethnic and religious composition layers provide demographic context and are not used as deterministic or causal penalties.
