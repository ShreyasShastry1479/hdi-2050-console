# HDI 2050 Projection Console: Methodology

**Version:** 1.2  
**Research status:** independent, reproducible scenario projection  
**Coverage:** 193 countries; 1,775 subdivisions in 164 countries  
**Forecast horizon:** HDR 2025 release baseline to 2050

## 1. Research Design

The primary research question is:

> How could demographic structure, institutional capacity, technological diffusion, resource and climate pressure, and subnational inequality shape national human-development trajectories by 2050?

The model estimates a conditional scenario outcome, not an unconditional forecast. It asks what 2050 HDI would be if the structural assumptions encoded in a scenario were to hold. Country rank is a derived output and is not the optimization target.

The project tests four related propositions:

1. Developing countries with human-capital absorption and institutional capacity can converge faster than frontier economies.
2. Frontier economies experience strong saturation and may stagnate or decline when ageing and shock risks outweigh marginal gains.
3. National gains can mask persistent internal dispersion.
4. Technology and demographic dividends improve HDI only when education, infrastructure, and governance can convert them into health, schooling, and income outcomes.

## 2. Data Provenance

### National baseline

The baseline is anchored to **UNDP Human Development Report 2025, Table 1**. The report cycle is 2025, while most component observations in that table refer to **2023**. The dashboard therefore uses “HDR 2025 baseline” as a release label, not as a claim that all inputs were observed in calendar year 2025.

National population totals and projections are used for global weighted means and subnational reconciliation. Additional structural features are assembled from the project's panel and country-factor modules. Every output row records its baseline and population source fields where available.

### Subnational baseline

Subnational HDI observations originate in the supplied `HDI 2025 By Subdivision.xlsx` workbook. Source regions are harmonized to ISO3 country codes and mapped to available ADM1 geometry through explicit aliases and country-specific crosswalks. Some source records are statistical regions rather than legally defined ADM1 units; those remain labeled as source regions in the table.

Where an official subdivision population projection is unavailable, population shares are modeled proxies. Seeded priors are used for countries with curated subdivision weights. Other countries use normalized priors based partly on internal HDI concentration, then evolve to 2050 using national population growth, urbanization, and growth prospects. These estimates are marked `modeled_proxy_not_official_subdivision_projection`.

## 3. HDI Mathematics

### Official UNDP structure

The official HDI is the geometric mean of health, education, and income dimension indices:

```text
HDI = (Health Index x Education Index x Income Index)^(1/3)
```

The UNDP technical notes define the goalposts and detailed construction. The project preserves the geometric reconciliation identity in its 2050 output.

### Console component calibration

The current implementation computes display/calibration indices as:

```text
Health_cal = clamp((life expectancy - 20) / (88 - 20), 0, 1)
EYS_cal    = clamp(expected years of schooling / 18, 0, 1)
MYS_cal    = clamp(mean years of schooling / 16, 0, 1)
Education_cal = sqrt(EYS_cal x MYS_cal)
Income_cal = clamp((ln(GNIpc) - ln(100)) / (ln(105000) - ln(100)), 0, 1)
```

These extended calibration goalposts differ from the standard UNDP goalposts. They were introduced to avoid premature saturation in long-horizon component displays. They must not be described as official UNDP dimension indices.

Raw 2050 component values are allocated from the modeled HDI gain using the health, education, and income attribution shares and a saturating headroom function. A common multiplicative factor is then found by binary search, with each component bounded to `[0, 1]`, so that:

```text
(Health_2050 x Education_2050 x Income_2050)^(1/3) = HDI_2050
```

The exported recomputed HDI field should equal the projected HDI after display rounding.

## 4. National Projection Model

The national model is a nonlinear convergence model with a frontier of 0.985:

```text
gap_i = max(0, 0.985 - HDI_i,baseline)
gain_i = gap_i x (1 - exp(-speed_i x years)) + adjustments_i
HDI_i,2050 = clamp(HDI_i,baseline + gain_i, 0.250, 0.985)
```

The pre-adjustment convergence speed is a weighted sum:

```text
raw speed = 0.11 Income
          + 0.29 Education
          + 0.15 Health
          + 0.18 Governance
          + 0.16 Demographics
          + 0.09 Future readiness
          + 0.02 State capacity
```

The model then applies continuous multipliers for development stage, industrialization, catch-up readiness, human-development persistence, growth prospects, demographic conversion, digital infrastructure, frontier saturation, institutional bottlenecks, and expected shock risk. The speed is clipped to `[0.001, 0.055]`.

Additive adjustments cover recovery potential, social mobility, resource-windfall conversion, demographic adaptation, demographic dividend, workforce depth, digital infrastructure, climate and inequality drag, dependency, ageing, expected shocks, resource volatility, and regression risk. Positive gains are capped by development stage so low-HDI catch-up is possible without implying instant convergence.

Country trajectory labels are descriptive outputs used for interpretation. Where country-specific assumptions remain in code, including GDP growth priors, education-reform factors, and expected regression risk, they are scenario assumptions and not learned causal coefficients.

## 5. Scenarios and Sensitivity

The baseline CSV is immutable in the browser. Scenario controls apply transparent client-side adjustments to the baseline projection:

- **Aggressive AI adoption:** raises the return to future readiness, digital infrastructure, and human-capital absorption.
- **Climate/resource mitigation:** reduces climate and resource drags.
- **Stagnation stress test:** suppresses gains and increases exposure to structural drag.
- **Inclusive mobility and cohesion:** explores stronger public-service access, language access, identity recognition, and intergenerational mobility. Identity composition itself has no direct HDI coefficient.
- **Custom tuning:** applies user-selected AI, cloud, resource-relief, human-capital, and inclusion changes.

Scenario outputs are comparative stress tests, not separately trained forecasts.

## 6. Uncertainty

P10 and P90 values are heuristic structured bounds around the baseline scenario. Width depends on baseline development level, state capacity, inequality, climate exposure, political stability, fertility, and trajectory class. Conflict-recovery and frontier-jumper cases receive wider bounds; frontier countries receive narrow bounds because of mathematical saturation.

These bounds are useful for scenario comparison but are **not calibrated probabilistic intervals**. Their nominal coverage should not be interpreted as an 80% empirical guarantee.

## 7. Subnational Projection and Reconciliation

For subdivision `j` in country `i`:

```text
deviation_ij = source HDI_ij - weighted source mean_i
shrink_i = clamp(0.86 - 0.38 x max(0, HDI_i,2050 - HDI_i,baseline), 0.42, 0.86)
preliminary HDI_ij,2050 = HDI_i,2050 + deviation_ij x shrink_i
```

Values are bounded and iteratively re-centered so population-weighted means exactly recover the national targets:

```text
sum_j(weight_ij,2025 x HDI_ij,2025) = HDI_i,baseline
sum_j(weight_ij,2050 x HDI_ij,2050) = HDI_i,2050
```

This procedure preserves relative internal ordering and assumes partial convergence. It does not independently forecast every province's health, education, and income components. Researchers should therefore treat subdivision results as a national-consistent allocation scenario.

## 8. Validation

The included historical backtest contains 2,170 held-out country-year observations from 2014-2023:

| Metric | Result |
|---|---:|
| MAE | 0.00199 |
| RMSE | 0.00286 |
| Mean error | -0.00001 |
| R-squared | 0.99958 |
| 90th percentile absolute error | 0.00416 |

This is a short-horizon panel backtest using lagged/current historical covariates. It demonstrates numerical fit for one-step historical prediction; it does **not** validate 25-year geopolitical, technological, demographic, or climate assumptions. Long-horizon credibility therefore depends on scenario sensitivity, transparent priors, and repeated model updates as new observations arrive.

## 9. Population, Identity, and Religious Context

Population-identity and religious-composition layers are used as contextual demographic research, not as direct deterministic inputs to HDI. Composition diversity and concentration are descriptive statistics only. They are excluded from the HDI adjustment equation and from the service-delivery-gap calculation.

This distinction is ethically and analytically necessary: identity is not a causal deficit or an inherent capability. The research question is whether institutions distribute health, education, infrastructure, and economic opportunity equitably. Effects may enter the scenario model only through measurable and policy-relevant conditions such as horizontal inequality, unequal service access, exclusion, institutional capacity, language access, or conflict exposure.

The following responsible-use constraints apply:

- Census and self-identification categories are socially defined, fluid, and not comparable without qualification across countries or years.
- The model does not rank identity or religious groups, infer individual characteristics, or prescribe a preferred population composition.
- Identity-category transition represents changes in reporting or classification, not a desired movement toward a majority group.
- Outputs must not be used for profiling, exclusion, surveillance, immigration eligibility, or claims about group capability.
- Group-level scenarios require local source review, affected-community input, and uncertainty disclosure before publication or policy use.

## 10. External Poverty and Electricity Context

The Research page joins the HDI scenarios to two external descriptive series: the global electricity-generation mix and regional extreme-poverty totals. These series are analytical context only. They are not training features, do not modify country HDI scores, and should not be interpreted as causal estimates of the effect of electricity or poverty on HDI.

The poverty visualization uses `data/output/extreme_poverty_regional_1990_2040.csv`, downloaded from Our World in Data's processing of Lakner et al. (2025) and the World Bank Poverty and Inequality Platform, updated in March 2026. It contains eight complete annual series for 1990-2040: seven World Bank regions plus the world aggregate. Extreme poverty is defined as living below $3 per person per day in 2021 international dollars.

The poverty projection has two materially different forecast regimes:

- 2026-2030 uses World Bank and IMF growth forecasts.
- 2031-2040 applies average country growth observed during 2015-2024.

The later reversal is therefore conditional on recent growth persistence. It is not extrapolated to 2050 by this project. Regional totals reconcile to the published world aggregate within rounding. Country policy should use national poverty lines and household-survey evidence alongside the common international line.

### Africa industrial and digital strategy

The Research page adds an evidence synthesis on poverty reduction in Africa through industrialization, productive digitization, human capability, and regional trade. It is a policy framework rather than a causal estimate or a new model input. The synthesis separates three types of content:

- **Observed indicators:** published measures of workforce growth, employment structure, electricity access, and mobile-internet use.
- **Conditional estimates:** published modeling such as the World Bank's estimate that full African Continental Free Trade Area implementation could raise regional income by about 7 percent and lift 30 million people from extreme poverty by 2035. This is a modeled benefit under implementation assumptions, not a forecast.
- **Project interpretation:** illustrative value chains, sequencing, safeguards, and country entry pathways. These are hypotheses for policy analysis and require country-specific feasibility, distributional, fiscal, environmental, and political-economy review.

The dashboard's country table joins these illustrative pathways to existing scenario outputs for HDI, industrial acceleration, digital infrastructure, and human-capital absorption. Changing the active scenario updates the displayed model values, but the pathways themselves do not change country scores. The analysis does not treat Africa as a single economy: infrastructure, institutions, firm capabilities, labor markets, geography, and feasible specialization differ materially across 54 countries.

The proposed mechanism is complementary rather than substitutive. Manufacturing and agro-processing can create tradable output, supplier networks, and scalable employment; digital systems can lower payment, coordination, logistics, and information costs. Neither can substitute for reliable electricity, transport, water, housing, foundational health and education, capable public administration, or worker protections. The framework therefore avoids claims that low wages alone create competitiveness or that connectivity alone creates structural transformation.

## 11. ML Research Layer

The ML Lab is an auditable secondary layer over the published projection table. It does not silently replace the core scenario model.

### Counterfactual surrogate

`build_ml_research_artifacts.py` fits an Extra Trees regressor to the 193 national rows. The target is the published 2025-to-2050 HDI gain and the 27 features cover the 2025 HDI anchor and dimensions, technology and digital readiness, health and renewable inputs, institutional efficiency, demographics, human-capital absorption, resource and volatility drag, inequality and climate risk, growth prospects, recovery, diversification, catch-up readiness, workforce momentum, dependency pressure, and the existing trajectory-speed assumption.

Five-fold cross-validation against the existing scenario output gives:

| Surrogate-fidelity metric | Result |
|---|---:|
| R-squared | 0.94611 |
| MAE | 0.00712 HDI |
| RMSE | 0.01023 HDI |

These values measure how closely the surrogate reproduces the current scenario engine. They are **not** long-horizon forecast-validation statistics. The historical backtest in Section 8 remains the separate observed-data validation exercise.

The browser preserves each country's published 2050 HDI as its anchor. Each user-controlled policy proxy is perturbed separately. The magnitude comes from the local ensemble response, while the sign is constrained so increasing an explicitly beneficial lever cannot lower HDI merely because of a non-monotonic tree split. Effects are added and capped at +/-0.080 HDI for interface stability. Education-budget uplift is an explicitly labeled proxy that adds 0.012 Education Index points per percentage point of GDP; it is not an estimated fiscal treatment effect.

### Explainability and diagnostics

- Baseline TreeSHAP values reconcile to the ensemble's fitted gain and describe feature attribution inside the surrogate. They do not establish causal effects.
- Live slider chips are one-at-a-time local perturbations, not recomputed Shapley values.
- Isolation Forest scores flag the top unusual multivariate national and subdivision profiles for manual review. A flag does not prove a source error.
- Seven subdivision peer clusters are selected from a six-to-twelve-cluster search using silhouette score. Standardized nearest-neighbor search returns five structurally similar regions outside the reference country where possible.
- Subdivision features combine HDI level and gain, relative national position, population scale and growth, digital readiness, demographics, human-capital absorption, resource drag, momentum, and future readiness. Similarity does not imply that two regions share institutions, history, or the same appropriate policy.

The artifact is rebuilt with:

```powershell
python build_ml_research_artifacts.py
```

It writes `data/output/ml_research_artifacts.json`, containing the exported trees, validation metadata, baseline SHAP values, anomaly scores, cluster profiles, and twin-region keys.

## 12. Policy Screening

The dashboard generates transparent screening flags for low component scores, ageing/dependency pressure, weak digital infrastructure, negative governance or resource contributions, and large subnational gaps. These are prioritization prompts, not causal treatment recommendations. A policy decision requires local administrative data, budget constraints, implementation evidence, and stakeholder review.

## 13. Limitations

- The model cannot predict wars, pandemics, border changes, technological discontinuities, or policy reversals.
- Several future-oriented and institutional inputs are scenario priors or modeled estimates.
- Country-specific assumptions increase realism but reduce the purity of a single globally estimated model.
- The uncertainty bounds are heuristic and not empirically calibrated for 2050 coverage.
- Subnational populations are modeled proxies in many countries.
- ADM1 geometry and statistical-region definitions do not always align.
- National HDI can conceal distributional inequality; the project does not yet calculate official IHDI by ethnic group or subdivision.
- The external poverty series is not a poverty forecast generated by the HDI model and does not establish that HDI growth will be distributionally inclusive.
- Historical backtest performance should not be extrapolated mechanically to the 2050 horizon.
- The ML surrogate can reproduce inherited scenario assumptions and any remaining biases in the published projection table; high surrogate fidelity does not make those assumptions true.
- Counterfactual sliders operate on policy proxies and local model sensitivity. They should not be interpreted as causal returns to spending without country-specific identification and implementation evidence.

## 14. Reproducibility

```powershell
python run_2050.py
python build_subdivision_hdi.py
python build_ml_research_artifacts.py
python hdi_gpt_server.py
```

Primary outputs:

- `data/output/hdi_2050_rankings.csv`
- `data/output/subdivision_hdi_2025_2050.csv`
- `data/output/backtest_predictions.csv`
- `data/output/ethnic_composition_2050_ai_model.csv`
- `data/output/religious_composition_2025_2050.csv`
- `data/output/extreme_poverty_regional_1990_2040.csv`
- `data/output/ml_research_artifacts.json`

## 15. Literature and Source Foundation

- United Nations Development Programme. [Human Development Report 2025, Table 1](https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Statistical_Annex_HDI_Table.pdf).
- United Nations Development Programme. [Human Development Report 2025 technical notes](https://hdr.undp.org/sites/default/files/2025_HDR/hdr2025_technical_notes.pdf).
- Barro, R. J., & Sala-i-Martin, X. (1992). [Convergence](https://www.journals.uchicago.edu/doi/10.1086/261816). *Journal of Political Economy, 100*(2), 223-251.
- Easterly, W., & Levine, R. (1997). [Africa's Growth Tragedy: Policies and Ethnic Divisions](https://doi.org/10.1162/003355300555466). *Quarterly Journal of Economics, 112*(4), 1203-1250.
- Alesina, A., Devleeschauwer, A., Easterly, W., Kurlat, S., & Wacziarg, R. (2003). [Fractionalization](https://www.nber.org/system/files/working_papers/w9411/w9411.pdf). *Journal of Economic Growth, 8*(2), 155-194.
- Romer, P. M. (1990). [Endogenous Technological Change](https://doi.org/10.1086/261725). *Journal of Political Economy, 98*(5), S71-S102.
- World Bank. [June 2025 Update to Global Poverty Lines](https://www.worldbank.org/en/news/factsheet/2025/06/05/june-2025-update-to-global-poverty-lines).
- World Bank. [Poverty, Prosperity, and Planet Report](https://www.worldbank.org/en/publication/poverty-prosperity-and-planet).
- Our World in Data. [Total population living in extreme poverty by world region](https://ourworldindata.org/grapher/projections-extreme-poverty-wb).
- World Bank. [Africa's Pulse, October 2025](https://www.worldbank.org/en/publication/africa-pulse).
- World Bank. [Digital Africa: Technological Transformation for Jobs](https://www.worldbank.org/en/region/afr/publication/digital-africa).
- World Bank. [African Continental Free Trade Area could lift 30 million people from extreme poverty](https://www.worldbank.org/en/news/press-release/2020/07/27/african-continental-free-trade-area).
- International Energy Agency. [Stepping Up the Value Chain in Africa](https://www.iea.org/reports/stepping-up-the-value-chain-in-africa).
- International Energy Agency. [Financing Electricity Access in Africa](https://www.iea.org/reports/financing-electricity-access-in-africa).
- United Nations Industrial Development Organization. [Industrial Development Report 2024 for Africa](https://www.unido.org/news/unido-launches-industrial-development-report-2024-africa).
- GSMA. [The Mobile Economy Africa 2026](https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-economy/africa/).
