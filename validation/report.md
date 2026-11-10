# Validation report: model extrinsic vs. manual broker quotes

Quotes read: 18 from `manual_quotes.csv`

## 1. Extrinsic vs. observed spread

| ticker | expiry | strike | spot | r(T) | q | model ext. | quote mid ext. | half-spread | inside? | source |
|---|---|---|---|---|---|---|---|---|---|---|
| VTI | 2028-12-15 | 165.00 | 368.56 | 0.0456 | 0.0106 | 1.586 | 2.740 | 1.750 | yes | manual quote |
| VTI | 2028-12-15 | 170.00 | 368.56 | 0.0456 | 0.0106 | 1.757 | 3.034 | 1.500 | yes | manual quote |
| VTI | 2028-12-15 | 180.00 | 368.56 | 0.0456 | 0.0106 | 2.138 | 3.371 | 1.750 | yes | manual quote |
| VTI | 2028-12-15 | 190.00 | 368.56 | 0.0456 | 0.0106 | 2.577 | 3.959 | 1.750 | yes | manual quote |
| VTI | 2028-12-15 | 200.00 | 368.56 | 0.0456 | 0.0106 | 3.080 | 4.546 | 1.750 | yes | manual quote |
| VTI | 2028-12-15 | 375.00 | 368.56 | 0.0456 | 0.0106 | 30.981 | 31.625 | 1.700 | yes | manual quote |
| VTI | 2028-12-15 | 400.00 | 368.56 | 0.0456 | 0.0106 | 37.095 | 37.400 | 1.600 | yes | manual quote |
| VTI | 2028-12-15 | 410.00 | 368.56 | 0.0456 | 0.0106 | 32.251 | 32.450 | 1.550 | yes | manual quote |
| IWM | 2028-12-15 | 125.00 | 280.03 | 0.0456 | 0.0097 | 1.366 | 1.995 | 1.710 | yes | manual quote |
| IWM | 2028-12-15 | 135.00 | 280.03 | 0.0456 | 0.0097 | 1.764 | 2.228 | 1.565 | yes | manual quote |
| IWM | 2028-12-15 | 145.00 | 280.03 | 0.0456 | 0.0097 | 2.240 | 2.740 | 1.640 | yes | manual quote |
| GLD | 2028-12-15 | 200.00 | 369.04 | 0.0456 | 0.0000 | 0.533 | 1.583 | 1.825 | yes | manual quote |
| GLD | 2028-12-15 | 220.00 | 369.04 | 0.0456 | 0.0000 | 0.998 | 2.733 | 1.750 | yes | manual quote |
| GLD | 2028-12-15 | 250.00 | 369.04 | 0.0456 | 0.0000 | 2.315 | 5.395 | 1.600 | no | manual quote |
| GLD | 2028-12-15 | 300.00 | 369.04 | 0.0456 | 0.0000 | 7.542 | 12.832 | 1.500 | no | manual quote |
| GLD | 2028-12-15 | 375.00 | 369.04 | 0.0456 | 0.0000 | 28.582 | 36.212 | 1.525 | no | manual quote |
| GLD | 2028-12-15 | 400.00 | 369.04 | 0.0456 | 0.0000 | 40.113 | 48.081 | 1.225 | no | manual quote |
| GLD | 2028-12-15 | 410.00 | 369.04 | 0.0456 | 0.0000 | 41.798 | 49.475 | 1.525 | no | manual quote |

Inside the half-spread: **13/18** (non-synthetic: 13/18)

Acceptance (INV-7): >= 2/3 inside — **PASS** (72% of 18 real quote(s))

### Residual gap attribution (outside rows)

| ticker | strike | model IV | quote IV | IV gap (pts) |
|---|---|---|---|---|
| GLD | 250.00 | 0.2154 | 0.2650 | +4.96 |
| GLD | 300.00 | 0.2053 | 0.2501 | +4.47 |
| GLD | 375.00 | 0.2016 | 0.2405 | +3.89 |
| GLD | 400.00 | 0.2031 | 0.2412 | +3.82 |
| GLD | 410.00 | 0.2039 | 0.2402 | +3.63 |

- GLD: mean IV gap +4.15 pts, range 1.33 pts — IV gap varies with moneyness — smile shape (rho/eta) contributes.

## 2. (rho, eta) pinning vs. frozen priors

| ticker | pinned_rho | prior_rho | pinned_eta | prior_eta | source |
|---|---|---|---|---|---|
| VTI | -0.48 | -0.700 | 1.71 | 1.100 | manual quote |
| GLD | -0.08 | 0.150 | 0.72 | 0.900 | manual quote |

## 3. kappa / sigma_inf sensitivity on the 2y ATM band

| ticker | vol index | 2y base ATM sigma | kappa+0.5 | kappa-0.5 | sigma_inf (frozen) |
|---|---|---|---|---|---|
| VTI | 0.20 | 0.1871 | 0.1855 | 0.1896 | 0.1800 |
| IWM | 0.22 | 0.1947 | 0.1915 | 0.1998 | 0.1800 |
| GLD | 0.25 | 0.2045 | 0.1993 | 0.2127 | 0.1800 |

Note: when the vol-index level equals sigma_inf (VTI/GLD at the seed levels), theta = sigma_inf^2 * tau + (v0 - sigma_inf^2) * (1 - exp(-kappa * tau)) / kappa is identically sigma_inf^2 * tau, so the kappa columns are flat by mathematical identity, not by harness defect; kappa sensitivity is only observable where level != sigma_inf (e.g. IWM).

## 4. MOVE D_eff conversion check (VGIT/MUB)

Observed ^MOVE level (bp): 92.2 (observed 2026-11-10 11:02 PT)

| ETF | D_eff | converted IV | listed-option range | inside? |
|---|---|---|---|---|
| VGIT | 5.1 | 0.0470 | [0.03, 0.08] | yes |
| MUB | 5.5 | 0.0507 | [0.02, 0.06] | yes |

