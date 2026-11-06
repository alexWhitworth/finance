# Validation report: model extrinsic vs. manual broker quotes

Quotes read: 12 from `manual_quotes.csv`

## 1. Extrinsic vs. observed spread

| ticker | expiry | strike | spot | r(T) | q | model ext. | quote mid ext. | half-spread | inside? |
|---|---|---|---|---|---|---|---|---|---|
| VTI | 2027-01-15 | 150.00 | 300.00 | 0.0422 | 0.0130 | 0.213 | 0.213 | 0.150 | yes |
| VTI | 2027-01-15 | 160.00 | 300.00 | 0.0422 | 0.0130 | 0.311 | 0.311 | 0.150 | yes |
| VTI | 2026-06-18 | 148.00 | 300.00 | 0.0430 | 0.0130 | 0.000 | -0.000 | 0.150 | yes |
| VTI | 2027-01-15 | 140.00 | 300.00 | 0.0422 | 0.0130 | 0.287 | 0.287 | 0.150 | yes |
| VTI | 2027-01-15 | 155.00 | 300.00 | 0.0422 | 0.0130 | 0.258 | 0.259 | 0.150 | yes |
| IWM | 2027-07-15 | 110.00 | 225.00 | 0.0414 | 0.0130 | 0.704 | 0.704 | 0.176 | yes |
| IWM | 2027-07-15 | 120.00 | 225.00 | 0.0414 | 0.0130 | 1.011 | 1.011 | 0.253 | yes |
| IWM | 2027-01-15 | 108.00 | 220.00 | 0.0422 | 0.0130 | 0.199 | 0.199 | 0.150 | yes |
| GLD | 2027-01-15 | 120.00 | 240.00 | 0.0422 | 0.0130 | 0.001 | 0.001 | 0.150 | yes |
| GLD | 2027-01-15 | 130.00 | 240.00 | 0.0422 | 0.0130 | 0.003 | 0.003 | 0.150 | yes |
| GLD | 2026-06-18 | 118.00 | 235.00 | 0.0430 | 0.0130 | 0.000 | 0.000 | 0.150 | yes |
| GLD | 2027-07-15 | 135.00 | 240.00 | 0.0414 | 0.0130 | 0.065 | 0.065 | 0.150 | yes |

Inside the half-spread: **12/12**

Acceptance (INV-7): >= 2/3 inside — **PASS** (100%)

## 2. (rho, eta) pinning vs. frozen priors

| ticker | pinned_rho | prior_rho | pinned_eta | prior_eta |
|---|---|---|---|---|---|
| VTI | -0.72 | -0.700 | 1.08 | 1.100 |
| GLD | 0.14 | 0.150 | 0.92 | 0.900 |

## 3. kappa / sigma_inf sensitivity on the 2y ATM band

| ticker | vol index | 2y base ATM sigma | kappa+0.5 | kappa-0.5 | sigma_inf (frozen) |
|---|---|---|---|---|---|
| VTI | 0.18 | 0.1800 | 0.1800 | 0.1800 | 0.1800 |
| IWM | 0.20 | 0.1866 | 0.1851 | 0.1889 | 0.1800 |
| GLD | 0.18 | 0.1800 | 0.1800 | 0.1800 | 0.1800 |

## 4. MOVE D_eff conversion check (VGIT/MUB)

Observed ^MOVE level (bp): 100.0

| ETF | D_eff | converted IV | listed-option range | inside? |
|---|---|---|---|---|
| VGIT | 5.1 | 0.0510 | [0.03, 0.08] | yes |
| MUB | 5.5 | 0.0550 | [0.02, 0.06] | yes |

