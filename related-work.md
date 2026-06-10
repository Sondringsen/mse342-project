# Related Work: Diffusion Models for Time Series

Papers and codebases relevant to the MSE342 project (diffusion model + topological loss for financial time series generation).

---

## Directly relevant: financial time series generation

### CoFinDiff (IJCAI 2025)
**Paper:** https://arxiv.org/abs/2503.04164  
**Notes:** Purpose-built for financial time series (price, volume, volatility, spread). Conditional diffusion via cross-attention on trend/volatility conditions. Captures stylized facts. Improves deep hedging on generated data.  
**Caveats:** Uses DDPM (not Song 2021 SDE). Converts time series to 152×16 wavelet images before diffusion — makes plugging in topological loss harder (need inverse wavelet round-trip to get back to time series).

### Takahashi & Mizuno 2025 — cited in proposal as [Takahashi_2025]
**Paper:** https://arxiv.org/abs/2410.18897  
**Code:** Not publicly available.  
**Notes:** DDPM + Haar wavelet transform for multivariate financial time series. Same wavelet-image approach as CoFinDiff. Published in Quantitative Finance journal.

### TRADES / DeepMarket (2025)
**Paper:** https://arxiv.org/abs/2502.07071  
**Code:** https://github.com/LeonardoBerti00/DeepMarket  
**Notes:** Generates realistic limit order book (LOB) data. Strong on stylized facts. Too specialised (LOB-focused) for our use case.

---

## Score-based / SDE framework

### SigDiffusions (ICLR 2025)
**Paper:** https://arxiv.org/abs/2406.10354  
**Code:** https://github.com/Barb0ra/SigDiffusions  
**Notes:** Uses Song 2021 SDE framework exactly as specified in the proposal. Operates natively on multivariate time series (no wavelet conversion) — topological loss can be plugged in directly on generated samples $\hat{x}_0$. Log-signature embeddings preserve algebraic path structure.  
**Caveats:** Not trained on financial data — experiments use synthetic sines, predator-prey ODEs, household power consumption. Data mismatch is a tuning problem, not architectural.

### CSDI — Conditional Score-based Diffusion for Imputation (NeurIPS 2021)
**Paper:** https://arxiv.org/abs/2107.03502  
**Code:** https://github.com/ermongroup/CSDI  
**Notes:** Despite "score-based" in the name, uses DDPM (not the continuous SDE framework). Operates natively on multivariate time series with a 2D transformer (temporal attention + feature attention). Primary use case is imputation — unconditional generation requires treating all values as missing. Trained on PhysioNet (healthcare) and air quality data, not financial.  
**Verdict for this project:** Architecturally clean for adding topological loss (native time series, no wavelet), but wrong data domain and designed for imputation not generation.

### Song et al. 2021 — SDE framework (base reference from proposal)
**Paper:** https://arxiv.org/abs/2011.13456  
**Code:** https://github.com/yang-song/score_sde  
**Notes:** The foundational SDE framework the proposal follows. VP-SDE / VE-SDE / sub-VP-SDE. Score network $s_\theta(x, t)$ trained via denoising score matching.

---

## Survey / overview

### Awesome TimeSeries Diffusion Models
**Repo:** https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model  
**Notes:** Comprehensive list of diffusion model papers for time series and spatio-temporal data.

---

## Recommendation summary

| Model | SDE framework | Native time series | Financial data | Topo loss easy |
|---|---|---|---|---|
| SigDiffusions | Yes | Yes | No | Yes |
| CoFinDiff | No (DDPM) | No (wavelet images) | Yes | Hard |
| CSDI | No (DDPM) | Yes | No | Yes |
| TRADES | No | No (LOB events) | Yes (LOB) | Hard |

**Best starting point:** SigDiffusions — matches the proposal's SDE math, operates on raw time series, topological loss plugs in directly on $\hat{x}_0$. Fine-tuning on financial data is straightforward.
