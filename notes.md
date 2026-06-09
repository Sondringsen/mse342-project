# What remains
- Implement a way to easily turn on and off topological loss
- Run experiments
- Can we run on gpu on cluster


# Outline for report
## Abstract
- Summary of what we did and the results
- Our contribution - add topological loss

## Introduction
- Overview of synthetic financial data generation
- Diffusion models for time series generation (Hoetal2020)
- Define stylized facts (Cont2001)
- Need to test on downstream task (theis2016noteevaluationgenerativemodels)
- Deep hedging models have been popularized and need more data than what is readily available from real data. This motivates the use of synthetic data. 
- TDA in finance. We do this because it has been shown to predict crashes etc. (GIDEA2018820) It is therefore nice to test whether this helps in synthtetic data generation. The gain from this might not show up test sets where the market is in a 'normal' state.
- Test on real data
- Include references for all of the above

## Method
### DDPM
- Follow the setup of the CoFinDiff Paper (tanaka2025cofindiffcontrollablefinancialdiffusion)
- We use open source code from https://github.com/EmmanuelleB985/FinDiffusion (include as footnote). This is not the authors' code, but implemented by Oxford researchers. They don't use the Haar Wavelet, so that is the only difference.
- We generate in return space
- We generate 252 (1-year) sequences of price data
- They also have a way to generate conditional data, we want to generate unconditional data. Conditioning can be useful if you have some view of the market you want to incorporate.

### TDA 
- We closely follow the setup in (GIDEA2018820)
- Point clouds
- Vietoris ripz complex (can use cech complex, but it's more computationally expensive)
- Betti numbers
- Persistence diagrams 
- Stability theorem
- How to make them differentiable - persistence landscape (bubenik2015)
- Topological loss

### Stylized facts and visualization (Cont2001)
- We evaluate the generated data on stylized facts. This is more of a sanity check than a final evaluation. At the end of the day we care about how useful the data is for downstream tasks (reference). 
- Stylized facts considered: kurtosis, vol clustering, leverage effect, 
- We also plot the data for a quick sanity check. 

### Hedging model (Buehler2019)
- Follow the setup in the CoFinDiff paper:
- Five-layer peceptron with layernorms and ReLU activations. 
- European call with 30 day ttl (they used 300 minutes in the cofindiff paper, refer to data section where the data deviation is discussed in more detail)
- Option a bit OTM the money because that is where liquidity is and we will most likely need to adjust our position somewhat
- Hedging model uses CVaR as loss. Discuss why we do this. This is good because many firms care about this, reference MIFID II? For some firms it might make more sense to train on expected value, however, the point of hedging isn't to maximize the expected value, it is to reduce risk...
- CVaR and drawdown. CoFinDiff paper used cvar and entropic risk measure.  

## Data 
- 30 liquid stocks from SP500
- 2005 - 2024 
- 60 - 20 - 20 train, val test
- This deviates from the original paper. This is from the original paper: "This study used 1-minute FLEX full historical data provided by Japan Exchange Group, Inc. The trading hours per day was 5 hours, with data spanning from January 1, 2015, to December 31, 2021." They used 11 Japanese stocks.
- It can be argued that their is more appropriate to do on a smaller time scale than what we do as a lot changes over such a long time frame. 


## Result
- Stylized facts table
- Compare hedging performance on real test data for three heding models:
    1. Hedging model trained on synthetic data from cofindiff 
    2. Hedging model trained on synthetic data from cofindiff with topological loss 
    3. Hedging model trained on real data (of course not overlapping with the test data) - this is important to compare as if we don't gain anything by training on synthetic data we don't gain anything.


## Discussion and future work
- Summarize the results and what we have shown
- Include more data. The pointcloud easily extend to multidimensional data. This is not as easy for the ddpm. We have seen ddpm for timeseries and tabular, but not with both.
- Do it on a smaller timescales
- Extend to more options. How is it affected by ttl and moneyness?

## Appendix
- Table of all stocks considered


# Plan today

- Fix the cluster. Start a job