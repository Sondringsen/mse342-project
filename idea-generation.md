# Project Idea Generation — MSE 342

## Your Idea: Diffusion Models with Topological Loss for Financial Time Series

### Assessment

This is a genuinely compelling idea, and I think it has real publication potential. Here is an honest breakdown.

**What is novel**: Using persistent homology as a differentiable loss signal to train or guide a score-based diffusion model for financial time series generation appears to be unexplored. There is a growing literature on topological losses in generative models (mostly for images and point clouds), and a separate literature on TDA applied to financial time series — but the intersection with diffusion models specifically is, to my knowledge, open.

**Why the Betti number framing is interesting**: The intuition you have is sound and maps well onto financial phenomena:

- **β₀ (connected components)**: In a sublevel-set filtration of a return series, β₀ captures distinct clusters or disconnected regions — a natural proxy for market *regimes* (bull, bear, crisis). A topological loss penalizing mismatch in β₀ persistence would push the generator to reproduce the right number and duration of regimes.
- **β₁ (loops/cycles)**: In a Vietoris–Rips or sliding-window filtration, β₁ captures recurrent structure — a reasonable proxy for *business cycles* or seasonality. This is less obvious but potentially very interesting.

The key insight is that these topological features are *global* properties of the time series that standard $L^2$ or spectral losses are blind to. This is exactly the kind of thing a vanilla diffusion model would fail to reproduce even if it matched marginal distributions perfectly.

**Potential challenges to think through**:

1. **Choice of filtration**: For time series, you need to decide *how* to build the simplicial complex. Common options: sublevel-set filtration on the 1D signal, Vietoris–Rips on a sliding-window embedding (Takens), or a path-signature approach. The choice materially affects what β₀ and β₁ capture and how informative they are.
2. **Differentiability**: You need gradients to flow back through the persistence computation. This is solvable — see Carrière et al. (2021) and the `Gudhi` / `giotto-tda` / `TopLayer` libraries — but adds implementation complexity.
3. **Computational cost**: Persistent homology on long time series is expensive. You will likely need to work on shorter windows (e.g., monthly rolling windows) and aggregate.
4. **What exactly is the loss?**: Options include Wasserstein distance between persistence diagrams, a summary statistic like total persistence, or a Betti-curve $L^2$ loss. These have different computational and theoretical properties.
5. **Baseline comparison**: You need a strong baseline. The most natural comparison is a diffusion model trained without the topological loss, evaluated on both standard metrics (FID / discriminative score / predictive score from TimeGAN) *and* topological metrics.

**Verdict**: This is feasible, novel, and well-scoped for a course project. If the topological loss demonstrably improves reproduction of stylized facts — particularly volatility clustering, heavy tails, and regime structure — that is a publishable result. I would encourage you to confirm the topic with the instructor early, as they may know of a preprint you are not aware of.

---

## Relevant Papers

### Diffusion Models for Time Series

- **Tang & Zhao (2025)** — "Score-based diffusion models via stochastic differential equations." *Statistic Surveys* 19: 28–64. *(On the syllabus — your primary reference for the SDE framework.)*
- **Tashiro et al. (2021)** — "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation." *NeurIPS 2021.* Good reference for conditional diffusion on time series.
- **Yuan & Qiao (2024)** — "Diffusion-TS: Interpretable Diffusion for General Time Series Generation." *ICLR 2024.* Recent strong baseline for time series generation with diffusion.
- **Yoon et al. (2019)** — "Time-series Generative Adversarial Networks (TimeGAN)." *NeurIPS 2019.* The standard evaluation framework and baseline; defines the discriminative and predictive scores you should report.

### Topological Losses in Generative Models

- **Moor et al. (2020)** — "Topological Autoencoders." *ICML 2020.* First prominent use of a persistent-homology loss to regularize a neural network's latent space. Key conceptual reference.
- **Gabrielsson et al. (2020)** — "A Topology Layer for Machine Learning." *AISTATS 2020.* Provides a differentiable topology layer you could adapt.
- **Carrière et al. (2021)** — "Optimizing Persistent Homology Based Functions." *ICML 2021.* The theoretical backbone for differentiating through persistence diagrams — essential reading for your implementation.
- **Hu et al. (2019)** — "Topology-Preserving Deep Image Segmentation." *NeurIPS 2019.* Introduces the Betti-matching loss for images; the conceptual analog of what you want to do for time series.

### TDA for Financial Time Series

- **Gidea & Katz (2018)** — "Topological Data Analysis of Financial Time Series: Landscapes of Crashes." *Physica A.* Shows that persistent homology of multivariate return series gives early warning signals for market crashes. Very relevant motivation.
- **Gidea (2017)** — "Topological Data Analysis of Critical Transitions in Financial Networks." Uses TDA to detect phase transitions — directly related to your regime-detection intuition.
- **Cont (2001)** — "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." *Quantitative Finance.* The canonical reference for stylized facts — you need this to define what your model should reproduce.

### Persistent Homology for Time Series (General)

- **Perea & Harer (2015)** — "Sliding Windows and Persistence: An Application of Topological Methods to Signal Analysis." *Foundations of Computational Mathematics.* Establishes the sliding-window (Takens) approach to computing β₁ for periodic signals.
- **Chazal & Michel (2021)** — "An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists." *Frontiers in AI.* Good survey if you need to connect your TDA course to the generative modeling context.

---

## Alternative Project Ideas

If you want to explore other directions, here are several ideas spanning course content, with and without TDA.

---

### Idea 1: RL Fine-Tuning of Financial Diffusion Models (No TDA)

**Concept**: The course covers RL-based fine-tuning of diffusion models (see Uehara et al. on the syllabus). The standard application is text-to-image or language. Apply the same machinery — specifically DDPO or DPOK-style policy gradient methods — to fine-tune a financial time series diffusion model against a *financial reward*: e.g., Sharpe ratio of a strategy trained on the generated data, or a stylized-fact matching score.

**Why interesting**: This directly connects Part I (diffusion models) to Part II (RL, stochastic control) of the course. It is also more directly grounded in the Uehara et al. paper already on the syllabus, so you have a clear technical starting point. The question "can RL-based fine-tuning improve simulator quality for finance?" is practically important and not fully answered.

**Publication angle**: There is no published paper applying DDPO/DPOK specifically to financial simulators. The RL-from-feedback framing is novel in this domain.

---

### Idea 2: Continuous-Time RL for Optimal Execution (No TDA)

**Concept**: Use the continuous-time RL framework (Part II of the course) to solve an optimal execution problem — e.g., the Almgren–Chriss liquidation problem — but with a *learned* market impact model rather than a parametric one. The agent learns both the dynamics and the control policy from simulated or historical data using actor-critic methods in continuous time.

**Why interesting**: This is a clean application of HJB + model-free RL that has immediate financial relevance. The connection between the HJB equation and the Q-function in continuous time is mathematically rich, and there is active recent work (see Hambly, Xu & Yang on the syllabus) that you can build on.

**Publication angle**: Most continuous-time RL for finance papers use parametric models. Learning the impact model end-to-end while solving the control problem simultaneously is less explored.

---

### Idea 3: Mean Field Game for Systemic Risk with TDA Diagnostics (TDA + MFG)

**Concept**: Model a large population of financial agents (banks, funds) as a mean field game where agents choose leverage or investment strategies. Solve for the MFG equilibrium. Then use TDA — specifically persistent homology of the agent state distribution over time — as a diagnostic tool to detect when the system is near a systemic-risk tipping point (analogous to Gidea & Katz but in a controlled MFG setting).

**Why interesting**: This connects Part III (MFG) of the course with TDA in a way that is more analytical than empirical. The TDA component is a lens for understanding the MFG dynamics rather than a training loss, which may be more tractable.

**Publication angle**: MFG + TDA is essentially unexplored. The framing of topological early-warning signals in a mean field model is novel.

---

### Idea 4: Topological Signatures as Conditioning for Diffusion Models (TDA + Diffusion)

**Concept**: Rather than using topology as a *loss*, use it as a *conditioning signal*. Compute persistent homology features (e.g., Betti curves or persistence images) from historical data and condition the diffusion model on these features at generation time. This allows you to generate time series with a *specified topological profile* — e.g., "generate a bear market with two regimes."

**Why interesting**: This is architecturally simpler than a topological loss (no need to differentiate through PH), and gives you explicit control over the topology of generated series. It could be combined with the CSDI or Diffusion-TS frameworks.

**Publication angle**: Topological conditioning of generative models for time series is novel, and the financial application is well-motivated.

---

## My Recommendation

Your original idea (diffusion + topological loss) is the strongest from a novelty and publishability standpoint, and it connects naturally to both your TDA course and the course material on score-based diffusion. I would suggest narrowing the scope to:

1. A single asset class (e.g., equity returns).
2. A single filtration (sublevel-set filtration of the 1D return series, or sliding-window Vietoris–Rips).
3. Two topological loss terms: one for β₀ (regime structure) and one for β₁ (cyclical structure).
4. Evaluation on 3–4 stylized facts (heavy tails, volatility clustering, autocorrelation of squared returns, leverage effect) plus the TimeGAN discriminative/predictive scores.

This is achievable in the project timeline and gives you a clear story: *topological structure is a missing inductive bias in financial time series generation, and we show a principled way to incorporate it.*

The report deadline is June 12th — that gives you roughly 6–7 weeks, which is tight but workable if you start with a simple baseline (e.g., Diffusion-TS without the topological loss) in the first two weeks.
