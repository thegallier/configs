# Advanced Feature Creation Methods for financial Time Series.

---

## 1. Introduction

Modern financial markets produce vast amounts of streaming, high-frequency data, which can be separated into three main components:

1. **Orders:** Limit order flow with timestamps, prices, sizes, cancellations, etc.  
2. **Levels:** Reconstructed limit order book snapshots at multiple price levels (including aggregated sizes and order IDs).  
3. **Trades:** Executed transactions with specific prices, volumes, and trade direction.

In this setting, we care about both:

- **Oran asset allocation**, assigning one weight per security.  
- **Classification tasks**, such as whether the best bid/offer goes up/down/unchanged, or whether the next trade is buy vs. sell.

Four key approaches are explored for modeling and analysis:

1. **Signature Methods (Rough Paths)** – capturing pathwise co-movements among price levels.  
2. **Matrix Motifs** – detecting recurring patterns in multi-level data.  
3. **Dynamic Time Warping (DTW)** – aligning similar but time-shifted sequences.  
4. **Time2Vec-based Transformers** – an attention-based architecture using Time2Vec encodings for asset allocation or classification tasks.

Additionally, **Transformers** (including the Time2Vec-based Portfolio Transformer) and **Mamba** (Bayesian/HPC framework) can focus on modeling orders and trades **conditioned on the current levels**. In this updated paper, we integrate the **Portfolio Transformer** approach presented by Kisiel & Gorse, which uses Time2Vec encoding for time-series input, and highlight how it can enhance attention-based asset allocation.

---

## 2. Methodologies

### 2.1 Signature Methods (Rough Paths)

- **Core Concept:**  
  Transform time series into iterated integrals (signatures) that capture pathwise features.

- **Relevance for Price Level Co-integration:**  
  Robust to capturing fine-grained dependencies and potential co-integration across price levels for one or multiple assets.

- **Drawbacks:**  
  Computational overhead can grow rapidly with high dimensionality; careful truncation or parallelization is often required.

---

### 2.2 Matrix Motifs

- **Core Concept:**  
  Convert time series into matrix profiles or recurrence plots to detect repeating “motifs.”

- **Relevance for Price Level Co-integration:**  
  Can reveal stable patterns or recurring shapes across price levels (or across multiple securities).

- **Drawbacks:**  
  Requires efficient indexing or dimension reduction for large-scale or streaming data.

---

### 2.3 Dynamic Time Warping (DTW)

- **Core Concept:**  
  Measures sequence similarity by non-linearly aligning two time series in the time dimension.

- **Relevance for Price Level Co-integration:**  
  Useful when multiple price levels move similarly but with slight time lags or speed differences.

- **Drawbacks:**  
  Naive DTW is \(O(n^2)\); advanced pruning or approximation may be necessary for real-time contexts.

---

### 2.4 Time2Vec-based Transformers

- **Core Concept (From Kisiel & Gorse, 2021):**  
  In their work, “*Portfolio Transformer for Attention-Based Asset Allocation*,” Kisiel & Gorse propose a Transformer-based architecture that uses **Time2Vec** to encode the time dimension.  
  - **Time2Vec**: An embedding that replaces or augments positional encoding by mapping time indices to a learned vector representation, capturing both linear and periodic time features.  
  - **Transformer Architecture**: Multi-head self-attention layers capable of learning dependencies across multiple assets or features in a sequence, crucial for dynamic asset allocation.

- **Relevance for Orders/Trades (Conditioned on Levels):**  
  - The Time2Vec-based Transformer can treat the reconstructed price levels as context, while focusing on sequences of orders or trades over time.  
  - The attention mechanism, combined with Time2Vec, captures important temporal dynamics without overly relying on standard positional encodings.

- **Strengths:**  
  - **Adaptive Time Embedding**: Time2Vec can better handle irregular or high-frequency sampling intervals, common in financial data.  
  - **Attention-based**: Transformers excel at capturing long-range dependencies and multi-asset relationships.  
  - **Direct Integration with Asset Allocation**: The architecture is specifically designed for portfolio optimization, making it highly relevant to Oran-style weight assignments.

- **Potential Limitations:**  
  - Complexity of Transformer models can be high, requiring significant computational resources.  
  - Sensitivity to hyperparameters (e.g., number of heads, embedding dimension, etc.) demands careful tuning.

---

## 3. Comparative Analysis for Asset Allocation & Classification

### 3.1 Accuracy

- **Signatures:**  
  Capture subtle co-integrations among price levels, beneficial for classification and regression tasks.

- **Matrix Motifs:**  
  Highly effective if repeating patterns define stable market regimes.

- **DTW:**  
  Aligns sequences with time distortions; can be accurate if partial matches are informative.

- **Time2Vec-based Transformers:**  
  Often achieve **state-of-the-art** results by leveraging attention over time, especially with well-chosen time embeddings that account for irregular sampling.

---

### 3.2 Computational Efficiency

- **Signatures:**  
  Incremental or GPU-based libraries help, but naive computations can be expensive in high dimensions.

- **Matrix Motifs:**  
  After initial profile calculations, repeated usage can be efficient, though massive dimension or streaming updates remain challenging.

- **DTW:**  
  \(O(n^2)\) complexity unless pruned or approximated.

- **Time2Vec-based Transformers:**  
  Transformers can be **\(O(n^2)\)** in sequence length for the attention mechanism. With large or high-frequency data, memory usage can be substantial, but recent sparse attention techniques can mitigate this.

---

### 3.3 Real-Time Predictive Capacity

- **Signatures:**  
  Suitable for rolling windows; incremental updates are feasible for streaming.

- **Matrix Motifs:**  
  Online matrix profile methods exist, but scaling to many securities in real time may require high-performance systems.

- **DTW:**  
  Streaming DTW approximations and local alignments are possible but still may be computationally intense.

- **Time2Vec-based Transformers:**  
  Achieving real-time performance requires careful batch sizing, streaming adaptation, or engineering solutions (e.g., chunked attention). Still, the model’s interpretability via attention weights can aid quick decision-making in asset allocation.

---

## 4. Key Literature and Code Repositories

Below are selected references (including the newly integrated **Time2Vec-based Transformer approach**) and code resources.

### 4.1 Signature Methods (Rough Paths)

**Papers:**
1. Lyons, T., & Ni, H. (2019). *Learning from the Past, Predicting the Statistics for the Future, Learning an Irreversible Dynamics*. Annual Reviews in Statistics.  
2. Chevyrev, I., & Kormilitzin, A. (2016). *A Primer on the Signature Method in Machine Learning*. arXiv:1603.03788.  
3. Morrill, D., Salvi, C., & Tindel, S. (2022). *Rough Path Techniques for Financial Time Series Analysis*. Quantitative Finance Review.  
4. Perez Arribas, I. (2018). *An Introduction to the Signature Method in Machine Learning for Finance*. SSRN Electronic Journal.  
5. Ni, H., Lyons, T., & Wang, Y. (2020). *Signature Methods in High-Frequency Finance*. Quantitative Analysis of Finance.

**GitHub Repositories:**
1. [**signatory**](https://github.com/patrick-kidger/signatory) – PyTorch library for iterated integrals.  
2. [**esig**](https://github.com/bottler/esig) – Python library for signature-based computations.  
3. [**DeepSignature**](https://github.com/crispitagorico/DeepSignature) – Tutorials/demos for neural networks with signature transforms.

---

### 4.2 Matrix Motifs

**Papers:**
1. Yeh, C.-C. M., Kavantzas, N., & Keogh, E. (2016). *Matrix Profile I: All Pairs Similarity Joins for Time Series*. IEEE ICDM.  
2. Zebende, G. F. (2019). *Recurrence Quantification Analysis and Matrix Motifs in Stock Markets*. Physica A.  
3. Silva, D. F., Batista, G. E., & Keogh, E. (2019). *Symbolic Matrix Profiles for Time Series Pattern Recognition*. Data Mining and Knowledge Discovery.  
4. Linardi, M., et al. (2018). *Matrix and Tensor Profiling for Multivariate Time Series*. SIAM International Conference on Data Mining.  
5. Wu, Y., & Chen, K. (2020). *Motif Discovery and Clustering in Financial Time Series Using Matrix Profiles*. Quantitative Finance & Economics.

**GitHub Repositories:**
1. [**matrixprofile**](https://github.com/matrix-profile-foundation/matrixprofile) – Tools for matrix profile calculations.  
2. [**stumpy**](https://github.com/TDAmeritrade/stumpy) – Efficient Python library for matrix profiles.  
3. [**mpx**](https://github.com/matrix-profile-foundation/mpx) – Online matrix profile libraries.

---

### 4.3 Dynamic Time Warping (DTW)

**Papers:**
1. Berndt, D. J., & Clifford, J. (1994). *Using Dynamic Time Warping to Find Patterns in Time Series*. AAAI Workshop.  
2. Ratanamahatana, C. A., & Keogh, E. (2004). *Everything You Know About Dynamic Time Warping is Wrong*. KDD.  
3. Zhang, J., & Wu, L. (2016). *DTW-Based Clustering in Financial Time Series Forecasting*. IEEE Transactions on Systems, Man, and Cybernetics.  
4. Paparrizos, J., & Gravano, L. (2015). *k-Shape: Efficient and Accurate Clustering of Time Series*. SIGMOD.  
5. Mori, U., Mendiburu, A., & Lozano, J. A. (2016). *Distance Measures for Time Series in R: Review and Extension to the Financial Domain*. R Journal.

**GitHub Repositories:**
1. [**dtw-python**](https://github.com/pierre-rouanet/dtw-python) – Python library for DTW.  
2. [**fastdtw**](https://github.com/slaypni/fastdtw) – Approximate DTW.  
3. [**tslearn**](https://github.com/tslearn-team/tslearn) – Toolkit for time-series clustering/classification.

---

### 4.4 Time2Vec-based Transformers

**Paper:**
- Kisiel, D., & Gorse, D. (2021). *Portfolio Transformer for Attention-Based Asset Allocation*. University College London, Department of Computer Science.

**Key Ideas:**
- **Time2Vec** encoding for improved temporal embeddings in attention-based models.  
- **Transformer** architecture specialized for portfolio optimization tasks.

**Possible Repositories (General Transformer + Time2Vec):**
1. [**Time2Vec reference code**](https://github.com/ojeda-e/time2vec-pytorch) – An unofficial PyTorch implementation of Time2Vec.  
2. [**Transformers**](https://github.com/huggingface/transformers) – Hugging Face library (not specifically for finance but a common base for attention models).

---

## 5. Integration with Transformers and Mamba

Recall that:

- **Signature Methods, Matrix Motifs, and DTW** primarily handle the *co-integration of price levels*.  
- **Transformers and Mamba** focus on *orders and trades conditioned on those levels*.

With the **Time2Vec-based Transformer** approach (from Kisiel & Gorse), we have an even more specialized Transformer variant for **asset allocation** and time-series modeling:

### 5.1 Transformers (Including Time2Vec-Based) for Orders and Trades

- **Conditioning on Price Levels:**  
  The Time2Vec-based Transformer can reference current limit order book (levels) while modeling the temporal structure of orders and trades.

- **Adaptive Time Encoding:**  
  Time2Vec enriches standard positional embeddings, allowing the model to handle irregular sampling intervals or emphasize cyclical/periodic patterns in intraday trading (e.g., auction open, midday lull, closing auction).

- **Attention Mechanism:**  
  Scales well for capturing multi-security or multi-feature relationships. Coupled with Time2Vec, it can more accurately weigh historical order/trade events relevant to the current context.

---

### 5.2 Mamba for Orders and Trades (Conditioned on Levels)

- **Bayesian Inference & Uncertainty:**  
  Probabilistic approaches help estimate the uncertainty around next-trade direction, best bid/offer movements, or optimal portfolio weights.

- **Time2Vec + Mamba Synergy:**  
  One could embed the Time2Vec-based Transformer in a Bayesian framework, learning distributions over attention weights, time-encoding parameters, or other hyperparameters critical for real-time risk assessments.

---

### 5.3 Signatures, Matrix Motifs, & DTW for Price-Level Co-integration

- These techniques remain potent for capturing multi-asset or multi-level dependencies. They can be used to derive features that feed the Time2Vec-based Transformer (or a standard Transformer + Mamba pipeline) for classification or regression tasks.

---

## 6. Three New Research Directions (Incorporating Time2Vec)

1. **Time2Vec-Enhanced Signatures for High-Frequency Co-Integration**  
   - Compute rough path signatures of price-level co-movements, then incorporate Time2Vec embeddings to encode the precise timing of each segment.  
   - This hybrid approach may better capture intraday seasonality or microstructure effects for real-time classification (e.g., next trade buy/sell) or weight allocation.

2. **Matrix Motif-Assisted Portfolio Transformer**  
   - Use matrix motifs to detect stable regimes or recurring patterns in price co-integration.  
   - Condition the Kisiel & Gorse Portfolio Transformer on motif labels or motif-based embeddings, allowing it to selectively focus on repeated intraday patterns that historically correlate with certain trades or price movements.

3. **Bayesian (Mamba) Time2Vec Transformer for Streamed Order/Trade Data**  
   - Integrate the Time2Vec-based Transformer within a Bayesian (Mamba) framework for continuous learning from streaming order book data.  
   - Posterior updates enable real-time re-estimation of model parameters, capturing shifts in market volatility or liquidity, and providing asset allocation weights with uncertainty estimates.

---

## 7. Conclusion

By adding the **Time2Vec-based Transformer**—as presented by Kisiel & Gorse—into the set of methods for financial time-series modeling, we enrich the toolkit for high-frequency market microstructure analysis and Oran-style asset allocation. Here is a summary of the four approaches:

1. **Signature Methods (Rough Paths):** Best for capturing pathwise co-integration among price levels.  
2. **Matrix Motifs:** Ideal for identifying recurring patterns, robust to moderate regime shifts.  
3. **Dynamic Time Warping (DTW):** Powerful for handling time misalignments, useful in partial sequence matching.  
4. **Time2Vec-based Transformers:** A cutting-edge, attention-based approach using specialized time embeddings—particularly effective for asset allocation and real-time order/trade modeling when conditioned on price levels.

In conjunction with **Mamba** (for Bayesian inference) and real-time streaming adaptations, these techniques offer a robust, multi-angle perspective on market data. They accommodate everything from microstructure-level classification (e.g., next trade direction) to macro-level portfolio optimization (e.g., assigning Oran weights). As data scales continue to grow, hybrid solutions—merging co-integration analysis (Signatures, Matrix Motifs, DTW) with advanced deep learning paradigms (Time2Vec-based Transformers + Mamba)—will likely shape the next frontier of intelligent, risk-aware financial modeling.
