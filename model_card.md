# Model Card: Attention-to-ANFIS (A2ANFIS)

## Model Details
- **Developed by:** Namal Rathnayake, Masashi Minamide, Michio Kawamiya, Yukinobu Hoshino  
- **Model Type:** Neuro-symbolic (Hybrid Deep Learning and Fuzzy Logic)  
- **Architecture:** Convolutional Encoder with Squeeze-and-Excitation (SE) blocks fused with an Adaptive Neuro-Fuzzy Inference System (ANFIS)  
- **Language:** Python (PyTorch)  
- **License:** MIT

## Intended Use
- **Primary Use:** Deterministic regression of moist convective intensity (vertical velocity $w$) in tropical cyclones  
- **Intended Users:** Meteorological researchers, disaster management agencies, and AI-weather model developers  
- **Out-of-Scope:** Generative texture synthesis or high-frequency cloud image generation (use GANs/Diffusion models for those tasks)

## Training Data
- **Dataset:** High-resolution mesoscale simulation of Hurricane Harvey (2017)  
- **Source:** Advanced Research Weather Research and Forecasting (WRF-ARW) model with EnKF data assimilation  
- **Input Features:** 3D specific humidity tensors ($X \in \mathbb{R}^{42 \times 15 \times 15}$) covering 42 vertical pressure levels  
- **Target Variable:** Log-transformed middle-tropospheric vertical velocity ($y = \log(w+1)$)

## Performance & Evaluation
- **Hardware:** NVIDIA GeForce RTX 5090 GPU (24GB GDDR7)  
- **Primary Metrics:** RMSE (0.3452), $R^2$ (0.8654)  
- **Key Findings:** A2ANFIS significantly outperforms generative baselines (ARCGAN, CCDM) in high-intensity regimes and maintains stability under extreme lead times ($T-60$ min)

## Limitations
- **Saturation Effect:** Model predictions for extreme events may plateau at lead times exceeding 40 minutes due to inherent atmospheric information loss  
- **Domain Specificity:** Currently validated on tropical cyclone moisture fields; performance on mid-latitude synoptic systems or dry convection has not yet been established  

## Ethical Considerations & Transparency
- **Epistemic Integrity:** Designed as a "Glass Box" architecture to prevent "epistemic hallucinations" common in stochastic generative AI  
- **Interpretability:** Internal decision-making is verifiable via learnable If-Then logic rules and channel-wise attention heatmaps  
