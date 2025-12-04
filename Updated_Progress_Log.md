# URS Project Progress Log

**Project Theme:** Machine Learning for Virtual Sensor Development & Survival Analysis Across Datasets  
**Supervisor:** Dr. Gupta  
**Last Updated:** December 4, 2025

---

## 2025-09-18 (First Meeting)

### What I Planned
- Reading virtual sensor essays (overview stage)
- Set up Python + GitHub repo
- Download JANAF CO₂ dataset (C-095.txt)
- Try first visualization (Cp vs T)

---

## 2025-09-18 – 2025-09-25 (Week 1)

### What I Did

**Repository Setup:**
- Created GitHub repository with `README.md` and `progress.md`
- Implemented version control workflow (commits, pushes)
- Organized project structure: `data/`, `src/`, `output/` directories

**Data Acquisition & Processing:**
- Downloaded JANAF thermodynamic database: `data/C-095.txt` (CO₂)
- Added second molecule dataset: `data/C-093.txt` (CO)
- Extracted temperature-dependent thermodynamic properties

**Visualization Development:**
- Wrote `src/visualize_co2.py` → successfully plotted Cp (heat capacity) vs Temperature for CO₂
- Explored multiple thermodynamic properties: Cp, S (entropy), ΔfH (formation enthalpy), ΔfG (free energy) vs T
- Wrote `src/visualize_co_cp_vs_T.py` → Cp vs T plot for CO
- Generated PNG outputs for both molecules

**Key Accomplishments:**
- Successfully read, parsed, and visualized JANAF data files
- Learned GitHub workflow for scientific reproducibility
- Established foundation for thermodynamic property analysis

### Essay References Reviewed

1. Martin, D., Kühl, N., & Satzger, G. (2021). Virtual sensors. *Business & Information Systems Engineering*, 63(3), 315–323.
2. Albertos, P., & Goodwin, G. C. (2002). Virtual sensors for control applications. *Annual Reviews in Control*, 26(1), 101–112.

### Next Steps
- Deepen understanding of virtual sensor principles
- Prepare for supervised machine learning methods

---

## 2025-09-25 – 2025-10-02 (Week 2)

### What I Did

**1. Learning Supervised Machine Learning (SML): Regression**

- Line-by-line review of [Google Colab tutorial](https://colab.research.google.com/drive/1FfikNXcsL1t77IHjIh0tLhHhvhrnKHir)
- Implemented exploratory data analysis (EDA) techniques
- Created heatmap and pairplot to visualize feature correlations
- Learned normalization and data preprocessing fundamentals

**2. Kaggle Dataset Exploration: Medical Insurance Cost Dataset**

**Dataset Overview:**
- 1,338 records with 6 features (age, sex, BMI, children, smoker, region)
- Target: insurance charges (continuous regression)
- Identified missing values and data quality issues

**Visualizations Generated:**
- Scatter plot: Age vs Charges (identified three distinct clusters)
- Box-whisker plots: Charges distribution by categorical features
- Heatmap: Feature correlation matrix
- Histograms: Individual feature distributions

### Key Insights from Meeting

**New Visualization Type:** Discovered violin plots as superior to box plots for dense distributions
- Combines box plot structure with kernel density estimation
- Better reveals multimodal distributions in data

**Library Insights:**
- Seaborn: Preferred for humanities research, higher-level abstractions
- Matplotlib: More granular control, better for publication-quality figures

**Pattern Discovery:**
- Identified three distinct charge clusters in Age vs Charges scatter plot
- Cluster 1 (15–25k): Younger non-smokers with relatively stable charges
- Cluster 2 (30–35k): Middle-aged individuals with moderate charges
- Cluster 3 (40–65k): Smokers with dramatically higher charges

### Next Steps
1. **Advanced Modeling Strategy (Charges vs Age):**
   - Set three thresholds based on observed clusters
   - Perform separate linear regression for each segment
   - Validate piecewise model vs. single unified model

2. **Broader Data Exploration:**
   - Explore correlations among all features
   - Run baseline linear regression analyses (BMI, smoker status, family status)
   - Identify non-linear relationships

---

## 2025-10-02 – 2025-10-09 (Week 3)

### What I Did

**Categorical Data Visualization:**
- Generated violin plots: Charges distribution by sex, smoker status, region
- Box-whisker analysis: How categorical variables affect charge levels
- Stacked bar charts: Feature composition across charge ranges

**Segmented Regression Analysis:**
- Implemented piecewise linear regression with cluster-based thresholds:
  - Segment 1: Charges 0–17k (young, non-smokers)
  - Segment 2: Charges 15–32k (middle-aged, non-smokers)
  - Segment 3: Charges 31–65k (smokers and high-BMI)
- Explored quadratic regression within segments to capture non-linearity
- Calculated segment-specific R² and MAE

**Pattern Recognition:**
- Identified BMI threshold of 30 as significant predictor for charge increases
- Discovered smoker status as the single strongest predictor
- Quantified age-charge interaction effects

### Key Concepts Learned
- When to use piecewise vs. unified regression models
- Non-linear relationships and polynomial feature engineering
- Threshold identification and domain interpretation

### Next Steps
- Transition to Artificial Neural Networks (ANN) for capturing complex non-linear relationships
- Prepare to build multi-layer neural networks

---

## 2025-10-09 – 2025-10-30 (Weeks 4 & 5)

### What I Did

**1. Data Preprocessing**

**Categorical Feature Engineering:**
- Applied one-hot encoding to sex, smoker status, and region
- Converted categorical variables to binary/numeric representations
- Handled missing values through appropriate imputation

**Normalization:**
- Implemented Z-score normalization using StandardScaler
- Normalized all 8 features to mean = 0, standard deviation = 1
- Critical for neural network training convergence

**Data Splitting:**
- Training set: 1,070 samples (80%)
- Test set: 268 samples (20%)
- Random seed: 42 (reproducibility)

**2. ANN Model Development**

**Architecture Design – Information Funnel Principle:**
```
Input Layer:     8 neurons (normalized features)
Hidden Layer 1:  64 neurons (ReLU activation)
Hidden Layer 2:  32 neurons (ReLU activation)
Hidden Layer 3:  16 neurons (ReLU activation)
Output Layer:    1 neuron (linear activation)
─────────────────────────────────────
Total Parameters: 3,201
```

**Rationale:**
- Progressive neuron reduction (8 → 64 → 32 → 16 → 1) implements information funnel principle
- Early layers capture broad feature patterns; later layers compress to actionable predictions
- ReLU activation introduces non-linearity without vanishing gradient problems
- Linear output for continuous regression task

**Training Configuration:**
- Optimizer: Adam (adaptive learning rates)
- Loss Function: Mean Squared Error (MSE)
- Epochs: 100 (stopped early at epoch 40 when validation loss plateaued)
- Batch Size: 32

**3. Model Performance Metrics**

| Metric | Value | Interpretation |
|--------|-------|---|
| **R² Score** | 0.8349 | Model explains 83.49% of charge variance |
| **RMSE** | $5,063.29 | Average prediction deviation |
| **MAE** | $3,355.92 | Typical absolute error per prediction |
| **Mean % Error** | 42.26% | Average percentage deviation |
| **Median % Error** | 24% | Typical percentage error (lower bound) |

**Performance Breakdown:**
- Strong accuracy (R² > 0.83) on low-to-mid charge cases ($15–40k)
- Higher errors on extreme cases ($40–65k range, largely smokers)
- Bias towards underestimating high charges (tendency to predict closer to mean)

**4. Generated Outputs**

**Visualizations:**
- `training_history.png` – Loss curves showing smooth convergence
- `actual_vs_predicted.png` – Scatter plot with R² = 0.8349 and fitted line
- `error_distribution.png` – Histograms of absolute and percentage errors
- `ann_architecture.png` – Network structure diagram with layer specifications

**Data & Models:**
- `insurance_predictions_comparison.csv` – Detailed predictions for all 268 test cases
- `insurance_ann_model.h5` – Saved trained model (TensorFlow/Keras format)
- `complete_ann_model.py` – Full, reproducible implementation pipeline

**Documentation:**
- Comprehensive written report covering:
  - ANN fundamentals (forward propagation, backpropagation, activation functions)
  - Architecture design decisions and information funnel principle
  - Step-by-step code walkthrough with explanations
  - Interpretation of all visualizations and metrics
  - Model strengths, limitations, and real-world applications

### Key Concepts Learned

**Neural Network Fundamentals:**
- **Forward Propagation:** Data flows left-to-right through layers
- **Backpropagation:** Gradients flow right-to-left to update weights
- **Activation Functions:** ReLU for hidden layers (non-linearity), linear for regression output

**Information Funnel Principle:**
- Progressive neuron reduction creates a "bottleneck" architecture
- Earlier layers extract general patterns; later layers specialize
- Compression encourages learning of essential features
- Reduces overfitting by limiting model capacity

**Normalization Importance:**
- Unscaled features can dominate learning (e.g., BMI 20–30 vs Age 18–65)
- StandardScaler ensures all features contribute equally
- Dramatically improves convergence speed and stability

**Model Evaluation Metrics:**
- R²: Overall fit quality (0–1 scale)
- RMSE: Penalizes large errors more than MAE
- MAE: Interpretable in original units
- Percentage error: Domain-dependent evaluation

**Error Analysis:**
- Identified systematic bias in high-charge predictions
- Understood error distribution (normal-ish, with right tail)
- Connected errors to data patterns (smoker status dominance)

### Next Steps
- Apply ANN methodology to **two additional Kaggle datasets** (regression and classification)
- Explore advanced techniques:
  - Dropout regularization (prevent overfitting)
  - Early stopping validation
  - Feature importance analysis
  - Hyperparameter tuning (hidden layer sizes, learning rate)
- Compare different architectures and optimization strategies

---

## 2025-11-01 – 2025-11-06 (Week 6)

### What I Did

#### **Dataset #1: Restaurant Tips Prediction (Regression)**

**Exploratory Data Analysis:**

- **Dataset:** 244 restaurant transactions from the Seaborn tips dataset
- **Features:** total_bill (USD), tip (USD), sex, smoker status, day, time, party size
- **Target:** Predict tip amount (continuous)

**Comprehensive Visualizations Generated:**

1. **Scatter Plots:**
   - Total bill vs tip (R² = 0.4566, colored by party size)
   - Revealed linear trend with significant scatter
   - Party size indicated via color gradient

2. **Box Plots:**
   - Tip amount by party size (1–6 people)
   - Showed median trends and outliers
   - Revealed increasing spread with larger parties

3. **Tip Percentage Analysis:**
   - Calculated tip% = (tip / total_bill) × 100
   - Discovered counter-intuitive pattern: larger parties tip lower percentage

4. **Correlation Heatmap:**
   - Identified top predictors: total_bill (r = 0.676), party_size (r = 0.489)
   - Gender and smoking status nearly uncorrelated (|r| < 0.1)

5. **Violin Plots:**
   - Tip distributions across sex, smoker status, day, time
   - Revealed multimodal distributions and outliers

**Key Findings:**

| Discovery | Value | Interpretation |
|-----------|-------|---|
| Solo diners avg tip% | 21.73% | Highest percentage despite smallest bills |
| Tip increase per person | +$0.71/person | Absolute increase with party size |
| Tip% decrease per person | -0.92%/person | Percentage decreases with party size |
| Gender-tip correlation | r < 0.1 | Negligible relationship |
| Smoker-tip correlation | r < 0.1 | No significant effect |
| Day-Time correlation | r = 0.874 | Strong multicollinearity |

**ANN Model Development:**

**Architecture – Two Hidden Layers:**
```
Input:     7 features (normalized)
Hidden 1:  32 neurons (ReLU)
Hidden 2:  16 neurons (ReLU)
Output:    1 neuron (linear)
```

**Experiments:**
- Basic model: Used only raw features
- Feature-engineered model: Added interaction terms (bill × party_size, etc.)
- Compared validation performance to linear regression baseline

**Generated Visualizations:**
- Model loss and MAE during training
- Overfitting indicator: Validation loss vs training loss
- Predictions vs actual with regression baseline overlay
- Learning curves for 50+ epochs

**Model Performance Comparison:**

| Model | R² | MAE (USD) | Winner | Reason |
|-------|-----|---------|--------|---------|
| **Linear Regression** | 0.46 | $0.75 | ✅ | Linear relationships dominate |
| **Basic ANN** | 0.18 | $0.83 | ❌ | Underfitting; insufficient data |
| **Feature-Engineered ANN** | 0.22 | $0.81 | ❌ | Marginal improvement |

**Critical Insight:**
With only 244 samples and a simple linear relationship (Pearson r = 0.68), the ANN architecture could not justify its complexity. **Conclusion:** Neural networks require ~1000+ samples to overcome the bias-variance tradeoff and outperform linear models.

---

#### **Dataset #2: Titanic Survival Prediction (Binary Classification)**

**Exploratory Data Analysis:**

- **Dataset:** 891 passengers from the Titanic, predicting survival (Alive/Dead)
- **Target Variable:** Patient_Status (binary classification)
- **Features:** 12 original features (age, sex, class, fare, family, etc.)

**Data Cleaning:**
- Imputed 177 missing Age values (19.9% of dataset) using median strategy
- Dropped high-cardinality features (Cabin codes, Ticket numbers, full names)
- Final feature count: 11 after one-hot encoding

**Generated 7 Comprehensive Visualizations:**

1. **Box Plots:**
   - Age distribution by survival status
   - Survived median: 28.0 years | Deceased median: 28.0 years
   - Minimal age difference between groups

2. **Stacked Bar Charts (Survival Rates by Feature):**
   - Gender: Females 74.2%, Males 18.9% (55.3% gap!)
   - Passenger Class: 1st=63.0%, 2nd=47.3%, 3rd=24.2% (hierarchical)
   - Age Groups: Children 0–12 (60–68%), Elderly 60+ (21%)

3. **Violin Plots:**
   - Fare distribution by survival status
   - Survived group had higher median fare

4. **Correlation Heatmap:**
   - Feature importance ranking for survival

**Key Findings:**

| Factor | Effect | Correlation | Strength |
|--------|--------|-------------|----------|
| **Gender** | Females 74.2%, Males 18.9% | Massive gap (55.3%) | **Dominant** |
| **Passenger Class** | 1st→2nd→3rd (63%→47%→24%) | r = -0.338 | Moderate-strong |
| **Age (0–12 years)** | 60–68% survival | Favorable for children | Strong |
| **Age (60+ years)** | ~21% survival | Unfavorable for elderly | Negative |
| **Fare** | Higher fare → higher survival | r = +0.257 | Weak-moderate |
| **Family Composition** | Solo (30%), Small (45%), Large (17%) | Weak predictor | Marginal |

**Survival Statistics:**
- Total passengers: 891
- Survived: 342 (38.4%)
- Died: 549 (61.6%)

---

**ANN Classification Model Development:**

**Architecture – Three Hidden Layers with Regularization:**
```
Input:      11 features (normalized)
Hidden 1:   32 neurons (ReLU)
Hidden 2:   16 neurons (ReLU)
Hidden 3:    8 neurons (ReLU)
Output:      1 neuron (Sigmoid for binary classification)
```

**Training Configuration:**
- Optimizer: Adam with adaptive learning rates
- Loss Function: Binary crossentropy
- Epochs: 121 total (early stopping triggered at ~80 epochs)
- Early Stopping: Patience = 30, monitored validation loss
- Regularization: L2 (alpha = 0.01) to prevent overfitting
- Batch Size: 32

**Generated 4 Critical Evaluation Visualizations:**

1. **Training History:**
   - Loss curves (training vs validation) showing convergence at ~80 epochs
   - MAE curves across all epochs
   - Clear inflection point where validation loss plateaus

2. **Overfitting Indicator:**
   - Gap analysis between training and validation loss
   - Final gap = 0.10 (mild-to-moderate overfitting)
   - Indicates model is learning some noise but still generalizes

3. **Confusion Matrices:**
   - ANN vs Logistic Regression comparison
   - True Positives, True Negatives, False Positives, False Negatives

4. **ROC Curves (Receiver Operating Characteristic):**
   - Threshold-independent performance evaluation
   - AUC comparison between ANN and Logistic Regression
   - Shows discrimination ability across all thresholds

**Model Performance Comparison:**

| Metric | ANN | Logistic Regression | Winner | Notes |
|--------|-----|---------------------|--------|-------|
| **Accuracy** | 80.45% | 80.45% | Tie | Both equally accurate |
| **Precision** | 0.8269 | 0.7931 | **ANN** | Fewer false positives |
| **Recall** | 0.6232 | 0.6667 | **Logistic Reg** | Catches more survivors |
| **F1 Score** | 0.7107 | 0.7244 | **Logistic Reg** | Balanced metric favors LogReg |
| **ROC AUC** | 0.8526 | 0.8433 | **ANN** | Marginal ANN advantage |

**Performance Interpretation:**

- **Accuracy (80%):** Both models match baseline (if always predicting "survived")
- **Precision (0.83 ANN):** When model predicts survival, it's correct 83% of the time (ANN is conservative)
- **Recall (0.62–0.67):** Models catch 62–67% of actual survivors; miss 33–38% (false negatives)
- **F1 Score (0.71):** Balanced precision-recall metric; Logistic Regression slightly better
- **ROC AUC (0.85):** Excellent discrimination ability; both models significantly outperform random (0.50)

**Critical Insight – Precision vs Recall Trade-off:**
- **ANN is more conservative:** Higher precision (fewer false alarms) but lower recall (misses more survivors)
- **Logistic Regression is more liberal:** Lower precision (more false alarms) but higher recall (catches more survivors)
- **Application decides:** In Titanic context, catching more survivors (high recall) may be more important than avoiding false alarms

---

### Key Concepts Learned

#### **Statistical Analysis Techniques**

- **Multicollinearity Detection:** Identified Day-Time correlation (r = 0.874) requiring feature handling
- **Tip Percentage Normalization:** Controlled for bill amount to fairly compare tipping behavior across groups
- **ROC Curve Interpretation:** Threshold-agnostic performance evaluation; AUC measures ranking discrimination
- **Confusion Matrix Analysis:** Understanding TN, TP, FN, FP to diagnose model behavior

#### **Neural Network Fundamentals**

- **Overfitting Mechanisms:** Why validation loss diverges from training loss (learning noise, not signal)
- **Early Stopping Rationale:** Prevents memorization of noise; stops at optimal point before degradation
- **Feature Engineering Importance:** Interaction terms can improve performance; necessity depends on data size
- **Small Dataset Limitations:** ANNs require sufficient data (1000+ samples) to beat simpler models; underfitting with 244 samples
- **Binary Classification Output:** Sigmoid activation produces probability (0–1); threshold (default 0.5) determines decision

#### **Model Comparison Techniques**

- **When to Use ANN vs Linear/Logistic Regression:**
  - 1000+ samples: ANN likely wins
  - 100–500 samples: Linear/Logistic Regression competitive
  - < 100 samples: Linear/Logistic Regression preferred
  
- **Precision vs Recall Trade-offs:**
  - High precision: Minimize false positives (costly mistakes)
  - High recall: Minimize false negatives (missing positives)
  - Application determines which matters more

- **ROC AUC as Threshold-Agnostic Metric:**
  - Accuracy alone is misleading (especially imbalanced datasets)
  - AUC measures ranking ability across ALL thresholds
  - Immune to threshold changes; robust comparison

#### **Visualization Mastery**

- **Plot Selection:**
  - Box plots: Distribution and outliers for grouped data
  - Violin plots: Probability density + quartiles for dense distributions
  - Stacked bar charts: Composition and percentages across categories
  - Heatmaps: Correlation matrices and feature importance

- **Training Curves:** Diagnosing convergence, overfitting, underfitting
- **ROC Curves:** Comparing model discrimination across decision thresholds
- **Confusion Matrices:** Error breakdown (TP, TN, FP, FN)

---

### Documentation Generated

#### **Tips Dataset:**
- **Report:** `URS Kaggle Tip dataset analysis and ANN model development.docx`
  - Executive summary with business recommendations
  - Complete EDA with statistical analysis
  - Model development and comparison with baseline
  - Actionable insights for restaurant operations

- **Code Files:**
  - `src/ANN_tip.py` – Basic ANN implementation
  - `src/tip_visualizations.py` – Comprehensive EDA visualizations
  - Graphs: Scatter plots, box plots, violin plots, heatmap, training history, overfitting analysis

#### **Titanic Dataset:**
- **Report:** `URS Kaggle Titanic dataset data and ANN model development.docx`
  - Full classification analysis with historical context
  - Data cleaning and missing value imputation strategy
  - Comprehensive EDA (7 visualizations)
  - Model evaluation with confusion matrices and ROC curves

- **Code Files:**
  - `src/titanic_ann_classification.py` – Complete ANN pipeline with all visualizations
  - Graphs: 7 EDA plots + 4 model evaluation plots (training history, overfitting, confusion matrices, ROC curves)

---

## 2025-11-08 – 2025-11-25 (Week 7 & 8)

### What I Did

**Meeting Preparation (Nov 11):**
- Reviewed broad overview of three datasets: Breast Cancer (BRCA), Himalayan Climbing, and Thermodynamic (Virtual Sensor)
- Compiled initial exploratory analysis across all datasets
- Prepared summary visualizations for discussion with Dr. Gupta

---

## 2025-11-25 – 2025-12-04 (Week 9 & 10)

### What I Did

#### **Focus: Diesel Engine Virtual Sensor Development (Thermodynamic Dataset)**

**1. Dataset Understanding & Preprocessing**

**Engine Combustion Data Overview:**
- 217 engine operating points (samples)
- 13 potential input sensors measured in real-time:
  - **6 Key Inputs:** Torque, p_0, T_IM, P_IM, EGR_Rate, ECU_VTG_Pos
  - **7 Potential Additional Inputs:** MF_FUEL, p_21, p_31, T_21, T_31, q_MI, Rail Pressure
  
- **3 Target Outputs (virtual sensor predictions):**
  - MF_IA: Intake air mass flow (kg/h)
  - NOx_EO: Engine-out NOx emissions (ppm)
  - SOC: Start of combustion angle (deg)

**Data Processing Pipeline:**
- Loaded raw data: `Data_vaibhav_colored.csv`
- Exploratory analysis to understand feature ranges and distributions
- Created processed dataset: `df_processed.csv` with normalized features

**2. Comparative Model Analysis: 6 Key Inputs vs 13 All Inputs**

**Experiment Design:**
Built two ANN models to determine if additional 7 sensors provide value:

**Model A (6-Input Baseline):**
```
Architecture: 6 → 64 → 32 → 16 → 1 (multi-output to 3 targets)
Training: 172 samples (80%)
Testing: 45 samples (20%)
```

**Model B (13-Input Extended):**
```
Architecture: 13 → 128 → 64 → 32 → 1 (multi-output to 3 targets)
Same training/test split
```

**Performance Comparison Results:**

| Output | Metric | 6-Input Model | 13-Input Model | Difference | Verdict |
|--------|--------|---------------|---|---|---|
| **MF_IA** | R² (Test) | 0.9945 | 0.9970 | +0.0025 (0.25% gain) | ✅ Marginal |
| | MAE | 22.6 kg/h | 16.1 kg/h | -28.5% (better) | — |
| **NOx_EO** | R² (Test) | 0.9891 | 0.9912 | +0.0021 (0.21% gain) | ✅ Marginal |
| | MAE | 29.8 ppm | 25.1 ppm | -15.6% (better) | — |
| **SOC** | R² (Test) | 0.9841 | 0.9802 | **-0.0039 (loss!)** | ❌ Worse |
| | MAE | 0.27 deg | 0.29 deg | **+5.8% (worse)** | ❌ Worse |

**Average Performance:**
- 6-Input Average R²: **0.9892**
- 13-Input Average R²: **0.9894**
- **Improvement: <0.02% (negligible!)**

**Key Insight – SOC Degradation:**
The 13-input model actually **worsens** performance for SOC:
- R² drops by 0.39% (0.9841 → 0.9802)
- MAE increases by 5.8% (0.27 → 0.29 deg)
- Clear indicator of **overfitting** on small dataset (217 samples)

**3. Analysis Visualizations Generated**

**Pair Plots (Feature-Output Relationships):**
- `pairplot_MF_IA.jpg` – MF_IA with all 6 key inputs
- `pairplot_NOx_EO.jpg` – NOx_EO with all 6 key inputs
- `pairplot_SOC.jpg` – SOC with all 6 key inputs

**Model Performance Visualizations:**
- `viz_1_predicted_vs_actual_clear_circles.jpg` – 6-input model scatter plot (R² = 0.9945)
- `viz_2_confusion_matrix_classification.jpg` – Classification metrics comparison
- `viz_3_mae_comparison.jpg` – MAE across outputs for both models
- `viz_4_r2_comparison.jpg` – R² comparison: 6-input vs 13-input
- `viz_5_residuals_distribution.jpg` – Error distribution analysis
- `viz_6_metrics_heatmap.jpg` – Heatmap of all performance metrics

**4. Virtual Sensor Design Decision**

**Recommendation: USE ONLY 6 KEY INPUTS**

**Rationale:**

1. **Information Sufficiency:**
   - 6 key inputs capture >99% of predictive information
   - 7 additional inputs are largely redundant (high correlation with key inputs)
   - Each output achieves R² > 0.98, exceeding industrial standards

2. **Overfitting Evidence:**
   - SOC performance degrades with 13 inputs (clear overfitting sign)
   - Only 217 samples; adding 7 features increases parameters, risk of memorizing noise
   - Physics-based: 6 inputs represent complete thermodynamic state (load, air path, EGR, turbo)

3. **Cost-Benefit Analysis:**

   **Adding 7 Additional Physical Sensors:**
   - Hardware cost: $1000–2000+ per vehicle
   - Installation complexity and failure points
   - Calibration and maintenance burden
   - No meaningful accuracy improvement (SOC actually worsens)

   **Recommended Solution:**
   - Deploy 6-input model: $0 additional hardware
   - Achieves 0.9892 R² average (98.92%)
   - Simpler, more robust, production-ready

**5. Multi-Architecture Neural Network Framework**

Designed 4-tier virtual sensor system (all using 6 key inputs):

```
Tier 1: MLP (Primary real-time sensor)
├─ Runs every engine cycle (10ms)
├─ Predicts MF_IA, NOx_EO, SOC
├─ Latency: <1ms
└─ ECU-embedded firmware

Tier 2: LSTM (Temporal drift monitoring)
├─ Hourly analysis of 10-reading sequences
├─ Detects sensor degradation
├─ Background monitoring
└─ Cloud-based processing

Tier 3: Ensemble (Uncertainty quantification)
├─ 5 identical MLPs with different seeds
├─ Provides mean ± confidence intervals
├─ High-confidence prediction validation
└─ Optional on-board or cloud

Tier 4: Autoencoder (Anomaly detection)
├─ Monitors 6 input sensor health
├─ Detects abnormal operating points
├─ Real-time background monitoring
├─ ECU-embedded firmware
└─ Raises maintenance alerts
```

**6. Documentation Generated**

- **`Virtual_Sensor_Multi_Architecture.md`** – Comprehensive multi-tier design guide with code examples
- **`Virtual_Sensor_KeyInputs_Rewrite.md`** – Focus on 6-key-input design with full justification
- **Model Comparison Report** – Detailed 6 vs 13 input analysis with visualizations
- **`ANN_All_Inputs_Complete.py`** – Full training pipeline for multi-output models
- **Pickle Files** – Saved trained model weights for deployment

### Key Concepts Learned

**Virtual Sensor Development:**
- Feature sufficiency analysis using comparative modeling
- Multi-input ANN design for simultaneous multi-output prediction
- Overfitting detection in small datasets
- Cost-benefit analysis for adding hardware vs software

**Deep Learning Architecture Design:**
- Information funnel principle for multi-layer networks
- Multi-output regression with shared hidden layers
- Trade-offs between model complexity and data size
- Regularization necessity with limited samples

**Neural Network for Physical Systems:**
- Mapping physics-based sensor inputs to predicted outputs
- Validation through pairplots and residual analysis
- ECU deployment constraints (latency, memory, power)
- Real-time prediction requirements

**Production Readiness Criteria:**
- Model accuracy >98% exceeds industrial sensor tolerances (±3–5%)
- Simplified input set reduces failure modes
- Multi-tier architecture provides redundancy and monitoring
- Clear deployment path from research to fleet

---

## 2025-12-04 (Meeting Update)

### Project Status Summary

**Completed:**
1. ✅ Supervised Machine Learning: Regression (Insurance dataset)
2. ✅ Supervised Machine Learning: Binary Classification (Titanic dataset)
3. ✅ Multi-dataset exploration (Tips, Insurance, Titanic, Breast Cancer BRCA overview)
4. ✅ Virtual sensor design using diesel engine thermodynamic data
5. ✅ Comparative analysis: 6-input vs 13-input neural network models
6. ✅ Multi-architecture framework design (MLP, LSTM, Ensemble, Autoencoder)

**Key Achievements:**
- Built production-grade ANN models with 98%+ accuracy
- Developed comprehensive visualization and EDA skills
- Mastered model evaluation metrics (R², MAE, precision, recall, AUC)
- Designed practical virtual sensor replacing 3 physical sensors
- Documented complete implementation roadmaps

**Learning Progression:**
- Week 1–3: Foundational supervised learning
- Week 4–5: ANN development with information funnel architecture
- Week 6: Multi-dataset comparison (regression + classification)
- Week 7–10: Virtual sensor development with multi-architecture framework

**Next Steps (Post-Meeting):**
1. Implement multi-architecture virtual sensor (Phases 1–4)
2. Extend to Breast Cancer BRCA dataset (survival prediction)
3. Himalayan climbing dataset analysis (environmental factors on survival)
4. Production deployment planning for virtual sensor
5. Advanced techniques: Dropout, batch normalization, hyperparameter optimization

---

## Technical Skills Summary

| Category | Skills | Proficiency |
|----------|--------|------------|
| **Python Libraries** | Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn | ⭐⭐⭐⭐⭐ |
| **ML Algorithms** | Linear Regression, Logistic Regression, ANN (MLPs), LSTM, Autoencoder | ⭐⭐⭐⭐ |
| **Data Processing** | Normalization, one-hot encoding, missing value imputation, feature engineering | ⭐⭐⭐⭐⭐ |
| **Model Evaluation** | R², MAE, RMSE, Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrices | ⭐⭐⭐⭐⭐ |
| **Visualization** | EDA plots, box plots, violin plots, heatmaps, training curves, ROC curves | ⭐⭐⭐⭐⭐ |
| **Research Methods** | Experimental design, comparative analysis, validation strategies, documentation | ⭐⭐⭐⭐ |
| **GitHub & Version Control** | Repository management, commits, reproducibility | ⭐⭐⭐⭐ |
| **Deployment Thinking** | ECU constraints, latency requirements, production readiness | ⭐⭐⭐⭐ |

---

## Research Questions Addressed

1. ✅ **How do we build accurate neural networks for regression and classification?**
   - Answered through Insurance (regression) and Titanic (classification) projects

2. ✅ **Can additional sensors improve virtual sensor accuracy?**
   - Answered with 6-input vs 13-input comparison: No (overfitting, negligible gains)

3. ✅ **How do we predict survival across different domains?**
   - Partial (Titanic); more analysis needed (BRCA, Himalayan)

4. ⏳ **What are best practices for multi-architecture neural network design?**
   - Designed framework; implementation in progress

5. ⏳ **How does technology, time period, and socioeconomic structure affect survival?**
   - BRCA dataset shows HER2+ targeted therapy effect (technology factor)
   - Titanic shows gender and class effects (socioeconomic factor)
   - Himalayan data pending (environmental factor)

---
