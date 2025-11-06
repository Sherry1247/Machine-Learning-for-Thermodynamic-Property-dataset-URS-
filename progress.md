# URS Project Progress log

## 2025-09-18 (First Meeting)
### What I Planned
- Reading virtual sensor essays (overview stage)
- Set up Python + GitHub repo
- Download JANAF CO₂ dataset (C-095.txt)
- Try first visualization (Cp vs T)

---

## 2025-09-18 -- 2025-09-25 (Week 1)
### What I Did
- Created repo with `README.md` and `progress.md`
- Added `data/C-095.txt` (CO₂ JANAF table)
- Wrote `src/visualize_co2.py` → successfully plotted Cp vs T
- Learned how to commit + push files to GitHub
- Explored multiple plots (Cp, S, ΔfH, ΔfG vs T)
- Added second dataset `data/C-093.txt` for CO
- Wrote `src/visualize_co_cp_vs_T.py` → plotted Cp vs T for CO

### Essay References
1. Martin, D., Kühl, N., & Satzger, G. (2021). Virtual sensors. *Business & Information Systems Engineering*, 63(3), 315-323.
2. Albertos, P., & Goodwin, G. C. (2002). Virtual sensors for control applications. *Annual Reviews in Control*, 26(1), 101-112.

---

## 2025-09-25 -- 2025-10-02 (Week 2)
### What I Did
1. **Learning supervised machine learning (SML): regression**
   - Reviewed each line of code in [Google Colab tutorial](https://colab.research.google.com/drive/1FfikNXcsL1t77IHjIh0tLhHhvhrnKHir)
   - Created heatmap and pairplot to visualize data

2. **Explored Kaggle dataset: [Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)**
   - Created multiple visualizations: scatter plot, box-whisker plot, heatmap, histogram

### Key Insights from Meeting
- **New Visualization Type:** Learned about violin plots for displaying data distribution
- **Library Differences:** Seaborn (preferred in humanities research) vs. Matplotlib
- **Pattern Discovery:** Identified three distinct clusters in Charges vs. Age scatter plot

### Next Steps
1. **Advanced Modeling Strategy (Charges vs. Age)**
   - Set three thresholds based on observed clusters
   - Perform separate linear regression for each segment
2. **Broader Data Exploration**
   - Explore correlations among all variables
   - Run baseline regression analyses (BMI, smoker status, etc.)

---

## 2025-10-02 -- 2025-10-09 (Week 3)
### What I Did
- **Categorical Visualization:** Generated violin plots for charges vs sex, smoker, and region
- **Segmented Regression:** Applied piecewise linear (and quadratic) regression with data segmentation (0-17k, 15k-32k, 31k-60k)
- **Pattern Recognition:** Identified BMI threshold of 30 as significant predictor for charge increases

### Next Steps
- Delve deeper into Artificial Neural Networks (ANN)

---

## 2025-10-09 -- 2025-10-30 (Weeks 4 & 5)
### What I Did

**1. Data Preprocessing**
- Applied one-hot encoding for categorical variables (sex, smoker, region)
- Implemented Z-score normalization (StandardScaler) for all features
- Split data: 1,070 training samples (80%), 268 test samples (20%)

**2. ANN Model Development**
- Built neural network with **information funnel architecture**:
  - Input: 8 features
  - Hidden layers: 64 → 32 → 16 neurons (ReLU activation)
  - Output: 1 neuron (linear activation)
  - Total parameters: 3,201
- Trained for 100 epochs using Adam optimizer and MSE loss
- Model converged around epoch 40

**3. Model Performance**
- **R² Score:** 0.8349 (explains 83.49% of variance)
- **RMSE:** $5,063.29
- **MAE:** $3,355.92
- **Mean Percentage Error:** 42.26% (median: 24%)
- Strong performance on low-to-mid cost cases; higher errors on extreme/rare cases

**4. Generated Outputs**
- `training_history.png` - Learning curves showing convergence
- `actual_vs_predicted.png` - Scatter plot with R² = 0.8349
- `error_distribution.png` - Histograms of absolute and percentage errors
- `ann_architecture.png` - Network structure diagram
- `insurance_predictions_comparison.csv` - Detailed prediction analysis for all 268 test cases
- `insurance_ann_model.h5` - Saved trained model
- `complete_ann_model.py` - Full implementation pipeline

**5. Documentation**
- Wrote comprehensive report explaining:
  - ANN fundamentals and architecture design
  - Step-by-step code walkthrough
  - Interpretation of all visualizations and metrics
  - Model strengths, limitations, and real-world applications

### Key Concepts Learned
- Forward/backpropagation and activation functions
- Information funnel principle (progressive compression)
- Importance of normalization for neural network training
- Model evaluation: R², RMSE, MAE, percentage errors
- Error pattern analysis and identifying model weaknesses

### Next Steps
- Apply ANN methodology to two additional Kaggle datasets
- Explore advanced techniques: dropout, early stopping, feature importance analysis
- Compare different architectures and optimization strategies

## 2025-11-01 -- 2025-11-06 (Week 6)

[Permalink: 2025-11-01 -- 2025-11-06 (Week 6)](progress.md#2025-11-01----2025-11-06-week-6)

### What I Did

**1. Dataset #1: Restaurant Tips Prediction (Regression)**

**Exploratory Data Analysis:**
- Analyzed 244 restaurant transactions with 7 features (total_bill, tip, sex, smoker, day, time, size)
- Generated comprehensive visualizations:
  - Scatter plots: Total bill vs. tip (R² = 0.4566), colored by party size
  - Box plots: Tip amount by party size
  - Tip percentage analysis: Discovered inverse relationship (r = -0.143) between party size and tip percentage
  - Correlation heatmap: Identified primary predictors (total_bill: r = 0.676, party_size: r = 0.489)
  - Violin plots: Analyzed tip distributions across categorical variables (sex, smoker, day, time)

**Key Findings:**
- **Solo diners tip highest percentage** (21.73%) despite smallest bills
- **Party size shows dual effect**: Higher absolute tips (+$0.71 per person) but lower percentages (-0.92% per person)
- **Gender and smoking status** have negligible correlation with tips (r < 0.1)
- **Day-Time multicollinearity** detected (r = 0.874) requiring careful feature handling

**ANN Model Development:**
- Built two-hidden-layer architecture (32 → 16 neurons, ReLU activation)
- Compared basic model vs. feature-engineered model (with interaction terms)
- Generated training visualizations:
  - Model loss and MAE during training
  - Overfitting indicator (validation vs. training loss gap)
  - Predictions vs. actual comparisons with linear regression baseline
  
**Model Performance:**
- **Basic ANN**: R² = 0.18, MAE = $0.83 (underperformed due to small dataset)
- **Linear Regression Baseline**: R² = 0.46, MAE = $0.75 (winner)
- **Conclusion**: Linear relationships dominate; neural networks require more data (1000+ samples) to benefit

---

**2. Dataset #2: Titanic Survival Prediction (Binary Classification)**

**Exploratory Data Analysis:**
- Analyzed 891 passengers with 12 features predicting survival probability
- Data cleaning: Imputed 177 missing Age values (19.9%), dropped high-cardinality features (Cabin, Ticket, Name)
- Generated 7 comprehensive visualizations:
  - Box plots: Age distribution by survival status
  - Stacked bar charts: Survival rates by gender (74% female vs. 19% male), passenger class (63% 1st vs. 24% 3rd), age groups
  - Violin plots: Fare distribution by survival
  - Correlation heatmap for feature importance ranking

**Key Findings:**
- **Gender is THE dominant predictor**: 55.3% survival gap (females 74.2%, males 18.9%)
- **Passenger class hierarchy**: Clear 1st (63%) → 2nd (47%) → 3rd (24%) survival progression
- **Age matters for extremes**: Children (0-12) achieved 60-68% survival; elderly (60+) only 21%
- **Family composition weak predictor**: Solo travelers 30%, small families 45%, large families 17%
- **Pclass-Survival correlation**: r = -0.338 (moderate-strong), Fare-Survival: r = +0.257 (weak-moderate)

**ANN Classification Model:**
- Built two-hidden-layer architecture (32 → 16 neurons, ReLU → Softmax)
- Input: 11 features after one-hot encoding
- Training: 121 epochs with early stopping (patience=30), L2 regularization (alpha=0.01)
- Generated 4 critical visualizations:
  1. **Training history**: Loss and MAE curves showing convergence at ~80 epochs
  2. **Overfitting indicator**: Gap analysis (final gap = 0.10, mild-to-moderate overfitting)
  3. **Confusion matrices**: ANN vs. Logistic Regression comparison
  4. **ROC curves**: AUC comparison showing discrimination ability

**Model Performance:**

| Metric | ANN | Logistic Regression | Winner |
|--------|-----|---------------------|--------|
| **Accuracy** | 80.45% | 80.45% | Tie |
| **Precision** | 0.8269 | 0.7931 | ANN |
| **Recall** | 0.6232 | 0.6667 | Logistic Reg |
| **F1 Score** | 0.7107 | 0.7244 | Logistic Reg |
| **ROC AUC** | 0.8526 | 0.8433 | ANN (marginal) |

**Insights:**
- Both models achieve **excellent discrimination** (AUC ~0.85)
- ANN is more **conservative** (higher precision, fewer false positives)
- Logistic Regression is more **liberal** (higher recall, catches more survivors)
- Primary weakness: Both miss ~35% of actual survivors (false negatives 23-26)

---

### Key Concepts Learned

**Statistical Analysis:**
- Multicollinearity detection and mitigation strategies
- Tip percentage normalization for fair cross-group comparison
- ROC curve interpretation for model discrimination ability
- Confusion matrix analysis for understanding error types

**Neural Network Fundamentals:**
- Overfitting mechanisms: Why validation loss diverges from training loss
- Early stopping rationale: Preventing memorization of noise
- Feature engineering importance: Interaction terms can improve performance
- Small dataset limitations: ANNs require sufficient data to outperform simpler models

**Model Comparison Techniques:**
- When to use ANN vs. Linear/Logistic Regression
- Precision vs. Recall trade-offs in classification
- ROC AUC as a threshold-agnostic performance metric
- Why accuracy alone is misleading for imbalanced or nuanced problems

**Visualization Mastery:**
- Appropriate plot selection: Box plots for distributions, violin plots for dense data, stacked bars for categorical comparisons
- Heatmaps for correlation analysis
- Training curves for diagnosing convergence and overfitting
- ROC curves for comparing model discrimination across thresholds

---

### Documentation Generated

**Tips Dataset:**
- `URS Kaggle Tip dataset analysis and ANN model development.docx` - Complete analysis with business recommendations
- Code files:
  - `src/ANN_tip.py` - Basic ANN implementation
  - `src/tip_visualizations.py` - Comprehensive EDA visualizations
  - Graphs: Scatter plots, box plots, violin plots, heatmap, training history, overfitting analysis

**Titanic Dataset:**
- `URS Kaggle Titanic dataset data and ANN model development.docx` - Full classification analysis
- Code files:
  - `src/titanic_ann_classification.py` - Complete ANN pipeline with all visualizations
  - Graphs: 7 EDA plots + 4 model evaluation plots (training history, overfitting, confusion matrices, ROC curves)

---
