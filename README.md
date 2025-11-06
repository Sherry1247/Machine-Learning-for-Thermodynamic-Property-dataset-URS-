
---

## 📊 Project Timeline & Progress

| Phase | Dates | Focus |
|-------|-------|-------|
| **Week 1** | Sep 18–25 | Repository setup, JANAF data exploration |
| **Week 2** | Sep 25–Oct 2 | ML fundamentals, Kaggle dataset analysis |
| **Week 3** | Oct 2–9 | Segmented regression, pattern recognition |
| **Weeks 4–5** | Oct 9–30 | ANN implementation (Medical Insurance), model evaluation |
| **Week 6** | Nov 1–6 | **Two dataset analysis: Tips (regression) + Titanic (classification)** |


See [`progress.md`](progress.md) for detailed timeline and accomplishments.

---

## 🔍 Key Findings

### Week 1–3: Exploratory Data Analysis
- ✅ Identified three distinct clusters in Charges vs. Age relationship
- ✅ Confirmed smoker status as dominant variance factor
- ✅ Found BMI threshold of 30 as significant predictor
- ✅ Validated effectiveness of segmented regression approach

### Week 4–5: Artificial Neural Network Implementation
- ✅ **Model Performance:** 83.49% R² on test data
- ✅ **Accuracy Metrics:** 
  - RMSE: $5,063.29
  - MAE: $3,355.92
  - Mean prediction accuracy for typical cases: 76%
- ✅ **Architecture:** Information funnel principle (64→32→16 neurons)
- ✅ **Generalization:** Strong test performance indicates no overfitting

---

## 📈 Model Evaluation

### Artificial Neural Network (Medical Insurance)

**Architecture:**
- Input Layer: 8 features (age, BMI, children, sex, smoker, region)
- Hidden Layers: 64 → 32 → 16 neurons (ReLU activation)
- Output Layer: 1 neuron (linear activation for regression)
- Total Parameters: 3,201

**Performance Metrics:**
- **Test R² Score:** 0.8349 (explains 83.49% of variance)
- **Root Mean Squared Error (RMSE):** $5,063.29
- **Mean Absolute Error (MAE):** $3,355.92
- **Training Convergence:** ~40 epochs

**Generated Outputs:**
1. `training_history.png` - Loss and MAE curves showing learning dynamics
2. `actual_vs_predicted.png` - Scatter plot validation
3. `error_distribution.png` - Error analysis histograms
4. `ann_architecture.png` - Network structure visualization
5. `insurance_predictions_comparison.csv` - Detailed predictions for all test cases
6. `insurance_ann_model.h5` - Saved model for deployment

## 📊 Week 6: Comparative Dataset Analysis

### Dataset #1: Restaurant Tips Prediction (Regression)

**Objective:** Predict tip amounts and optimize restaurant revenue through data-driven resource allocation

**Dataset:** 244 transactions with 7 features (total_bill, tip, sex, smoker, day, time, party_size)

**Key Discoveries:**
- ✅ **Inverse tip percentage relationship**: Solo diners tip 21.7% vs. large parties 14.6%
- ✅ **Primary predictors identified**: Total bill (r=0.676), Party size (r=0.489)
- ✅ **Business insight**: Saturday/Sunday dinners generate 37% higher tips than weekday lunch
- ✅ **Model finding**: Linear regression (R²=0.46) outperforms ANN (R²=0.18) due to small dataset

**Visualizations Generated:**
- Correlation heatmap with feature importance ranking
- Scatter plots: Total bill vs. tip (with R² annotation), colored by party size
- Box & violin plots: Tip distributions across categorical variables
- Tip percentage analysis by party size
- Training history: Loss and MAE curves
- Overfitting indicator with gap analysis

**Business Recommendations:**
- Prioritize Saturday/Sunday dinner staffing (highest tip volume)
- Configure tables: 60% two-tops, 30% four-tops, 10% six-tops
- Focus upselling on large parties (10% bill increase → 17.5% tip increase)

---

### Dataset #2: Titanic Survival Prediction (Binary Classification)

**Objective:** Predict passenger survival probability using demographic and travel characteristics

**Dataset:** 891 passengers with 12 features (survival target: 342 survived, 549 died)

**Key Discoveries:**
- ✅ **Gender dominates**: 74.2% female survival vs. 18.9% male (55% gap)
- ✅ **Class hierarchy rigid**: 1st class 63% → 2nd class 47% → 3rd class 24%
- ✅ **Age effects**: Children (0-12) 60-68% survival, Elderly (60+) 21% survival
- ✅ **Model performance**: ANN and Logistic Regression tie at 80.45% accuracy, AUC ≈ 0.85

**Data Cleaning:**
- Imputed 177 missing Age values (19.9%) with median
- Dropped high-cardinality features: Cabin (77% missing), Ticket, Name
- One-hot encoded: Sex, Embarked; Created FamilySize feature

**ANN Architecture:**
- Input: 11 features → Hidden: 32→16 neurons (ReLU) → Output: 2 (Softmax)
- Training: 121 epochs, early stopping (patience=30), L2 regularization
- **Overfitting status**: Mild-to-moderate (gap=0.10), controlled by regularization

**Model Comparison:**

| Metric | ANN | Logistic Regression |
|--------|-----|---------------------|
| Accuracy | 80.45% | 80.45% |
| Precision | 82.69% | 79.31% |
| Recall | 62.32% | 66.67% |
| ROC AUC | 0.8526 | 0.8433 |
| **Trade-off** | Conservative (fewer false alarms) | Liberal (catches more survivors) |

**Visualizations Generated:**
- 7 EDA plots: Box plots (age), stacked bars (gender, class, age groups), violin (fare), heatmap
- 4 Model evaluation plots:
  1. Training history (loss & MAE convergence)
  2. Overfitting indicator (validation-training gap analysis)
  3. Confusion matrices (ANN vs. Logistic Regression)
  4. ROC curves comparison (AUC discrimination ability)

**Key Insight:** 
Despite identical accuracy, ANN favors precision (minimize false positives) while Logistic Regression favors recall (find all survivors). Choice depends on application priority.

---

### Comparative Insights: Regression vs. Classification

| Aspect | Tips (Regression) | Titanic (Classification) |
|--------|-------------------|--------------------------|
| **Dataset Size** | 244 samples | 891 samples |
| **Target Variable** | Continuous (tip $) | Binary (survived 0/1) |
| **Best Model** | Linear Regression | ANN ≈ Logistic Reg (tie) |
| **ANN Performance** | Underperformed (R²=0.18) | Excellent (80% acc, AUC=0.85) |
| **Key Lesson** | Small data → simple models win | Classification benefits from ANN capacity |
| **Overfitting Risk** | Severe (gap=0.37) | Moderate (gap=0.10) |
| **Primary Predictor** | Total bill (r=0.68) | Gender (74% vs 19%) |


---

## 💡 Methodology

### 1. Data Preprocessing
- **Categorical Encoding:** One-hot encoding for categorical variables
- **Normalization:** Z-score standardization (StandardScaler)
- **Train-Test Split:** 80/20 with stratification

### 2. Modeling Approaches Tested
- Linear regression
- Piecewise linear regression (segmented by thresholds)
- Polynomial regression
- **Artificial Neural Networks** (final approach)

### 3. Model Selection Rationale
- **Why ANN?** Captures non-linear relationships and feature interactions
- **Why 64→32→16 architecture?** Information funnel principle enables:
  - Sufficient capacity to learn diverse patterns
  - Progressive compression to prevent overfitting
  - Hierarchical feature representation

---

## 📚 Literature & References

1. Martin, D., Kühl, N., & Satzger, G. (2021). Virtual sensors. *Business & Information Systems Engineering*, 63(3), 315-323.
2. Albertos, P., & Goodwin, G. C. (2002). Virtual sensors for control applications. *Annual Reviews in Control*, 26(1), 101-112.
3. NIST Chemistry WebBook. (2023). Retrieved from https://webbook.nist.gov/chemistry/
4. Kaggle Medical Insurance Dataset. Retrieved from https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset

---
## 🎓 Learning Outcomes

Through this project, I have:

- ✅ Mastered data preprocessing and normalization techniques
- ✅ Understood regression modeling and segmentation strategies
- ✅ Built and trained neural networks from scratch
- ✅ Learned the importance of model evaluation and generalization
- ✅ Developed proficiency in Python ML/DL libraries (pandas, scikit-learn, TensorFlow)
- ✅ Gained experience with Git version control and documentation
- ✅ **NEW: Mastered binary classification with confusion matrices and ROC curves**
- ✅ **NEW: Understood overfitting mechanisms and mitigation strategies**
- ✅ **NEW: Learned when ANNs outperform vs. underperform simpler models**
- ✅ **NEW: Applied correlation analysis and feature importance ranking**
- ✅ **NEW: Developed business recommendation skills from data insights**


---

## 📖 Documentation

- **Project Report:** See `docs/URS_project_on_Kaggle_medical_insurance_dataset.docx`
- **Detailed Progress:** See `progress.md`
- **Code Comments:** All Python files include inline documentation

---

## 👤 Author

**Daisiqi** | Machine Learning & Data Science Researcher

*University of Wisconsin–Madison*

---

## 📄 License

This project is for educational and research purposes under the URS (Undergraduate Research Scholars) program.

---

## 🤝 Acknowledgments

- **Mentor & Advisor:** Dr. Gupta for guidance on ML methodology and project direction
- **NIST:** For providing comprehensive thermodynamic databases
- **Kaggle:** For medical insurance dataset and community resources
- **UW–Madison:** For supporting undergraduate research initiatives

---

## 📧 Questions or Feedback?

Feel free to open an issue or contact me directly. Contributions and suggestions are welcome!

---

**Last Updated:** November 6, 2025  
**Project Status:** Active (Week 6 Complete: Comparative Analysis of 2 Kaggle Datasets)

