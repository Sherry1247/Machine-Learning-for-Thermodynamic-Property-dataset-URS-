# Machine Learning for Thermodynamic Property Dataset (URS)

---

## 📊 Project Timeline & Progress

| Phase        | Dates         | Focus                                                                                         |
|--------------|--------------|-----------------------------------------------------------------------------------------------|
| **Week 1**   | Sep 18–25    | Repository setup, JANAF data exploration                                                      |
| **Week 2**   | Sep 25–Oct 2 | ML fundamentals, Kaggle dataset analysis                                                      |
| **Week 3**   | Oct 2–9      | Segmented regression, pattern recognition                                                     |
| **Weeks 4–5**| Oct 9–30     | ANN implementation (Medical Insurance), model evaluation                                      |
| **Week 6**   | Nov 1–6      | Two dataset analysis: Tips (regression) + Titanic (classification); k-fold cross-validation   |

See [`progress.md`](progress.md) for detailed timeline and accomplishments.

---

## 🔍 Key Findings

### Week 1–3: Exploratory Data Analysis
- ✅ Identified three distinct clusters in Charges vs. Age relationship
- ✅ Confirmed smoker status as dominant variance factor
- ✅ Found BMI threshold of 30 as significant predictor
- ✅ Validated effectiveness of segmented regression approach

### Week 4–5: Artificial Neural Network Implementation (Medical Insurance)
- ✅ **Model Performance:** 83.49% R² on test data
- ✅ **Accuracy Metrics:**
  - RMSE: $5,063.29
  - MAE: $3,355.92
  - Mean prediction accuracy for typical cases: 76%
- ✅ **Architecture:** Information funnel principle (64→32→16 neurons)
- ✅ **Generalization:** Strong test performance indicates no overfitting

### Week 6: Comparative Dataset Analysis (Tips + Titanic)

**What I Accomplished:**

**1. Restaurant Tips Dataset (Regression):**
- ✅ Analyzed 244 transactions with regression and classification approaches
- ✅ Generated 8+ visualizations: correlation heatmap, scatter plots, box plots, violin plots, training curves
- ✅ Discovered inverse relationship: Solo diners tip 21.7% vs. large parties 14.6%
- ✅ Key finding: Linear regression (R²=0.46) outperformed ANN (R²=0.18) on small dataset
- ✅ Developed business recommendations for optimal staffing and table configuration

**2. Titanic Survival Dataset (Binary Classification):**
- ✅ Analyzed 891 passengers with 12 features, cleaned missing data (19.9% Age values)
- ✅ Generated 11+ visualizations: 7 EDA plots + 4 model evaluation plots (confusion matrices, ROC curves)
- ✅ Identified gender as dominant predictor: 74.2% female vs. 18.9% male survival
- ✅ Compared ANN vs. Logistic Regression: Both achieved 80.45% accuracy, AUC ≈ 0.85
- ✅ Understood precision-recall trade-offs: ANN favors precision, Logistic Regression favors recall

**3. Technical Skills Mastered:**
- ✅ Binary classification with softmax activation and categorical cross-entropy loss
- ✅ Confusion matrix analysis and ROC curve interpretation
- ✅ Overfitting detection and mitigation (early stopping, L2 regularization)
- ✅ Model comparison methodology and metric selection
- ✅ K-fold cross-validation concept for small dataset performance estimation

**4. Key Insights:**
- **When to use ANNs:** Classification tasks with 500+ samples; Regression with 1000+ samples
- **When to use Linear/Logistic Models:** Small datasets (< 500 samples), linear relationships dominate
- **Overfitting indicators:** Validation-training loss gap > 0.15 suggests severe overfitting
- **Feature engineering matters:** Interaction terms and domain knowledge improve model performance

---

## 📈 Model Evaluation

### Artificial Neural Network (Medical Insurance - Weeks 4-5)

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

---

## 💡 Methodology

### 1. Data Preprocessing
- **Categorical Encoding:** One-hot encoding for categorical variables
- **Normalization:** Z-score standardization (StandardScaler)
- **Train-Test Split:** 80/20 with stratification
- **Missing Data Handling:** Median imputation for numerical features

### 2. Modeling Approaches Tested
- Linear regression
- Piecewise linear regression (segmented by thresholds)
- Polynomial regression
- Logistic regression (for classification)
- **Artificial Neural Networks** (primary approach)

### 3. Model Selection Rationale
- **Why ANN?** Captures non-linear relationships and feature interactions
- **Why 64→32→16 architecture?** Information funnel principle enables:
  - Sufficient capacity to learn diverse patterns
  - Progressive compression to prevent overfitting
  - Hierarchical feature representation
- **When to use simpler models?** Small datasets (< 500 samples) with linear relationships

### 4. Model Evaluation Techniques
- **Regression:** R², RMSE, MAE, percentage errors, residual analysis
- **Classification:** Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC
- **Overfitting detection:** Training vs. validation loss gap analysis
- **Cross-validation:** K-fold methodology for robust performance estimation

---

## 📚 Literature & References

**Virtual Sensors (Project Foundation):**
1. Martin, D., Kühl, N., & Satzger, G. (2021). Virtual sensors. *Business & Information Systems Engineering*, 63(3), 315-323.
2. Albertos, P., & Goodwin, G. C. (2002). Virtual sensors for control applications. *Annual Reviews in Control*, 26(1), 101-112.

**Thermodynamic Data Sources:**
3. NIST Chemistry WebBook. (2023). Retrieved from [https://webbook.nist.gov/chemistry/](https://webbook.nist.gov/chemistry/)
   - JANAF Thermochemical Tables: CO₂ (C-095.txt), CO (C-093.txt)

**Kaggle Datasets Used:**
4. Medical Insurance Cost Dataset. Retrieved from [https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)
   - 1,338 samples, 7 features (age, sex, BMI, children, smoker, region, charges)
5. Restaurant Tips Dataset. Retrieved from [https://www.kaggle.com/datasets/jsphyg/tipping](https://www.kaggle.com/datasets/jsphyg/tipping)
   - 244 samples, 7 features (total_bill, tip, sex, smoker, day, time, size)
6. Titanic Survival Dataset. Retrieved from [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
   - 891 samples, 12 features (survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked)

---

## 🎓 Learning Outcomes

Through this project, I have:

**Weeks 1-3: Data Analysis Fundamentals**
- ✅ Mastered data preprocessing and normalization techniques
- ✅ Understood regression modeling and segmentation strategies
- ✅ Developed proficiency in visualization libraries (matplotlib, seaborn)
- ✅ Learned pattern recognition and correlation analysis

**Weeks 4-5: Neural Network Development**
- ✅ Built and trained neural networks from scratch using TensorFlow/Keras
- ✅ Understood forward/backpropagation and activation functions
- ✅ Learned the importance of model evaluation and generalization
- ✅ Developed proficiency in Python ML/DL libraries (pandas, scikit-learn, TensorFlow)
- ✅ Gained experience with Git version control and documentation

**Week 6: Advanced Model Comparison & Evaluation**
- ✅ Mastered binary classification with confusion matrices and ROC curves
- ✅ Understood overfitting mechanisms and mitigation strategies (early stopping, L2 regularization)
- ✅ Learned when ANNs outperform vs. underperform simpler models
- ✅ Applied correlation analysis and feature importance ranking
- ✅ Developed business recommendation skills from data insights
- ✅ K-fold cross-validation methodology for small dataset performance estimation
- ✅ Model trade-off analysis: Precision vs. recall, bias-variance trade-off
- ✅ Comparative analysis: Regression vs. classification problem formulation

---

## 📖 Documentation

**Project Reports:**
- `docs/URS_project_on_Kaggle_medical_insurance_dataset.docx` - Medical Insurance ANN analysis (Weeks 4-5)
- `docs/URS Kaggle Tip dataset analysis and ANN model development.docx` - Tips dataset regression analysis (Week 6)
- `docs/URS Kaggle Titanic dataset data and ANN model development.docx` - Titanic classification analysis (Week 6)

**Progress Tracking:**
- `progress.md` - Detailed weekly progress log with meeting notes and learning outcomes

**Code Files:**
- `src/visualize_co2.py` - JANAF CO₂ thermodynamic data visualization
- `src/visualize_co_cp_vs_T.py` - CO heat capacity analysis
- `src/complete_ann_model.py` - Medical insurance ANN implementation
- `src/ANN_tip.py` - Tips dataset ANN regression
- `src/tip_visualizations.py` - Comprehensive tips dataset EDA
- `src/titanic_ann_classification.py` - Titanic survival classification with model comparison

**All Python files include inline documentation and comments.**

---

## Next Steps

**Immediate Goals (Week 7-8):**
1. Find 3-5 new datasets for classification and regression practice
2. Learn advanced classification methods (Decision Trees, Random Forest, XGBoost)
3. Apply classification to Tips dataset (predict tip categories: Low/Medium/High)
4. Begin work on virtual sensor datasets provided by Dr. Gupta
5. Initialize WARS project: Extreme weather survival probability prediction

**Learning Objectives:**
- Master k-fold cross-validation implementation in Python
- Understand ensemble methods (bagging, boosting)
- Learn feature selection and dimensionality reduction techniques
- Explore hyperparameter tuning with GridSearchCV

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
- **Kaggle:** For medical insurance, tips, and Titanic datasets and community resources
- **UW–Madison:** For supporting undergraduate research initiatives

---

## 📧 Questions or Feedback?

Feel free to open an issue or contact me directly. Contributions and suggestions are welcome!

---

**Last Updated:** November 6, 2025  
**Project Status:** Active (Week 6 Complete: Comparative Analysis of Tips & Titanic Datasets)
