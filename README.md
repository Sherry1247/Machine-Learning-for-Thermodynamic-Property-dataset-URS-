
---

## 📊 Project Timeline & Progress

| Phase | Dates | Focus |
|-------|-------|-------|
| **Week 1** | Sep 18–25 | Repository setup, JANAF data exploration |
| **Week 2** | Sep 25–Oct 2 | ML fundamentals, Kaggle dataset analysis |
| **Week 3** | Oct 2–9 | Segmented regression, pattern recognition |
| **Weeks 4–5** | Oct 9–30 | ANN implementation, model evaluation, documentation |

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

**Last Updated:** October 30, 2025  
**Project Status:** Active (Weeks 4-5 Complete, Future Phases Planned)
