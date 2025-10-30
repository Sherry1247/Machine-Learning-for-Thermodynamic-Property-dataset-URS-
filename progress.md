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
