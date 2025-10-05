## 2025-09-18 (First Meeting)
### What I planned
Reading virtual sensor essays (overview stage) \
Set up Python + GitHub repo \
Download JANAF CO₂ dataset (C-095.txt) \
Try first visualization (Cp vs T) 

## 2025-09-18 -- 2025-09-25 (Week1)
### What I did:
Created repo with README.md and progress.md \
Added data/C-095.txt (CO₂ JANAF table) \
Wrote script src/visualize_co2.py → successfully plotted Cp vs T \
Learned how to commit + push files to GitHub \
Explored multiple plots (Cp, S, ΔfH, ΔfG vs T) \
Added second dataset data/C-093.txt for CO \
Wrote script src/visualize_co_cp_vs_T.py → plotted Cp vs T for CO

### Essay reference:
1. Martin, D., Kühl, N., & Satzger, G. (2021). Virtual sensors. Business & Information Systems Engineering, 63(3), 315-323.
2. Albertos, P., & Goodwin, G. C. (2002). Virtual sensors for control applications. Annual Reviews in Control, 26(1), 101-112.

## 2025-09-25 -- 2025-10-02 (Week2):
### What I did:
1. Learning supervised machine learning (SML): regression \
https://colab.research.google.com/drive/1FfikNXcsL1t77IHjIh0tLhHhvhrnKHir?usp=sharing#scrollTo=jDq0_zBkipp7 
- Review each line of code in colab
- Write a full code to visualize data provided by colab: create heatmap and pairplot 

2. Explore datasets in kaggle: Medical Insurance Cost Dataset \
https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset
- Write a full code to visualize data from kaggle (in csv file): create scatter plot, box-whisker plot, heatmap, and histogram distribution graph. 

### What I learned from meeting:
- New Visualization Type: I expanded my knowledge of data visualization by learning about the violin plot as an effective way to display the distribution of data across different categories.
- Library Differences :  We discussed the practical differences between Python's visualization libraries, Seaborn and Matplotlib. A key insight is that Seaborn is often utilized more frequently by researchers in the humanities field due to its higher-level, aesthetically pleasing interfaces and statistical focus.
- Scatter Plot Pattern: The most significant finding was the pattern observed in the Charges vs. Age scatter plot. The data does not follow a single, simple linear trend; instead, it explicitly shows three distinct patterns or clusters of charges relative to age.

### Next Step:
1. Advanced Modeling Strategy (Charges vs. Age)
    - Segmentation: Set three barriers (thresholds) based on the observed clusters in the Charges vs. Age scatter plot to divide the data into distinct groups.
    - Modeling: For each of the three identified patterns, perform a separate linear regression analysis to model the cost-age relationship within that specific segment.
2. Broader Data Exploration and Feature Analysis
    - Correlation & Visualization: I will explore how correlation exists among all variables in the dataset, looking for deeper patterns beyond the initial numerical heatmap. This involves generating and analyzing more specialized visualization graphs to fully understand the relationships between both numerical and categorical features.
    - Comprehensive Regression: I will run regression analyses for variables beyond the initial segmentation plan (e.g., using all numerical and encoded categorical variables like BMI and Smoker status) to establish baseline predictive power and quantify their individual impact on insurance charges.

## 2025-10-02 -- 2025-10-09 (week 3)
### What I did:
- Model Development and testing
    - Categorical visualization: generated violin plots for charges vs sex, smoker, and region, visually confirming that smoker status as the dominant variance factor.
    - Segmented Regression: Applied Piecewise Linear (and later Quadratic) regression by segmenting the data based on the target variable (charges: 0-17k, 15k-32k, 31k-60k).






