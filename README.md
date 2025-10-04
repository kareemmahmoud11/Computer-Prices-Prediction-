# Computer-Prices-Prediction-
Predict laptop and desktop prices from hardware specifications using advanced regression models

# 💻 Computer Prices Prediction

## Overview
This project predicts the retail price of laptops and desktops based on their hardware specifications.  

The dataset contains ~100,000 rows with around 30 specifications including CPU, GPU, RAM, storage, display, and other attributes.  
The target column is **price**.

This is a tabular regression challenge designed to test your ability to:
- Engineer meaningful features from categorical and numeric data
- Handle high-cardinality categorical variables (brands, models, components)
- Build robust regression models (tree-based, boosting, deep learning, etc.)
- Manage large-scale data efficiently

---

## Dataset Description
- **Device Type:** Laptop / Desktop  
- **CPU:** brand, model, cores, threads, base/boost clock  
- **GPU:** brand, model, VRAM  
- **Memory:** RAM size, storage type/size/drive count  
- **Display:** type (LED, OLED, Mini-LED, IPS, VA, QLED), resolution, refresh rate, size  
- **Other Features:** release year, form factor, weight, PSU/battery, Wi-Fi, Bluetooth, warranty  
- **Target:** price

---

## Data Preprocessing & Feature Engineering

1. **Categorical Features Handling**  
   - Many categorical features had a large number of unique categories (e.g., `cpu_model`, `gpu_model`).  
   - Instead of dropping them, I **grouped categories** to reduce their number (e.g., from 100+ to 5) while keeping predictive power, since they are important features that strongly affect the price.

2. **Outlier Treatment**  
   - The `price` distribution was skewed (right-skewed), so I used the **IQR method** to handle outliers.  

3. **Numerical Features Analysis**  
   - Checked correlation among numerical features to see which ones have similar effects on the target.  
   - This helped identify redundant features to drop while keeping the most informative ones.  

4. **Data Preparation for Linear Models**  
   - Categorical features were converted to numerical using **`get_dummies`**.  
   - Scaling applied to numerical features to ensure proper training since linear regression is sensitive to feature magnitudes.  

---

## Models & Implementation

### 1️⃣ CatBoost Regressor
- **Why:** CatBoost handles categorical features natively, does not require scaling, and is robust to high-cardinality features.  
- **Hyperparameters:**
```python
iterations=1000
learning_rate=0.1
depth=6
l2_leaf_reg=8
subsample=0.8
early_stopping_rounds=50
verbose=100

- **Why:** Handles categorical features natively, does not require scaling, robust to high-cardinality features.  

**Performance:**
| Dataset | MAE | RMSE | R² |
|---------|-----|------|----|
| Train   | 128.43 | 166.82 | 0.90 |
| Test    | 132.27 | 173.68 | 0.90 |

**Observation:** Close train/test scores indicate no overfitting. CatBoost efficiently handled categorical features without preprocessing.

---

### 2️⃣ Linear Regression
- **Why:** Baseline linear model; requires encoding and scaling.  
- **Preprocessing:** Converted categorical features with `get_dummies()`, applied scaling.  

**Performance:**
| Dataset | MSE | R² |
|---------|-----|----|
| Train   | 31112.60 | 0.89 |
| Test    | 30311.40 | 0.89 |

**Observation:** Linear regression provides a solid baseline but cannot capture non-linear relationships well.

---

### 3️⃣ Polynomial Regression (Degree 2)
- **Why:** To capture non-linear relationships between features and price.  

**Performance:**
| Dataset | MSE | R² |
|---------|-----|----|
| Train   | 27814.09 | 0.90 |
| Test    | 29150.18 | 0.89 |

**Observation:** Captures interactions between features effectively, improving over linear regression.

---

### 4️⃣ Ridge Regression
- **Why:** Regularization reduces the effect of less important features and prevents overfitting without removing features.  

**Performance:**
| Dataset | MSE | R² | RMSE |
|---------|-----|----|------|
| Train   | 27814.15 | 0.90 | 166.78 |
| Test    | 29148.04 | 0.89 | 170.73 |

**Observation:** Ridge regression stabilized the linear model, improving generalization while keeping all features.

---

## Key Takeaways
- CatBoost performed exceptionally well without extensive preprocessing, handling categorical features natively.  
- Linear regression provided a strong baseline; Polynomial regression captured non-linear relationships.  
- Ridge regression regularization stabilized the linear model and reduced overfitting.  
- Feature engineering, including grouping high-cardinality categories, handling outliers, and removing redundant numerical features, significantly enhanced model performance.

---

## Conclusion
This project demonstrates the importance of:  
- **Feature Engineering:** Proper handling of categorical and numerical features, outlier treatment.  
- **Model Selection:** Choosing appropriate models for the data (tree-based vs linear-based).  
- **Hyperparameter Tuning:** Optimizing parameters for best performance.  

Both tree-based models (CatBoost) and linear-based models (Ridge, Polynomial Regression) achieved strong performance with careful preprocessing, resulting in accurate price prediction without overfitting.

---

## Short Project Description (GitHub Tagline)
Predict laptop and desktop prices using hardware specifications. Includes advanced preprocessing, handling high-cardinality categorical features, and multiple regression models (CatBoost, Linear, Polynomial, Ridge). Achieves strong predictive performance with minimal overfitting.
