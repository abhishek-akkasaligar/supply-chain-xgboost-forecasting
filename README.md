# Supply Chain Demand Forecasting with XGBoost

**Built by:** Abhishek Akkasaligar | Supply Chain Planning Analyst | Bengaluru, India
**LinkedIn:** [linkedin.com/in/abhishek-akkasaligar-48829076](your-link)

---

## Project Overview

End-to-end machine learning pipeline for demand forecasting in a manufacturing 
import planning environment. Built on 3 years of SAP shipment data covering 
531 Korea-to-Europe materials.

**Key Results:**
- ✅ 38 individual XGBoost models trained on high-runner parts (85.6% of volume)
- ✅ 89% win rate vs live SAP planning system (34 of 38 parts)
- ✅ +2.27 million units/week accuracy improvement on high runners
- ✅ 16 parts flagged with <2 weeks stock (stockout risk)
- ✅ 37 lead time discrepancies caught (system: 120 days, actual: 45-80 days)
- ✅ 390 MTO materials incorrectly classified for statistical forecasting

## Business Context

**Industry:** Electronics manufacturing import planning  
**Scope:** APAC-EMEA supply flows (Korea → Europe)  
**Materials:** 531 finished goods SKUs  
**Data Source:** SAP ERP shipment history (3 years)  
**Planning System:** SAP IBP (live system being compared against)

## Demand Segmentation Methodology

All 531 materials classified into 5 demand patterns:

| Pattern | Count | % of Volume | Forecasting Approach |
|---------|-------|-------------|---------------------|
| True High Runner | 38 | 85.6% | XGBoost statistical forecasting |
| Regular Mover | 8 | 10% | Simple moving average / exponential smoothing |
| Lumpy | 8 | 3% | Croston's method / intermittent demand models |
| Sporadic | 87 | 1% | Manual review + safety stock optimization |
| MTO (Make-to-Order) | 390 | 0.4% | **No statistical forecast** - order-driven |

**Key Insight:** 390 MTO materials were incorrectly set up for statistical 
forecasting in SAP, causing phantom demand and excess inventory.

## Technical Approach

### Feature Engineering
- Lag features (1-week, 2-week, 4-week, 8-week, 12-week lags)
- Rolling statistics (4-week, 8-week, 12-week mean, std, min, max)
- Seasonality indicators (month, quarter, week-of-year)
- Material classification dummies (segmentation labels)
- Lead time features (actual vs planned transit)
- Supplier performance metrics (OTD %, quality score)

### Model Architecture
- **Algorithm:** XGBoost (Gradient Boosting)
- **Why XGBoost over ARIMA/Prophet:**
  - Handles non-linear demand patterns
  - Captures feature interactions automatically
  - Robust to outliers (common in international logistics)
  - Better performance on high-dimensional sparse data
- **Training:** 38 separate models (one per high-runner material)
- **Validation:** Time-based train/test split (last 6 months held out)
- **Evaluation:** MAE, RMSE, MAPE vs live SAP IBP forecasts

### Python Stack
```python
pandas==2.0.3
numpy==1.24.3
xgboost==2.0.0
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.11.1
```

## Results Deep Dive

### Model Performance
```
Material ID    XGBoost MAE    SAP IBP MAE    Improvement    Winner
-----------    -----------    -----------    -----------    ------
MAT-001        12,450         18,320         +32%           XGBoost
MAT-002         8,920         15,100         +41%           XGBoost
MAT-003        22,100         21,800         -1%            SAP IBP
...            ...            ...            ...            ...
Overall        14,200         16,800         +15.5%         XGBoost
```

### Business Impact
1. **Stockout Prevention:** 16 parts flagged with <2 weeks on-hand stock
2. **Lead Time Correction:** 37 parts had SAP master data errors (120 days vs 45-80 actual)
3. **Planning Efficiency:** 390 MTO parts removed from statistical forecasting
4. **Working Capital:** Estimated 8-12% inventory reduction potential on corrected parts

## How to Run

```bash
# Clone repository
git clone https://github.com/abhishek-akkasaligar/supply-chain-xgboost-forecasting.git
cd supply-chain-xgboost-forecasting

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_data_preparation.ipynb
```

## What I Learned

1. **Domain knowledge beats algorithms:** The 89% win rate came from understanding 
   which materials to model (high runners) and which to exclude (MTO), not from 
   tuning hyperparameters.

2. **Data quality > model complexity:** Catching lead time discrepancies in master 
   data had more immediate business impact than any algorithm improvement.

3. **Operationalize or die:** A model that lives in a Jupyter notebook is worthless. 
   The real work was presenting findings to senior management and driving planning 
   corrections.

## Certifications & Background

- **IIT Bombay:** Supply Chain Analytics with AI & ML Applications (2026)
- **CSCMP:** Supply Chain Foundations (Demand Planning, Inventory, Transportation, Warehousing)
- **Current Role:** Material Planning Analyst at TE Connectivity (APAC-EMEA flows)
- **Previous:** Caterpillar Inc (Material Availability Analyst, Material Planner)

## Connect With Me

- LinkedIn: [linkedin.com/in/abhishek-akkasaligar-48829076]
- Email: abhishek.akkasaligar@gmail.com
- Open to: Supply Chain Data Science | Demand Planning | Inventory Optimization roles

---

**Note:** This repository contains sanitized sample data and methodology documentation. 
Actual company data has been removed for confidentiality. The approach and results 
are fully representative of the production implementation.
