# **End-to-End Credit Risk Intelligence & Predictive Analytics Pipeline**
Leveraging XGBoost & SHAP for Interpretable Financial Forecasting

📌 Executive Summary Financial institutions lose billions annually to loan defaults. This project implements a robust machine learning pipeline to predict default probability at the point of application. By combining XGBoost's predictive power with SHAP's interpretability, this system provides both high accuracy and the "Reason Codes" required for regulatory compliance in banking.

🧠 The Problem Statement The objective is to minimize Credit Risk by identifying high-risk applicants while reducing False Negatives (missed defaults), which are the most costly errors for a lender.

⚙️ Engineering Pipeline The project follows a modular Machine Learning Lifecycle (MLLC):

Data Synthesis & Strategy Since real-world financial data is often proprietary, I engineered a synthetic dataset of 40,000 records reflecting realistic banking distributions: Target: default (1 = Default, 0 = Non-Default). Logic: Ground truth is influenced by non-linear interactions between Debt-to-Income (DTI) and Credit Scores.

Feature Engineering Beyond raw data, I engineered features to capture "Ability to Pay": Loan-to-Income (LTI) Ratio: A critical metric for assessing debt burden. Numerical Stability: Applied outlier clipping to Income features to ensure gradient descent stability in XGBoost.

Modeling & Interpretability Algorithm: XGBoost (Extreme Gradient Boosting) was chosen for its ability to handle tabular data and capture non-linear relationships. Explainability (XAI): Integrated SHAP (SHapley Additive exPlanations) to break down individual predictions—turning a "black box" model into a "glass box" model.

📊 Performance & Insights _Key Risk Drivers (SHAP Analysis)

Debt-to-Income (DTI): The most significant predictor; values > 0.4 show exponential risk increase.

Credit Score: Strong inverse correlation with default probability.

LTI Ratio: Engineered feature proved to be a top 3 predictor
**Prepared by:** Farzan Ahmad

**Objective:** To demonstrate a full-cycle data analytics solution that predicts loan defaults and quantifies financial risk using Machine Learning, SQL, and Power BI.

---

### **Executive Summary: The "Money Safety Report"**

*This report is designed to explain the Credit Risk Management System in a way that is simple, visual, and easy to follow. I have translated the complex code and database structures into a "Money Safety" framework.*

#### **🚀 Project Goal**

Predicting which bank loans might not be paid back so we can keep money safe.

#### **1. What is this project?**

Imagine you lend  to 1,000 different friends. Some will pay you back quickly, but others might forget or run out of money. This project is a **smart computer system** that looks at a person's history (how much they earn and their "credit score") to predict if they will be a "Good Borrower" or a "Risky Borrower."

#### **2. How the "Brain" Works (The Logic)**

We used three main "ingredients" to calculate how much money might be at risk. We call this **Expected Credit Loss (ECL)**.

* **PD (Probability of Default):** The  chance that a friend won't pay you back.
* **LGD (Loss Given Default):** If they don't pay you back, how much of the money is gone forever? (Usually, we can get some back by selling their collateral, like a bike or a car).
* **EAD (Exposure at Default):** Exactly how much money did they still owe us at the moment they stopped paying?

**The Secret Formula:**


---

### **3. Technical Architecture (End-to-End Workflow)**

To build this solution, I implemented a 4-stage technical pipeline:

1. **Data Engineering (SQL):** * Designed and implemented a relational database schema in **MySQL**.
* Created the `credit_portfolio` table to handle high-dimensional financial data including income, debt ratios, and risk scores.
* Integrated Python with SQL via **SQLAlchemy** for automated data ingestion.


2. **Predictive Modeling (Python):** * Developed a **Logistic Regression** model using `Scikit-Learn` to estimate the **Probability of Default (PD)**.
* Engineered features from synthetic customer data to identify patterns in default behavior.
* Conducted **Stress Testing** by simulating economic downturns to see how risk levels (ECL) would rise under pressure.


3. **Data Visualization (Power BI):** * Built an interactive executive dashboard (`CREDIT.pbix`).
* Visualized key metrics such as **Total ECL vs. Stressed ECL**, **Risk by Income Bracket**, and **Credit Score Distribution**.
* Enabled "drill-down" capabilities to allow managers to see specific risky loan segments.


4. **Automated Reporting:** * Generated statistical plots (`pycredit.png`) using `Matplotlib`/`Plotly` to validate model performance and feature correlation.

---

### **4. Key Business Insights**

* **Proactive Risk Mitigation:** By identifying high-PD customers early, the bank can adjust interest rates or limit exposure, potentially saving millions in bad debt.
* **Capital Adequacy:** The system calculates the exact reserve capital required (ECL), ensuring the bank meets regulatory standards while remaining profitable.
* **Trend Analysis:** The dashboard reveals that customers with a credit score below 600 represent  of the total risk, allowing for targeted policy changes.

---

### **5. Technical Skills Demonstrated**

* **Programming:** Python (Pandas, NumPy, Scikit-Learn)
* **Databases:** SQL (Schema Design, ETL, Data Persistence)
* **BI & Analytics:** Power BI (DAX, Interactive Visualization)
* **Finance:** Credit Risk Modeling (IFRS 9 / CECL Standards)

---
<img width="1302" height="727" alt="powercredit" src="https://github.com/user-attachments/assets/3846f18a-4b58-4732-9f14-ef804a5a01e3" />
<img width="943" height="639" alt="pycredit" src="https://github.com/user-attachments/assets/1d6e07f2-fdd9-4adb-b8bd-e6ddf2d424cc" />

