CREATE DATABASE credit_risk_db;
USE credit_risk_db;

CREATE TABLE credit_portfolio (
    Income FLOAT,
    Loan_Amount FLOAT,
    Credit_Score FLOAT,
    Tenure INT,
    Default_Flag INT,
    PD FLOAT,
    LGD FLOAT,
    EAD FLOAT,
    ECL FLOAT,
    PD_Stress FLOAT,
    ECL_Stress FLOAT,
    Year INT
);
