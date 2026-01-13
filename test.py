from __future__ import annotations

import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# 1. CREATE CREDIT PORTFOLIO DATA
# -----------------------------
def generate_synthetic_portfolio(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic credit portfolio DataFrame.

    Columns: Income, Loan_Amount, Credit_Score, Tenure, Default
    """
    np.random.seed(seed)
    df = pd.DataFrame(
        {
            "Income": np.random.normal(60000, 15000, n),
            "Loan_Amount": np.random.normal(200000, 50000, n),
            "Credit_Score": np.random.normal(680, 50, n),
            "Tenure": np.random.randint(1, 10, n),
            "Default": np.random.binomial(1, 0.15, n),
        }
    )
    return df


def train_pd_model(data: pd.DataFrame, features=None, target: str = "Default"):
    """Train a logistic regression PD model and add `PD` column to the DataFrame.

    Returns the trained model and modified DataFrame.
    """
    if features is None:
        features = ["Income", "Loan_Amount", "Credit_Score", "Tenure"]
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    data = data.copy()
    data["PD"] = model.predict_proba(X)[:, 1]
    logger.info("Credit Scoring Model Performance:\n\n%s", classification_report(y_test, model.predict(X_test), zero_division=0))
    return model, data


def add_lgd(data: pd.DataFrame) -> pd.DataFrame:
    """Add a simple LGD rule to the DataFrame.

    - Credit_Score > 700 -> 25%
    - Credit_Score > 650 -> 40%
    - otherwise -> 60%
    """
    data = data.copy()
    data["LGD"] = np.where(
        data["Credit_Score"] > 700, 0.25, np.where(data["Credit_Score"] > 650, 0.40, 0.60)
    )
    return data


def add_ead(data: pd.DataFrame, ead_ratio: float = 0.9) -> pd.DataFrame:
    data = data.copy()
    data["EAD"] = data["Loan_Amount"] * ead_ratio
    return data


def compute_ecl(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ECL"] = data["PD"] * data["LGD"] * data["EAD"]
    return data


def stress_test(data: pd.DataFrame, pd_multiplier: float = 1.3) -> Tuple[pd.DataFrame, float]:
    data = data.copy()
    data["PD_Stress"] = data["PD"] * pd_multiplier
    data["ECL_Stress"] = data["PD_Stress"] * data["LGD"] * data["EAD"]
    stress_ecl = data["ECL_Stress"].sum()
    return data, stress_ecl

def build_combined_dashboard(
    data: pd.DataFrame, trend_df: pd.DataFrame, portfolio_ecl: float, stress_ecl: float, avg_pd: float, show: bool = True
) -> go.Figure:
    """Build and optionally display the combined dashboard figure.

    Returns the Plotly Figure so other programs can embed or save it.
    """
    # Use a clean template and cohesive styling
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Average Probability of Default",
            "Total Portfolio ECL",
            "PD Distribution",
            "ECL vs Credit Score",
            "ECL Trend Over Time",
            "Stress Testing Comparison",
        ),
    )

    # Indicator: Avg PD as percentage
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=avg_pd,
            number={"valueformat": ".2%"},
            title={"text": "Avg PD"},
        ),
        row=1,
        col=1,
    )

    # Indicator: Total ECL with currency formatting
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=portfolio_ecl,
            number={"valueformat": ",.0f", "prefix": "$"},
            title={"text": "Total ECL"},
        ),
        row=1,
        col=2,
    )

    # PD distribution - soft color and rounded bins
    fig.add_trace(
        go.Histogram(
            x=data["PD"],
            nbinsx=30,
            name="PD Distribution",
            marker=dict(color="#636efa", line=dict(color="#ffffff", width=0.5)),
            opacity=0.9,
        ),
        row=2,
        col=1,
    )

    # ECL vs Credit Score scatter with perceptual colorscale and subtle markers
    fig.add_trace(
        go.Scatter(
            x=data["Credit_Score"],
            y=data["ECL"],
            mode="markers",
            marker=dict(
                size=np.clip((data["Loan_Amount"] - data["Loan_Amount"].min()) / 20000 + 6, 6, 18),
                color=data["LGD"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="LGD"),
                opacity=0.75,
                line=dict(width=0.3, color="#222222"),
            ),
            hovertemplate="Score: %{x}<br>ECL: $%{y:,.0f}<br>LGD: %{marker.color:.2f}",
            name="ECL vs Credit Score",
        ),
        row=2,
        col=2,
    )

    # Trend line - thicker, branded color
    fig.add_trace(
        go.Scatter(
            x=trend_df["Year"],
            y=trend_df["ECL"],
            mode="lines+markers",
            line=dict(color="#EF553B", width=3),
            marker=dict(size=8),
            name="ECL Trend",
        ),
        row=3,
        col=1,
    )

    # Stress comparison - show percentage change annotation
    fig.add_trace(
        go.Bar(
            x=["Base Scenario", "Stress Scenario"],
            y=[portfolio_ecl, stress_ecl],
            marker_color=["#00cc96", "#ab63fa"],
            name="Stress Impact",
        ),
        row=3,
        col=2,
    )

    # Layout polish
    fig.update_layout(
        template="plotly_white",
        height=960,
        width=1250,
        title_text="📊 Integrated Credit Risk Analytics Dashboard",
        title_font=dict(size=20, family="Arial"),
        margin=dict(t=90, b=40, l=40, r=40),
        paper_bgcolor="#f8f9fb",
        plot_bgcolor="#f8f9fb",
        showlegend=False,
    )

    # Add annotation for stress delta
    delta_pct = (stress_ecl - portfolio_ecl) / max(portfolio_ecl, 1)
    fig.add_annotation(
        x=0.5,
        y=0.96,
        xref="paper",
        yref="paper",
        text=f"Stress impact: {delta_pct:+.1%} vs base",
        showarrow=False,
        font=dict(size=12, color="#444444"),
    )

    if show:
        fig.show()
    return fig


def get_db_engine(username: str = "root", password: str | None = None, host: str = "127.0.0.1", port: str = "3306", database: str = "credit_risk_db"):
    """Return a SQLAlchemy engine. Password resolves from parameter or `MYSQL_PASSWORD` env var.

    Note: calling programs should avoid hard-coding credentials in production.
    """
    if password is None:
        password = os.getenv("MYSQL_PASSWORD", "12345")
    url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
    return create_engine(url)


def save_to_sql(data: pd.DataFrame, engine, table_name: str = "credit_portfolio") -> None:
    df = data.copy()
    df.rename(columns={"Default": "Default_Flag"}, inplace=True)
    cols = [
        "Income",
        "Loan_Amount",
        "Credit_Score",
        "Tenure",
        "Default_Flag",
        "PD",
        "LGD",
        "EAD",
        "ECL",
        "PD_Stress",
        "ECL_Stress",
        "Year",
    ]
    df[cols].to_sql(table_name, con=engine, if_exists="replace", index=False)
    logger.info("Data successfully pushed to SQL database")


def main(show_dashboard: bool = True):
    df = generate_synthetic_portfolio()
    model, df = train_pd_model(df)
    df = add_lgd(df)
    df = add_ead(df)
    df = compute_ecl(df)
    portfolio_ecl = df["ECL"].sum()
    avg_pd = df["PD"].mean()
    df["Year"] = np.random.choice([2022, 2023, 2024], size=len(df))
    trend_df = df.groupby("Year", as_index=False)["ECL"].mean()
    df, stress_ecl = stress_test(df)

    build_combined_dashboard(df, trend_df, portfolio_ecl, stress_ecl, avg_pd, show=show_dashboard)

    engine = get_db_engine()
    save_to_sql(df, engine)


if __name__ == "__main__":
    main(show_dashboard=True)
