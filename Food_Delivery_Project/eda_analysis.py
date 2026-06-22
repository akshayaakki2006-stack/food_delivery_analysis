"""
eda_analysis.py
----------------
Performs Exploratory Data Analysis (EDA) on the Food Delivery dataset.

- If 'food_delivery_data.csv' is not found in the same folder, it will be
  generated automatically by calling dataset_generator.generate_dataset().
- Prints key business metrics to the terminal.
- Saves several charts as PNG files in the 'charts' folder.

Run:
    python eda_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for Windows/headless runs
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Setup paths (works regardless of current working directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "food_delivery_data.csv")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def ensure_dataset_exists():
    """Generate the dataset if the CSV file does not already exist."""
    if not os.path.exists(CSV_PATH):
        print("[INFO] CSV file not found. Generating dataset automatically...")
        # Import locally to avoid circular import issues / allow standalone use
        sys.path.insert(0, SCRIPT_DIR)
        from dataset_generator import generate_dataset
        generate_dataset(output_path=CSV_PATH)
    else:
        print(f"[INFO] Found existing dataset: {CSV_PATH}")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["Order_Date"])
    return df


def print_key_metrics(df: pd.DataFrame):
    delivered_df = df[df["Order_Status"] == "Delivered"]

    total_revenue = delivered_df["Order_Value"].sum()
    total_orders = len(df)
    delivered_orders = len(delivered_df)
    cancelled_orders = total_orders - delivered_orders
    avg_order_value = delivered_df["Order_Value"].mean()
    avg_delivery_time = delivered_df["Delivery_Time_Min"].mean()
    avg_customer_rating = delivered_df["Customer_Rating"].mean()
    cancellation_rate = (cancelled_orders / total_orders) * 100

    print("\n" + "=" * 55)
    print("           FOOD DELIVERY BUSINESS - KEY METRICS")
    print("=" * 55)
    print(f"Total Orders            : {total_orders:,}")
    print(f"Delivered Orders        : {delivered_orders:,}")
    print(f"Cancelled Orders        : {cancelled_orders:,}")
    print(f"Cancellation Rate       : {cancellation_rate:.2f}%")
    print(f"Total Revenue           : ${total_revenue:,.2f}")
    print(f"Average Order Value     : ${avg_order_value:,.2f}")
    print(f"Average Delivery Time   : {avg_delivery_time:.2f} minutes")
    print(f"Average Customer Rating : {avg_customer_rating:.2f} / 5.0")
    print("=" * 55 + "\n")

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "avg_delivery_time": avg_delivery_time,
        "avg_customer_rating": avg_customer_rating,
    }


def print_extra_insights(df: pd.DataFrame):
    delivered_df = df[df["Order_Status"] == "Delivered"]

    print("---- Top 5 Cities by Revenue ----")
    print(delivered_df.groupby("City")["Order_Value"].sum().sort_values(ascending=False).head(5).round(2))

    print("\n---- Top 5 Cuisines by Order Count ----")
    print(delivered_df["Cuisine_Type"].value_counts().head(5))

    print("\n---- Revenue by Payment Method ----")
    print(delivered_df.groupby("Payment_Method")["Order_Value"].sum().sort_values(ascending=False).round(2))

    print("\n---- Average Rating by Delivery Vehicle ----")
    print(delivered_df.groupby("Delivery_Vehicle")["Customer_Rating"].mean().round(2))
    print()


def save_chart(fig, filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[SAVED] {path}")


def generate_charts(df: pd.DataFrame):
    delivered_df = df[df["Order_Status"] == "Delivered"].copy()
    delivered_df["Order_Month"] = delivered_df["Order_Date"].dt.to_period("M").astype(str)

    # 1. Revenue by City
    fig, ax = plt.subplots()
    city_revenue = delivered_df.groupby("City")["Order_Value"].sum().sort_values(ascending=False)
    sns.barplot(x=city_revenue.values, y=city_revenue.index, color="steelblue", ax=ax)
    ax.set_title("Total Revenue by City")
    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("City")
    save_chart(fig, "revenue_by_city.png")

    # 2. Monthly Revenue Trend
    fig, ax = plt.subplots()
    monthly_revenue = delivered_df.groupby("Order_Month")["Order_Value"].sum().sort_index()
    monthly_revenue.plot(kind="line", marker="o", ax=ax, color="darkorange")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue ($)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save_chart(fig, "monthly_revenue_trend.png")

    # 3. Order Value Distribution
    fig, ax = plt.subplots()
    sns.histplot(delivered_df["Order_Value"], bins=40, kde=True, color="steelblue", ax=ax)
    ax.set_title("Distribution of Order Value")
    ax.set_xlabel("Order Value ($)")
    save_chart(fig, "order_value_distribution.png")

    # 4. Delivery Time Distribution
    fig, ax = plt.subplots()
    sns.histplot(delivered_df["Delivery_Time_Min"], bins=40, kde=True, color="seagreen", ax=ax)
    ax.set_title("Distribution of Delivery Time")
    ax.set_xlabel("Delivery Time (Minutes)")
    save_chart(fig, "delivery_time_distribution.png")

    # 5. Customer Rating Distribution
    fig, ax = plt.subplots()
    sns.countplot(x="Customer_Rating", data=delivered_df.round({"Customer_Rating": 0}), color="indianred", ax=ax)
    ax.set_title("Customer Rating Distribution (Rounded)")
    ax.set_xlabel("Rating")
    save_chart(fig, "customer_rating_distribution.png")

    # 6. Cuisine Popularity
    fig, ax = plt.subplots()
    cuisine_counts = delivered_df["Cuisine_Type"].value_counts()
    sns.barplot(x=cuisine_counts.values, y=cuisine_counts.index, color="mediumpurple", ax=ax)
    ax.set_title("Order Count by Cuisine Type")
    ax.set_xlabel("Number of Orders")
    save_chart(fig, "cuisine_popularity.png")

    # 7. Delivery Time vs Distance Scatter
    fig, ax = plt.subplots()
    sns.scatterplot(x="Distance_KM", y="Delivery_Time_Min", data=delivered_df.sample(min(2000, len(delivered_df))),
                     alpha=0.4, ax=ax, color="purple")
    ax.set_title("Delivery Time vs Distance")
    ax.set_xlabel("Distance (KM)")
    ax.set_ylabel("Delivery Time (Min)")
    save_chart(fig, "delivery_time_vs_distance.png")

    # 8. Payment Method Share (Pie Chart)
    fig, ax = plt.subplots()
    payment_counts = delivered_df["Payment_Method"].value_counts()
    ax.pie(payment_counts.values, labels=payment_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Payment Method Share")
    save_chart(fig, "payment_method_share.png")

    # 9. Order Status Breakdown
    fig, ax = plt.subplots()
    status_counts = df["Order_Status"].value_counts()
    sns.barplot(x=status_counts.index, y=status_counts.values, color="seagreen", ax=ax)
    ax.set_title("Order Status Breakdown")
    ax.set_ylabel("Number of Orders")
    save_chart(fig, "order_status_breakdown.png")

    # 10. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    numeric_cols = ["Order_Value", "Distance_KM", "Delivery_Time_Min", "Customer_Rating",
                    "Number_of_Items", "Delivery_Partner_Experience_Months"]
    corr = delivered_df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap of Key Numeric Features")
    save_chart(fig, "correlation_heatmap.png")


def main():
    print("Starting Food Delivery EDA Analysis...\n")
    ensure_dataset_exists()
    df = load_data()
    print(f"[INFO] Dataset loaded with shape: {df.shape}\n")

    print_key_metrics(df)
    print_extra_insights(df)

    print("Generating charts, please wait...")
    generate_charts(df)

    print("\n[DONE] EDA Analysis complete. Charts saved in the 'charts' folder.")


if __name__ == "__main__":
    main()
