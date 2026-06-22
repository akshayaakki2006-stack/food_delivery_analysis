"""
dataset_generator.py
---------------------
Generates a synthetic Food Delivery business dataset (>= 10,000 records)
and saves it as 'food_delivery_data.csv' in the same folder as this script.

Run directly:
    python dataset_generator.py

Or import and call generate_dataset() from another script (eda_analysis.py
does this automatically if the CSV is missing).
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
NUM_RECORDS = 12000

CSV_FILE_NAME = "food_delivery_data.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, CSV_FILE_NAME)


def generate_dataset(num_records: int = NUM_RECORDS, output_path: str = CSV_PATH, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic food delivery dataset and save it to CSV.

    Returns the generated DataFrame.
    """
    random.seed(seed)
    np.random.seed(seed)

    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"
    ]

    restaurant_names = [
        "Pizza Palace", "Burger Barn", "Sushi Spot", "Taco Town", "Curry House",
        "Noodle Nest", "BBQ Bros", "Pasta Place", "Wrap World", "Salad Stop",
        "Fried Chicken Co", "Vegan Vibes", "Steak Station", "Dim Sum Den", "Donut Depot"
    ]

    cuisines = [
        "Italian", "American", "Japanese", "Mexican", "Indian",
        "Chinese", "BBQ", "Mediterranean", "Thai", "Fast Food"
    ]

    payment_methods = ["Credit Card", "Debit Card", "Cash on Delivery", "Digital Wallet", "UPI"]
    order_status_options = ["Delivered", "Cancelled", "Delivered", "Delivered", "Delivered"]  # mostly delivered
    weather_conditions = ["Clear", "Rainy", "Cloudy", "Stormy", "Hot", "Cold"]
    delivery_vehicle = ["Bike", "Scooter", "Car", "Bicycle"]

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    order_ids = np.arange(100001, 100001 + num_records)

    cities_arr = np.random.choice(cities, num_records)
    restaurant_arr = np.random.choice(restaurant_names, num_records)
    cuisine_arr = np.random.choice(cuisines, num_records)
    payment_arr = np.random.choice(payment_methods, num_records, p=[0.30, 0.20, 0.15, 0.25, 0.10])
    status_arr = np.random.choice(order_status_options, num_records)
    weather_arr = np.random.choice(weather_conditions, num_records)
    vehicle_arr = np.random.choice(delivery_vehicle, num_records, p=[0.45, 0.30, 0.15, 0.10])

    # Random order dates
    random_days = np.random.randint(0, date_range_days, num_records)
    random_seconds = np.random.randint(0, 86400, num_records)
    order_dates = [start_date + timedelta(days=int(d), seconds=int(s)) for d, s in zip(random_days, random_seconds)]

    # Order value: base price per item * quantity + delivery fee
    num_items = np.random.randint(1, 6, num_records)
    price_per_item = np.round(np.random.uniform(4.0, 25.0, num_records), 2)
    food_cost = np.round(num_items * price_per_item, 2)

    delivery_fee = np.round(np.random.uniform(1.5, 8.0, num_records), 2)
    discount = np.round(np.random.choice(
        [0, 0, 0, 2, 5, 10],
        num_records,
        p=[0.4, 0.2, 0.1, 0.15, 0.1, 0.05]
    ).astype(float), 2)

    order_value = np.round(food_cost + delivery_fee - discount, 2)
    order_value = np.clip(order_value, 3.0, None)  # ensure positive

    # Delivery distance in km
    distance_km = np.round(np.random.uniform(0.5, 15.0, num_records), 2)

    # Delivery time in minutes - correlated loosely with distance + randomness
    base_time = 15 + distance_km * 2.2
    delivery_time_min = np.round(base_time + np.random.normal(0, 6, num_records), 1)
    delivery_time_min = np.clip(delivery_time_min, 8, 90)

    # Customer rating 1-5, correlated inversely with delivery time
    rating_noise = np.random.normal(0, 0.6, num_records)
    raw_rating = 5.2 - (delivery_time_min - 20) * 0.02 + rating_noise
    customer_rating = np.clip(np.round(raw_rating, 1), 1.0, 5.0)

    # Customer age and gender
    customer_age = np.random.randint(18, 65, num_records)
    customer_gender = np.random.choice(["Male", "Female", "Other"], num_records, p=[0.48, 0.48, 0.04])

    # Delivery partner experience (months)
    partner_experience_months = np.random.randint(1, 60, num_records)

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Order_Date": order_dates,
        "City": cities_arr,
        "Restaurant_Name": restaurant_arr,
        "Cuisine_Type": cuisine_arr,
        "Customer_Age": customer_age,
        "Customer_Gender": customer_gender,
        "Number_of_Items": num_items,
        "Food_Cost": food_cost,
        "Delivery_Fee": delivery_fee,
        "Discount": discount,
        "Order_Value": order_value,
        "Payment_Method": payment_arr,
        "Distance_KM": distance_km,
        "Delivery_Time_Min": delivery_time_min,
        "Delivery_Vehicle": vehicle_arr,
        "Delivery_Partner_Experience_Months": partner_experience_months,
        "Weather_Condition": weather_arr,
        "Order_Status": status_arr,
        "Customer_Rating": customer_rating,
    })

    # For cancelled orders, set rating and delivery time as NaN (more realistic)
    cancelled_mask = df["Order_Status"] == "Cancelled"
    df.loc[cancelled_mask, "Customer_Rating"] = np.nan
    df.loc[cancelled_mask, "Delivery_Time_Min"] = np.nan

    # Sort by date for readability
    df = df.sort_values("Order_Date").reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"[OK] Dataset generated successfully: {output_path}")
    print(f"[OK] Total records created: {len(df):,}")
    return df


if __name__ == "__main__":
    generate_dataset()
