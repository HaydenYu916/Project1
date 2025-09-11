import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
HA_URL = "https://loratestbed.cse.unsw.edu.au"  # Replace with your Home Assistant URL
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIyM2QzOTU5NWU5Mzg0YTc2OTcxMDM3NjZlZDlmYzU3MSIsImlhdCI6MTc1NjM0NTYxMCwiZXhwIjoyMDcxNzA1NjEwfQ.1og0x_zvPHoGRVdeQ2zXGu-eOSjWSXRKT6-t-5Fc42U"  # Replace with your token

# Headers for API requests
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def load_entities_from_file(filename: str = "acc_entities.json") -> List[Dict]:
    """
    Load entity data from saved JSON file
    """
    try:
        if not os.path.exists(filename):
            print(f"❌ File {filename} not found!")
            return []

        with open(filename, "r") as f:
            entities = json.load(f)

        print(f"✅ Loaded {len(entities)} entities from {filename}")
        return entities

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return []


def get_historical_data(
    entity_id: str, start_time: datetime, end_time: datetime
) -> List[Dict]:
    """
    Fetch historical data for a specific entity between start and end time
    """
    # Format timestamps for Home Assistant API
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()

    url = f"{HA_URL}/api/history/period/{start_iso}"
    params = {"filter_entity_id": entity_id, "end_time": end_iso}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()

        # API returns list of lists, get first list for our entity
        if data and len(data) > 0:
            entity_data = data[0]  # First (and should be only) entity
            print(f"📊 {entity_id}: {len(entity_data)} data points")
            return entity_data
        else:
            print(f"⚠️  {entity_id}: No data found")
            return []

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data for {entity_id}: {e}")
        return []


def fetch_all_historical_data(
    entities: List[Dict], start_time: datetime, end_time: datetime
) -> Dict[str, List[Dict]]:
    """
    Fetch historical data for all entities
    """
    all_data = {}

    print(f"\n🕐 Fetching data from {start_time} to {end_time}")
    print("=" * 60)

    for i, entity in enumerate(entities, 1):
        entity_id = entity["entity_id"]
        friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)

        print(f"[{i}/{len(entities)}] Fetching: {friendly_name}")

        historical_data = get_historical_data(entity_id, start_time, end_time)
        all_data[entity_id] = historical_data

    return all_data


def create_summary_statistics(all_data: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create summary statistics from historical data
    """
    summary_data = []

    for entity_id, data_points in all_data.items():
        if not data_points:
            continue

        # Extract numeric values (skip 'unavailable', 'unknown', etc.)
        values = []
        for point in data_points:
            try:
                value = float(point["state"])
                values.append(value)
            except (ValueError, TypeError):
                continue

        if not values:
            continue

        # Calculate delta consumption for accumulated data
        if len(values) >= 2:
            # For accumulated data, consumption = final_value - initial_value
            total_consumption = values[-1] - values[0]
        else:
            total_consumption = 0

        # If total_consumption is negative or zero, use sum of positive deltas
        if total_consumption <= 0:
            deltas = []
            for i in range(1, len(values)):
                delta = values[i] - values[i - 1]
                if delta > 0:  # Only count positive changes (actual consumption)
                    deltas.append(delta)
            total_consumption = sum(deltas) if deltas else 0

        # Calculate statistics
        stats = {
            "entity_id": entity_id,
            "friendly_name": data_points[0]
            .get("attributes", {})
            .get("friendly_name", entity_id),
            "unit": data_points[0].get("attributes", {}).get("unit_of_measurement", ""),
            "data_points": len(data_points),
            "valid_values": len(values),
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": sum(values) / len(values),
            "total_consumption": total_consumption,  # Add total consumption
            "first_value": values[0] if values else None,
            "last_value": values[-1] if values else None,
            "total_change": values[-1] - values[0] if len(values) >= 2 else 0,
        }
        summary_data.append(stats)

    return pd.DataFrame(summary_data)


def save_historical_data(
    all_data: Dict[str, List[Dict]], filename: str = "historical_data.json"
):
    """
    Save historical data to JSON file
    """
    try:
        with open(filename, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"💾 Historical data saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving historical data: {e}")


def create_pie_chart(summary_df: pd.DataFrame):
    """
    Create a single pie chart showing energy usage by device categories
    """
    if summary_df.empty:
        print("⚠️  No data available for pie chart visualization")
        return

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Filter out zero or negative values for meaningful ratios
    valid_data = summary_df[summary_df["total_consumption"] > 0].copy()

    if valid_data.empty:
        print("⚠️  No positive values found for pie chart visualization")
        return

    # Categorize devices by type
    def categorize_device(name):
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ["light", "lamp", "bulb", "led"]):
            return "Lights"
        elif any(keyword in name_lower for keyword in ["fan", "blower", "exhaust"]):
            return "Fans"
        elif any(keyword in name_lower for keyword in ["pump", "water"]):
            return "Pumps"
        elif any(keyword in name_lower for keyword in ["heater", "heating"]):
            return "Heating"
        elif any(keyword in name_lower for keyword in ["sensor", "monitor"]):
            return "Sensors"
        elif any(keyword in name_lower for keyword in ["controller", "control"]):
            return "Controllers"
        elif any(keyword in name_lower for keyword in ["fog", "mist"]):
            return "Fog"
        elif any(keyword in name_lower for keyword in ["irrigation", "sprinkler"]):
            return "Irrigation"
        elif any(
            keyword in name_lower
            for keyword in ["ac", "air conditioner", "aircon", "hvac"]
        ):
            return "Air Conditioner"
        else:
            return name  # Don't combine, show individual device name

    # Group devices by category, but keep individual names for "Other"
    valid_data["category"] = valid_data["friendly_name"].apply(categorize_device)

    # For known categories, group them; for individual devices, keep them separate
    known_categories = [
        "Lights",
        "Fans",
        "Pumps",
        "Heating",
        "Sensors",
        "Controllers",
        "Fog",
        "Irrigation",
        "Air Conditioner",
    ]

    category_data = []

    # Add grouped categories
    for category in known_categories:
        category_devices = valid_data[valid_data["category"] == category]
        if not category_devices.empty:
            category_sum = {
                "category": category,
                "avg_value": category_devices["avg_value"].sum(),
                "total_consumption": category_devices["total_consumption"].sum(),
                "max_value": category_devices["max_value"].sum(),
                "total_change": category_devices["total_change"].sum(),
            }
            category_data.append(category_sum)

    # Add individual devices (those not in known categories)
    individual_devices = valid_data[~valid_data["category"].isin(known_categories)]
    for _, device in individual_devices.iterrows():
        device_data = {
            "category": device["friendly_name"],
            "avg_value": device["avg_value"],
            "total_consumption": device["total_consumption"],
            "max_value": device["max_value"],
            "total_change": device["total_change"],
        }
        category_data.append(device_data)

    category_data = pd.DataFrame(category_data)

    # Create single pie chart
    plt.figure(figsize=(8, 5))

    values = category_data["total_consumption"]
    labels = category_data["category"]

    # Create pie chart with better formatting
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.05] * len(values),
    )

    plt.title(
        "Total Energy Consumption by Device Category",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Improve text formatting
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    plt.axis("equal")

    # Save the plot
    plt.savefig("energy_pie_chart.png", dpi=300, bbox_inches="tight")
    print("📊 Pie chart saved to energy_pie_chart.png")

    # Display ratios and statistics
    print("\n📈 Total Energy Consumption by Category:")
    print("=" * 50)

    total_consumption = category_data["total_consumption"].sum()

    for _, row in category_data.iterrows():
        consumption_ratio = (row["total_consumption"] / total_consumption) * 100

        print(f"\n🔌 {row['category']}")
        print(
            f"   Consumption Ratio: {consumption_ratio:.1f}% ({row['total_consumption']:.2f} units)"
        )
        print(f"   Average Usage: {row['avg_value']:.2f} units")
        print(f"   Peak Usage: {row['max_value']:.2f} units")

    # Show individual devices within categories
    print("\n📋 Individual Devices by Category:")
    print("=" * 50)

    for category in category_data["category"]:
        devices_in_category = valid_data[valid_data["category"] == category]
        if not devices_in_category.empty:
            print(f"\n🏷️  {category}:")
            for _, device in devices_in_category.iterrows():
                print(
                    f"   • {device['friendly_name']}: {device['total_consumption']:.2f} {device['unit']} total"
                )

    plt.show()


def main():
    print("📈 Home Assistant Historical Data Analyzer")
    print("=" * 50)

    # Load entities from saved file
    entities = load_entities_from_file("acc_entities.json")

    if not entities:
        print("No entities loaded. Please run the entity fetcher script first.")
        return

    # Display loaded entities
    print("\n📋 Loaded entities:")
    for entity in entities:
        friendly_name = entity.get("attributes", {}).get("friendly_name", "N/A")
        current_state = entity.get("state", "N/A")
        unit = entity.get("attributes", {}).get("unit_of_measurement", "")
        print(f"  • {friendly_name}: {current_state} {unit}")

    # Get time period from user
    print("\n🕐 Time Period Selection:")
    print("1. Last 24 hours")
    print("2. Last 7 days")
    print("3. Last 30 days")
    print("4. Custom period")

    choice = input("Select option (1-4): ").strip()

    now = datetime.now()

    if choice == "1":
        start_time = now - timedelta(hours=24)
        end_time = now
    elif choice == "2":
        start_time = now - timedelta(days=7)
        end_time = now
    elif choice == "3":
        start_time = now - timedelta(days=30)
        end_time = now
    elif choice == "4":
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        try:
            start_time = datetime.strptime(start_date, "%Y-%m-%d")
            end_time = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            print("❌ Invalid date format. Using last 24 hours instead.")
            start_time = now - timedelta(hours=24)
            end_time = now
    else:
        print("❌ Invalid choice. Using last 24 hours instead.")
        start_time = now - timedelta(hours=24)
        end_time = now

    # Fetch historical data
    all_data = fetch_all_historical_data(entities, start_time, end_time)

    # Create and display summary statistics
    print("\n📊 Summary Statistics:")
    print("=" * 80)

    summary_df = create_summary_statistics(all_data)

    if not summary_df.empty:
        # Display formatted summary
        for _, row in summary_df.iterrows():
            print(f"\n🔌 {row['friendly_name']}")
            print(f"   Entity ID: {row['entity_id']}")
            print(
                f"   Data Points: {row['data_points']} | Valid Values: {row['valid_values']}"
            )
            print(
                f"   Range: {row['min_value']:.2f} - {row['max_value']:.2f} {row['unit']}"
            )
            print(f"   Average: {row['avg_value']:.2f} {row['unit']}")
            print(
                f"   Change: {row['total_change']:.2f} {row['unit']} (from {row['first_value']:.2f} to {row['last_value']:.2f})"
            )

        # Save summary to CSV
        summary_df.to_csv("energy_summary.csv", index=False)
        print("\n💾 Summary statistics saved to energy_summary.csv")

        # Generate pie chart
        create_pie_chart(summary_df)
    else:
        print("⚠️  No valid data found for analysis")

    # Save raw historical data
    save_choice = (
        input("\n💾 Save raw historical data to file? (y/n): ").lower().strip()
    )
    if save_choice == "y":
        save_historical_data(all_data)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
