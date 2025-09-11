import requests
import json
from typing import List, Dict

# Configuration
HA_URL = "https://loratestbed.cse.unsw.edu.au"  # Replace with your Home Assistant URL
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIyM2QzOTU5NWU5Mzg0YTc2OTcxMDM3NjZlZDlmYzU3MSIsImlhdCI6MTc1NjM0NTYxMCwiZXhwIjoyMDcxNzA1NjEwfQ.1og0x_zvPHoGRVdeQ2zXGu-eOSjWSXRKT6-t-5Fc42U"  # Replace with your token

# Headers for API requests
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def get_all_entities() -> List[Dict]:
    """
    Fetch all entity states from Home Assistant
    Returns list of entity dictionaries with state info
    """
    url = f"{HA_URL}/api/states"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        entities = response.json()
        print(f"✅ Successfully fetched {len(entities)} entities")
        return entities

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching entities: {e}")
        return []


def filter_entities_by_keyword(
    entities: List[Dict], keyword: str = "acc"
) -> List[Dict]:
    """
    Filter entities that contain the specified keyword in their entity_id
    """
    filtered = []

    for entity in entities:
        entity_id = entity.get("entity_id", "")
        if keyword.lower() in entity_id.lower() and "cost" not in entity_id.lower():
            filtered.append(entity)

    print(f"🔍 Found {len(filtered)} entities containing '{keyword}'")
    return filtered


def display_filtered_entities(entities: List[Dict]):
    """
    Display filtered entities in a readable format
    """
    if not entities:
        print("No entities found!")
        return

    print("\n" + "=" * 80)
    print("FILTERED ENTITIES CONTAINING 'acc':")
    print("=" * 80)

    for entity in entities:
        entity_id = entity.get("entity_id")
        friendly_name = entity.get("attributes", {}).get("friendly_name", "N/A")
        state = entity.get("state", "N/A")
        unit = entity.get("attributes", {}).get("unit_of_measurement", "")

        print(f"📊 Entity ID: {entity_id}")
        print(f"   Name: {friendly_name}")
        print(f"   Current State: {state} {unit}")
        print(f"   Domain: {entity_id.split('.')[0]}")
        print("-" * 40)


def save_to_file(entities: List[Dict], filename: str = "acc_entities.json"):
    """
    Save filtered entities to a JSON file
    """
    try:
        with open(filename, "w") as f:
            json.dump(entities, f, indent=2)
        print(f"💾 Saved {len(entities)} entities to {filename}")
    except Exception as e:
        print(f"❌ Error saving to file: {e}")


def main():
    print("🏠 Home Assistant Entity Fetcher")
    print("Fetching all entities and filtering for 'acc'...")

    # Step 1: Get all entities
    all_entities = get_all_entities()

    if not all_entities:
        return

    # Step 2: Filter entities containing "acc"
    acc_entities = filter_entities_by_keyword(all_entities, "acc")

    # Step 3: Display results
    display_filtered_entities(acc_entities)

    # Step 4: Save to file (optional)
    if acc_entities:
        save_choice = input("\n💾 Save results to file? (y/n): ").lower().strip()
        if save_choice == "y":
            save_to_file(acc_entities)

    # Step 5: Show entity IDs only for easy copying
    if acc_entities:
        print("\n📋 Entity IDs for easy copying:")
        print("-" * 40)
        for entity in acc_entities:
            print(entity.get("entity_id"))


if __name__ == "__main__":
    main()
