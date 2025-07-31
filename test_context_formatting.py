#!/usr/bin/env python
"""
Test script to see the formatted question output
"""

from asset_dataset_loader import create_customer_service_loader

# Load the dataset
loader = create_customer_service_loader("synthetic_data/customer_service_dataset/evaluation_data.json")

# Get the first example
example = loader.examples[0]

# Get the formatted input
formatted_input = loader.get_example_input(example)

print("=== FORMATTED QUESTION ===")
print(formatted_input)
print("\n=== RAW QUESTION ===")
print(example["question"]) 