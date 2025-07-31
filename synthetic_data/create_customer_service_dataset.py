#!/usr/bin/env python
"""
Create Customer Service Dataset with Assets
Generates synthetic customer service data and converts it to asset-based format
"""

import os
import json
import sqlite3
import yaml
from pathlib import Path
from factory import CustomerServiceDataGenerator

def create_customer_service_dataset(output_dir: str = "customer_service_dataset"):
    """Create a complete customer service dataset with assets"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    assets_dir = output_path / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    print("Generating synthetic customer service data...")
    
    # Generate the dataset using the factory
    generator = CustomerServiceDataGenerator(seed=42)
    dataset = generator.generate_complete_dataset(
        num_customers=20,   # Reasonable size for testing
        num_tickets=30
    )
    
    print("Converting to asset-based format...")
    
    # 1. Save company policy as a text file
    policy_file = assets_dir / "company_policy.md"
    with open(policy_file, 'w') as f:
        f.write("# TechFlow Solutions Customer Service Policy\n\n")
        f.write(dataset["company_policy"])
    
    print(f"✅ Saved company policy to: {policy_file}")
    
    # 2. Create SQLite database with customers and orders
    db_file = assets_dir / "customer_service.db"
    create_sqlite_database(dataset, db_file)
    
    print(f"✅ Created database: {db_file}")
    
    # 3. Create evaluation format (questions and answers)
    eval_data = generator.create_evaluation_format(dataset)
    eval_file = output_path / "evaluation_data.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"✅ Created evaluation data: {eval_file}")
    
    # 4. Create dataset configuration YAML
    create_dataset_config(output_path)
    
    print(f"✅ Created dataset configuration")
    
    # 5. Create a sample policy query file for testing
    create_sample_files(assets_dir)
    
    print(f"\n🎉 Customer service dataset created successfully!")
    print(f"📁 Dataset directory: {output_path.absolute()}")
    print(f"📊 {len(eval_data)} evaluation examples")
    print(f"👥 {len(dataset['customers'])} customers in database")
    print(f"📦 {len(dataset['orders'])} orders in database")
    
    return str(output_path.absolute())

def create_sqlite_database(dataset: dict, db_file: Path):
    """Create SQLite database from the dataset"""
    
    # Remove existing database
    if db_file.exists():
        db_file.unlink()
    
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Create customers table
    cursor.execute('''
        CREATE TABLE customers (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            name TEXT NOT NULL,
            account_status TEXT NOT NULL,
            join_date DATE,
            total_orders INTEGER,
            total_spent REAL,
            phone TEXT,
            address_street TEXT,
            address_city TEXT,
            address_state TEXT,
            address_zip TEXT
        )
    ''')
    
    # Create orders table
    cursor.execute('''
        CREATE TABLE orders (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            order_date DATE,
            total_amount REAL,
            status TEXT,
            tracking_number TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers (id)
        )
    ''')
    
    # Create order_items table for detailed item information
    cursor.execute('''
        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            item_name TEXT,
            item_price REAL,
            item_sku TEXT,
            quantity INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders (id)
        )
    ''')
    
    # Insert customers
    for customer_id, customer_data in dataset["customers"].items():
        address = customer_data.get("address", {})
        cursor.execute('''
            INSERT INTO customers (
                id, email, name, account_status, join_date, total_orders, total_spent,
                phone, address_street, address_city, address_state, address_zip
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            customer_id,
            customer_data.get("email", ""),
            customer_data.get("name", ""),
            customer_data.get("account_status", "standard"),
            customer_data.get("join_date", ""),
            customer_data.get("total_orders", 0),
            customer_data.get("total_spent", 0.0),
            customer_data.get("phone"),
            address.get("street"),
            address.get("city"),
            address.get("state"),
            address.get("zip")
        ))
    
    # Insert orders and order items
    for order_id, order_data in dataset["orders"].items():
        cursor.execute('''
            INSERT INTO orders (
                id, customer_id, order_date, total_amount, status, tracking_number
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            order_id,
            order_data.get("customer_id", ""),
            order_data.get("order_date", ""),
            order_data.get("total_amount", 0.0),
            order_data.get("status", ""),
            order_data.get("tracking_number", "")
        ))
        
        # Insert order items
        for item in order_data.get("items", []):
            cursor.execute('''
                INSERT INTO order_items (
                    order_id, item_name, item_price, item_sku, quantity
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                order_id,
                item.get("name", ""),
                item.get("price", 0.0),
                item.get("sku", ""),
                item.get("quantity", 1)
            ))
    
    # Create helpful views
    cursor.execute('''
        CREATE VIEW customer_summary AS
        SELECT 
            c.id,
            c.email,
            c.name,
            c.account_status,
            COUNT(o.id) as order_count,
            COALESCE(SUM(o.total_amount), 0) as total_spent_calculated,
            MAX(o.order_date) as last_order_date
        FROM customers c
        LEFT JOIN orders o ON c.id = o.customer_id
        GROUP BY c.id, c.email, c.name, c.account_status
    ''')
    
    conn.commit()
    conn.close()

def create_dataset_config(output_path: Path):
    """Create the dataset configuration YAML file"""
    
    config = {
        "name": "customer_service_tickets",
        "description": "Customer service ticket resolution using company policies and customer data",
        "version": "1.0",
        "assets": [
            {
                "name": "policy_docs",
                "type": "filesystem",
                "path": "./assets/",
                "description": "Company policy documents and procedures"
            },
            {
                "name": "customer_db",
                "type": "sqlite",
                "path": "./assets/customer_service.db",
                "description": "Customer and order database"
            }
        ]
    }
    
    config_file = output_path / "dataset.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def create_sample_files(assets_dir: Path):
    """Create additional sample files for testing"""
    
    # Create a README for the policy docs
    readme_file = assets_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write("""# Customer Service Assets

This directory contains assets for the customer service dataset:

- `company_policy.md`: Complete company policy document
- `customer_service.db`: SQLite database with customer and order data
- `README.md`: This file

## Database Schema

### customers table
- id: Customer ID
- email: Customer email address
- name: Customer full name
- account_status: standard/premium/vip
- join_date: Date customer joined
- total_orders: Number of orders placed
- total_spent: Total amount spent
- phone: Customer phone number
- address_*: Customer address components

### orders table
- id: Order ID
- customer_id: Reference to customers table
- order_date: Date order was placed
- total_amount: Total order value
- status: Order status (delivered/shipped/processing/cancelled)
- tracking_number: Shipping tracking number

### order_items table
- id: Auto-increment ID
- order_id: Reference to orders table
- item_name: Product name
- item_price: Item price
- item_sku: Product SKU
- quantity: Quantity ordered

### customer_summary view
Aggregated customer information with calculated order statistics.
""")

if __name__ == "__main__":
    # Check if we have the required API key for the factory
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("The factory requires an API key to generate synthetic data")
        print("Set the environment variable and try again")
        exit(1)
    
    output_dir = create_customer_service_dataset()
    print(f"\nTo use this dataset:")
    print(f"python test_customer_service.py {output_dir}") 