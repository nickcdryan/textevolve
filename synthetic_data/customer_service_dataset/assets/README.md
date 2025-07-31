# Customer Service Assets

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
