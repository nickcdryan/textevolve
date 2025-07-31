#!/usr/bin/env python
"""
Simple test to verify MCP tool usage works
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path to import our modules
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def execute_tool(asset_name: str, tool_name: str, **kwargs) -> dict:
    """Execute a tool on a specific asset - ALWAYS use real data"""
    try:
        import sqlite3
        from pathlib import Path
        
        # For customer service data, implement direct asset access
        if asset_name == "customer_db":
            return execute_database_tool(tool_name, **kwargs)
        elif asset_name == "policy_docs":
            return execute_filesystem_tool(tool_name, **kwargs)
        
        return {"success": False, "error": f"Asset '{asset_name}' not found"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_database_tool(tool_name: str, **kwargs) -> dict:
    """Execute database tools on the customer service database"""
    try:
        import sqlite3
        from pathlib import Path
        
        # Use the actual database path
        db_path = Path("synthetic_data/customer_service_dataset/assets/customer_service.db")
        
        if not db_path.exists():
            return {"success": False, "error": f"Database not found at {db_path}"}
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        if tool_name == "sql_query":
            query = kwargs.get("query", "")
            params = kwargs.get("params", [])
            cursor.execute(query, params)
            
            if query.strip().upper().startswith("SELECT"):
                results = [dict(zip([col[0] for col in cursor.description], row)) 
                          for row in cursor.fetchall()]
                return {"success": True, "results": results, "row_count": len(results)}
            else:
                conn.commit()
                return {"success": True, "rows_affected": cursor.rowcount}
                
        elif tool_name == "describe_tables":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            return {"success": True, "tables": tables}
        
        conn.close()
        return {"success": False, "error": f"Unknown database tool: {tool_name}"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_filesystem_tool(tool_name: str, **kwargs) -> dict:
    """Execute filesystem tools on policy documents"""
    try:
        from pathlib import Path
        
        # Use the actual policy docs path
        docs_path = Path("synthetic_data/customer_service_dataset/assets")
        
        if not docs_path.exists():
            return {"success": False, "error": f"Documents directory not found at {docs_path}"}
        
        if tool_name == "read_file":
            file_path = docs_path / kwargs.get("path", "")
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {"success": True, "content": content[:500] + "...", "path": str(file_path)}
            else:
                return {"success": False, "error": f"File not found: {file_path}"}
        
        return {"success": False, "error": f"Unknown filesystem tool: {tool_name}"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_tools():
    """Test database and file tools"""
    
    print("🔧 Testing Database Tools")
    print("=" * 40)
    
    # Test 1: Describe tables
    result = execute_tool("customer_db", "describe_tables")
    print(f"Tables: {result}")
    
    # Test 2: Query for a customer
    result = execute_tool("customer_db", "sql_query", 
                         query="SELECT * FROM customers WHERE email = ?", 
                         params=["jane.doe@email.com"])
    print(f"Customer lookup: {result}")
    
    print("\n📁 Testing Filesystem Tools")
    print("=" * 40)
    
    # Test 3: Read policy file
    result = execute_tool("policy_docs", "read_file", 
                         path="company_policy.md")
    print(f"Policy file: {result}")

def solve(question):
    """Demonstrate proper tool usage for customer service"""
    
    print("🎫 Processing customer service ticket using TOOLS")
    print("=" * 50)
    
    # Extract customer email from ticket
    if '"email"' in question:
        # Simple extraction - in practice would use JSON parsing
        email_start = question.find('"email": "') + 10
        email_end = question.find('"', email_start)
        customer_email = question[email_start:email_end]
        print(f"📧 Extracted customer email: {customer_email}")
        
        # STEP 1: Look up customer in database using TOOLS
        print("\n🔍 Step 1: Looking up customer in database...")
        customer_result = execute_tool("customer_db", "sql_query", 
                                     query="SELECT * FROM customers WHERE email = ?", 
                                     params=[customer_email])
        
        if customer_result["success"] and customer_result["results"]:
            customer = customer_result["results"][0]
            print(f"✅ Found customer: {customer['name']} (ID: {customer['customer_id']})")
            print(f"   Status: {customer['account_status']}")
            print(f"   Total spent: ${customer['total_spent']}")
        else:
            print("❌ Customer not found in database")
        
        # STEP 2: Read company policy using TOOLS
        print("\n📋 Step 2: Reading company policies...")
        policy_result = execute_tool("policy_docs", "read_file", 
                                   path="company_policy.md")
        
        if policy_result["success"]:
            print(f"✅ Loaded policy document ({len(policy_result['content'])} chars)")
            print(f"   Content preview: {policy_result['content'][:200]}...")
        else:
            print("❌ Could not read policy document")
        
        # STEP 3: Generate resolution based on REAL data
        resolution = {
            "customer_lookup": {
                "status": "found" if customer_result["success"] and customer_result["results"] else "not_found",
                "customer_id": customer['customer_id'] if customer_result["success"] and customer_result["results"] else None,
                "account_status": customer['account_status'] if customer_result["success"] and customer_result["results"] else None
            },
            "policy_access": policy_result["success"],
            "tools_used": ["customer_db.sql_query", "policy_docs.read_file"],
            "resolution": "Generated using ACTUAL database and file tools"
        }
        
        return json.dumps(resolution, indent=2)
    
    return "Error: Could not extract customer email from ticket"

if __name__ == "__main__":
    
    print("🚀 Tool Usage Test")
    print("=" * 50)
    
    # Test tools directly
    test_tools()
    
    print("\n" + "=" * 50)
    print("🎫 Testing with Sample Ticket")
    print("=" * 50)
    
    # Test with a sample ticket
    sample_ticket = '''
    CUSTOMER SUPPORT TICKET:
    {
      "ticket_id": "TEST_001",
      "customer_info": {
        "email": "jane.doe@email.com",
        "name": "Jane Doe"
      },
      "subject": "Account login issue",
      "message": "Can't log into my account"
    }
    '''
    
    result = solve(sample_ticket)
    print("Final Result:")
    print(result) 