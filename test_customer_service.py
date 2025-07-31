#!/usr/bin/env python
"""
Test Customer Service Asset-Based System
Demonstrates the asset-based dataset system with customer service tickets
"""

import sys
import os
from pathlib import Path
from asset_dataset_loader import create_customer_service_loader
from enhanced_agent_system import EnhancedAgentSystem

def test_customer_service_system(dataset_dir: str):
    """Test the customer service system with assets"""
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return
    
    evaluation_file = dataset_path / "evaluation_data.json"
    if not evaluation_file.exists():
        print(f"Error: Evaluation data not found: {evaluation_file}")
        return
    
    print("🚀 Testing Customer Service Asset-Based System")
    print("=" * 50)
    
    # Create the asset-enabled dataset loader
    print("\n1. Loading dataset with assets...")
    try:
        loader = create_customer_service_loader(
            dataset_path=str(evaluation_file),
            shuffle=False,  # Keep deterministic for testing
            random_seed=42
        )
        print(f"   ✅ Dataset loaded successfully")
        
        # Show asset information
        asset_info = loader.get_asset_info()
        print(f"   📊 Dataset: {asset_info['dataset_info']['name']}")
        print(f"   🔧 Available tools: {list(asset_info['available_tools'].keys())}")
        
    except Exception as e:
        print(f"   ❌ Failed to load dataset: {e}")
        return
    
    # Test tool execution
    print("\n2. Testing tool execution...")
    test_tools(loader)
    
    # Create enhanced agent system
    print("\n3. Creating enhanced agent system...")
    try:
        agent = EnhancedAgentSystem(
            dataset_loader=loader,
            use_sandbox=False  # Use local execution for testing
        )
        print(f"   ✅ Enhanced agent system created")
        
    except Exception as e:
        print(f"   ❌ Failed to create agent system: {e}")
        return
    
    # Show system prompt context
    print("\n4. System prompt context:")
    print("-" * 30)
    context = agent.generate_system_prompt_context()
    print(context[:1000] + "..." if len(context) > 1000 else context)
    
    # Test script generation and execution
    print("\n5. Testing script generation...")
    test_script_generation(agent, loader)
    
    # Clean up
    print("\n6. Cleaning up...")
    agent.cleanup()
    print("   ✅ Cleanup completed")

def test_tools(loader):
    """Test tool execution capabilities"""
    
    print("   Testing database tools...")
    
    # Test describe_tables
    result = loader.execute_tool("customer_db", "describe_tables")
    if result["success"]:
        tables = result["tables"]
        print(f"   ✅ Found {len(tables)} tables: {', '.join(tables.keys())}")
    else:
        print(f"   ❌ describe_tables failed: {result['error']}")
    
    # Test SQL query
    result = loader.execute_tool("customer_db", "sql_query", 
                                query="SELECT COUNT(*) as customer_count FROM customers")
    if result["success"]:
        count = result["results"][0]["customer_count"]
        print(f"   ✅ Found {count} customers in database")
    else:
        print(f"   ❌ SQL query failed: {result['error']}")
    
    print("   Testing filesystem tools...")
    
    # Test read_file
    result = loader.execute_tool("policy_docs", "read_file", path="company_policy.md")
    if result["success"]:
        content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
        print(f"   ✅ Read policy file ({len(result['content'])} chars)")
        print(f"        Preview: {content_preview}")
    else:
        print(f"   ❌ read_file failed: {result['error']}")
    
    # Test search_file
    result = loader.execute_tool("policy_docs", "search_file", 
                                path="company_policy.md", query="refund")
    if result["success"]:
        print(f"   ✅ Found {len(result['matches'])} matches for 'refund'")
        if result["matches"]:
            print(f"        First match: {result['matches'][0]['text'][:100]}")
    else:
        print(f"   ❌ search_file failed: {result['error']}")

def test_script_generation(agent, loader):
    """Test script generation with tool context"""
    
    # Get a sample question
    samples = loader.get_examples(1)
    if not samples:
        print("   ❌ No samples available")
        return
    
    sample = samples[0]
    question = loader.get_example_input(sample)
    expected_answer = loader.get_example_output(sample)
    
    print(f"   📝 Sample question preview: {question[:200]}...")
    
    # Test tool interface generation
    print("   Testing tool interface generation...")
    tool_code = agent.create_tool_interface_code()
    if tool_code:
        print(f"   ✅ Generated tool interface ({len(tool_code)} chars)")
    else:
        print("   ℹ️  No tool interface (no assets)")
    
    # Create a simple test script
    print("   Creating test script...")
    test_script = '''
# Test script for customer service resolution
import json

def solve(question):
    """Solve a customer service ticket"""
    
    print("Analyzing customer service ticket...")
    
    # Test database access
    try:
        # Get customer database info
        db_result = execute_tool("customer_db", "describe_tables")
        if db_result["success"]:
            print(f"Database has {len(db_result['tables'])} tables")
        
        # Test policy document access
        policy_result = execute_tool("policy_docs", "read_file", path="company_policy.md")
        if policy_result["success"]:
            print(f"Policy document loaded ({len(policy_result['content'])} chars)")
        
        # Mock resolution plan
        resolution = {
            "customer_lookup": {"status": "found", "confidence": "high"},
            "policy_references": ["refund_policy", "escalation_procedure"],
            "actions": [{"type": "refund", "amount": 50.0, "reason": "policy_compliant"}],
            "priority": "medium"
        }
        
        return json.dumps(resolution, indent=2)
        
    except Exception as e:
        return f"Error: {str(e)}"

# Execute the solution
answer = solve(question)
print("Generated answer:", answer)
'''
    
    # Test script execution
    print("   Testing script execution...")
    try:
        result = agent.execute_script_with_tools(test_script, {"question": question})
        
        if result["success"]:
            print("   ✅ Script executed successfully")
            output = result.get("output", "")
            if output:
                print(f"        Output preview: {output[:300]}...")
        else:
            print(f"   ❌ Script execution failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"   ❌ Script execution exception: {e}")

def main():
    """Main function"""
    
    if len(sys.argv) != 2:
        print("Usage: python test_customer_service.py <dataset_directory>")
        print("Example: python test_customer_service.py customer_service_dataset")
        return
    
    dataset_dir = sys.argv[1]
    test_customer_service_system(dataset_dir)

if __name__ == "__main__":
    main() 