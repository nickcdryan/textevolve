#!/usr/bin/env python
"""
Enhanced Agent System with Asset Support
Extends the base AgentSystem to work with asset-enabled datasets and provide tools
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from agent_system import AgentSystem
from asset_dataset_loader import AssetDatasetLoader

class EnhancedAgentSystem(AgentSystem):
    """AgentSystem with asset support and tool integration"""
    
    def __init__(self, dataset_loader=None, use_sandbox=True):
        """
        Initialize enhanced agent system with asset support
        
        Args:
            dataset_loader: An AssetDatasetLoader instance
            use_sandbox: Whether to use Docker sandbox for code execution
        """
        
        # Initialize parent
        super().__init__(dataset_loader, use_sandbox)
        
        # Asset-specific attributes
        self.has_assets = False
        self.available_tools = {}
        self.tool_documentation = ""
        self.schema_documentation = ""
        
        # Check if this is an asset-enabled dataset
        if isinstance(dataset_loader, AssetDatasetLoader):
            self.has_assets = True
            asset_info = dataset_loader.get_asset_info()
            self.available_tools = asset_info["available_tools"]
            self.tool_documentation = asset_info["tool_documentation"] 
            self.schema_documentation = asset_info["schema_documentation"]
            
            print("Enhanced Agent System initialized with assets:")
            print(f"Available tools: {list(self.available_tools.keys())}")
    
    def generate_system_prompt_context(self) -> str:
        """Generate enhanced system prompt that includes tool documentation"""
        
        context_parts = []
        
        # Base context
        context_parts.append("You are an AI agent solving complex problems by writing and executing Python scripts.")
        
        if self.has_assets:
            context_parts.append("\nYou have access to external tools and data sources:")
            context_parts.append(self.tool_documentation)
            
            if self.schema_documentation:
                context_parts.append(self.schema_documentation)
            
            context_parts.append("\nCRITICAL TOOL USAGE REQUIREMENTS:")
            context_parts.append("=" * 40)
            context_parts.append("1. You MUST use execute_tool() to query databases and read files")
            context_parts.append("2. DO NOT hardcode customer data or policy information")
            context_parts.append("3. DO NOT copy-paste data from the question - query it dynamically")
            context_parts.append("4. ALWAYS use tools to verify customer information from the database")
            context_parts.append("5. ALWAYS use tools to read the actual policy documents")
            context_parts.append("")
            context_parts.append("Tool syntax:")
            context_parts.append("result = execute_tool(asset_name, tool_name, **kwargs)")
            context_parts.append("Example: result = execute_tool('customer_db', 'sql_query', query='SELECT * FROM customers WHERE email = ?', params=['user@email.com'])")
            context_parts.append("Example: result = execute_tool('policy_docs', 'read_file', path='company_policy.md')")
            context_parts.append("Always check result['success'] before using result data.")
            context_parts.append("")
            context_parts.append("IMPORTANT: The question may contain sample data for context, but you must")
            context_parts.append("query the ACTUAL database and files using tools, not use the sample data.")
        
        return "\n".join(context_parts)
    
    def create_tool_interface_code(self) -> str:
        """Generate Python code that provides the tool interface for scripts"""
        
        if not self.has_assets:
            return ""
        
        tool_code = '''
# Tool Interface - Available in all generated scripts
def execute_tool(asset_name: str, tool_name: str, **kwargs) -> dict:
    """
    Execute a tool on a specific asset
    
    Args:
        asset_name: Name of the asset to operate on
        tool_name: Name of the tool to execute
        **kwargs: Tool-specific arguments
        
    Returns:
        dict: Result with 'success' field and tool-specific data
    """
    import json
    import sys
    
    # This will be replaced with actual tool execution in the sandbox
    # For now, return mock data for script validation
    if asset_name == "customer_db":
        if tool_name == "sql_query":
            return {
                "success": True,
                "results": [{"id": 1, "name": "Test Customer", "email": "test@example.com"}],
                "columns": ["id", "name", "email"],
                "row_count": 1
            }
        elif tool_name == "describe_tables":
            return {
                "success": True,
                "tables": {
                    "customers": [{"name": "id", "type": "INTEGER", "primary_key": True}],
                    "orders": [{"name": "id", "type": "INTEGER", "primary_key": True}]
                }
            }
    elif asset_name == "policy_docs":
        if tool_name == "read_file":
            return {
                "success": True,
                "content": "Sample policy content...",
                "path": kwargs.get("path", "")
            }
        elif tool_name == "search_file":
            return {
                "success": True,
                "matches": [{"line": 10, "text": "refund policy section"}],
                "query": kwargs.get("query", "")
            }
    
    return {"success": False, "error": f"Tool '{tool_name}' not found for asset '{asset_name}'"}

'''
        return tool_code
    
    def execute_script_with_tools(self, script: str, sample: Dict) -> Dict:
        """Execute script with tool support"""
        
        if not self.has_assets:
            # Fall back to parent implementation
            return super().execute_script(script, sample)
        
        # Inject tool interface into script
        tool_interface = self.create_tool_interface_code()
        enhanced_script = tool_interface + "\n\n" + script
        
        # Execute in sandbox with tool support
        if self.use_sandbox and self.sandbox:
            return self._execute_in_sandbox_with_tools(enhanced_script, sample)
        else:
            return self._execute_locally_with_tools(enhanced_script, sample)
    
    def _execute_in_sandbox_with_tools(self, script: str, sample: Dict) -> Dict:
        """Execute script in sandbox with tool execution support"""
        
        # For now, use standard sandbox execution
        # In a full implementation, this would set up tool execution context
        result = self.sandbox.execute_script(script, sample)
        
        # Post-process to handle tool calls
        return self._process_tool_calls_in_result(result)
    
    def _execute_locally_with_tools(self, script: str, sample: Dict) -> Dict:
        """Execute script locally with tool execution support"""
        
        # Create a temporary script file with tool support
        script_path = self.scripts_dir / f"enhanced_script_{self.current_iteration}.py"
        
        # Create enhanced script with real tool execution
        enhanced_script = self._create_script_with_real_tools(script, sample)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_script)
        
        # Execute and return result
        return super().execute_script(enhanced_script, sample)
    
    def _create_script_with_real_tools(self, script: str, sample: Dict) -> str:
        """Create script with real tool execution capabilities"""
        
        # Extract variables from sample
        question = sample.get("question", "")
        sample_id = sample.get("id", f"example_{self.current_iteration}")
        
        # Import necessary modules
        imports = '''
import json
import sys
from pathlib import Path

# Add project root to path to import our modules
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

'''
        
        # Variable injection
        variables = f'''
# Sample data injection
question = {repr(question)}
sample_id = {repr(sample_id)}

'''
        
        # Real tool execution function with access to the dataset loader  
        tool_function = f'''
def execute_tool(asset_name: str, tool_name: str, **kwargs) -> dict:
    """Execute a tool on a specific asset - ALWAYS use real data"""
    try:
        # Import the dataset loader module to access the global instance
        import sys
        import os
        from pathlib import Path
        
        # Add project root to sys.path to import our modules
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Try to access the asset dataset loader directly
        try:
            from asset_dataset_loader import AssetDatasetLoader
            # Look for a global dataset_loader variable in the current script's globals
            if 'dataset_loader' in globals():
                return globals()['dataset_loader'].execute_tool(asset_name, tool_name, **kwargs)
        except ImportError:
            pass
        
        # For customer service data, implement direct asset access
        if asset_name == "customer_db":
            return execute_database_tool(tool_name, **kwargs)
        elif asset_name == "policy_docs":
            return execute_filesystem_tool(tool_name, **kwargs)
        
        return {{"success": False, "error": f"Asset '{{asset_name}}' not found"}}
        
    except Exception as e:
        return {{"success": False, "error": str(e)}}

def execute_database_tool(tool_name: str, **kwargs) -> dict:
    """Execute database tools on the customer service database"""
    try:
        import sqlite3
        from pathlib import Path
        
        # Use the actual database path
        db_path = Path("synthetic_data/customer_service_dataset/assets/customer_service.db")
        if not db_path.exists():
            # Try relative to current script location
            db_path = Path(__file__).parent / "synthetic_data/customer_service_dataset/assets/customer_service.db"
        
        if not db_path.exists():
            return {{"success": False, "error": f"Database not found at {{db_path}}"}}
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        if tool_name == "sql_query":
            query = kwargs.get("query", "")
            params = kwargs.get("params", [])
            cursor.execute(query, params)
            
            if query.strip().upper().startswith("SELECT"):
                results = [dict(zip([col[0] for col in cursor.description], row)) 
                          for row in cursor.fetchall()]
                return {{"success": True, "results": results, "row_count": len(results)}}
            else:
                conn.commit()
                return {{"success": True, "rows_affected": cursor.rowcount}}
                
        elif tool_name == "describe_tables":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            return {{"success": True, "tables": tables}}
            
        elif tool_name == "get_schema":
            table_name = kwargs.get("table_name", "")
            cursor.execute(f"PRAGMA table_info({{table_name}})")
            schema = cursor.fetchall()
            return {{"success": True, "schema": schema}}
        
        conn.close()
        return {{"success": False, "error": f"Unknown database tool: {{tool_name}}"}}
        
    except Exception as e:
        return {{"success": False, "error": str(e)}}

def execute_filesystem_tool(tool_name: str, **kwargs) -> dict:
    """Execute filesystem tools on policy documents"""
    try:
        from pathlib import Path
        
        # Use the actual policy docs path
        docs_path = Path("synthetic_data/customer_service_dataset/assets")
        if not docs_path.exists():
            # Try relative to current script location  
            docs_path = Path(__file__).parent / "synthetic_data/customer_service_dataset/assets"
        
        if not docs_path.exists():
            return {{"success": False, "error": f"Documents directory not found at {{docs_path}}"}}
        
        if tool_name == "read_file":
            file_path = docs_path / kwargs.get("path", "")
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {{"success": True, "content": content, "path": str(file_path)}}
            else:
                return {{"success": False, "error": f"File not found: {{file_path}}"}}
                
        elif tool_name == "search_file":
            file_path = docs_path / kwargs.get("path", "")
            query = kwargs.get("query", "")
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                matches = []
                for i, line in enumerate(lines, 1):
                    if query.lower() in line.lower():
                        matches.append({{"line": i, "text": line.strip()}})
                return {{"success": True, "matches": matches, "query": query}}
            else:
                return {{"success": False, "error": f"File not found: {{file_path}}"}}
                
        elif tool_name == "list_directory":
            dir_path = docs_path / kwargs.get("path", "")
            if dir_path.exists() and dir_path.is_dir():
                files = [f.name for f in dir_path.iterdir()]
                return {{"success": True, "files": files, "path": str(dir_path)}}
            else:
                return {{"success": False, "error": f"Directory not found: {{dir_path}}"}}
        
        return {{"success": False, "error": f"Unknown filesystem tool: {{tool_name}}"}}
        
    except Exception as e:
        return {{"success": False, "error": str(e)}}

'''
        
        # Wrap the script in a main function if it doesn't already have one
        if "def main(" not in script:
            # Indent the user script properly
            indented_script = '\n'.join(['    ' + line if line.strip() else '' for line in script.split('\n')])
            
            # Create main function wrapper
            script_wrapper = f'''
def main(question):
    """Main entry point for script execution"""
{indented_script}
    
    # Call the solve function if it exists
    if 'solve' in globals():
        return solve(question)
    else:
        return "Error: No solve function found in script"
'''
            return imports + variables + tool_function + script_wrapper
        else:
            return imports + variables + tool_function + "\n\n" + script
    
    def _process_tool_calls_in_result(self, result: Dict) -> Dict:
        """Process any tool calls that occurred during script execution"""
        
        # Check for tool call logs in the output
        output = result.get("output", "")
        
        # Look for tool execution patterns
        # This is a simplified approach - in practice, we'd use more sophisticated
        # inter-process communication or shared context
        
        return result
    
    def generate_script(self) -> str:
        """Generate script with enhanced prompting that includes tool context"""
        
        # Get the base script generation prompt
        base_script = super().generate_script()
        
        # If we have assets, enhance the prompt context
        if self.has_assets:
            # The tool documentation is already included in system prompt context
            # No need to modify the generated script itself
            pass
        
        return base_script
    
    def cleanup(self):
        """Clean up resources including asset managers"""
        
        # Clean up dataset loader assets if present
        if isinstance(self.dataset_loader, AssetDatasetLoader):
            self.dataset_loader.cleanup()

# Override the execute_script method to use tool support
def monkey_patch_execute_script(cls):
    """Monkey patch to use enhanced script execution"""
    original_execute_script = cls.execute_script
    
    def enhanced_execute_script(self, script: str, sample: Dict) -> Dict:
        if hasattr(self, 'has_assets') and self.has_assets:
            return self.execute_script_with_tools(script, sample)
        else:
            return original_execute_script(self, script, sample)
    
    return enhanced_execute_script

# Apply the monkey patch
EnhancedAgentSystem.execute_script = monkey_patch_execute_script(EnhancedAgentSystem) 