#!/usr/bin/env python3
"""
system_tools.py - System functions for script injection

This module contains the real implementations of system functions that get injected
into generated scripts at runtime. These functions provide core capabilities like
LLM calls, database access, file operations, and code execution.
"""

def call_llm(prompt, system_instruction=None):
    """Execute LLM API calls with proper error handling"""
    try:
        from google import genai
        from google.genai import types
        import os

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash", 
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    #thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                #thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
                contents=prompt
            )

        return response.text
    except Exception as e:
        print("Error calling Gemini API: " + str(e))
        return "Error: " + str(e)


def call_database(db_path, sql_query):
    """Execute SQL queries against a SQLite database with proper error handling"""
    import sqlite3
    import os
    from pathlib import Path
    
    print(f"  [DATABASE] Executing query on {db_path}: {sql_query[:100]}...")
    
    try:
        # Handle path resolution for sandbox environment
        if not os.path.isabs(db_path):
            # If relative path, assume it's in the workspace
            if '/workspace' in os.getcwd() or os.path.exists('/workspace'):
                workspace_path = Path('/workspace') / db_path
                if workspace_path.exists():
                    db_path = str(workspace_path)
                else:
                    # Try current working directory
                    current_path = Path(db_path)
                    if current_path.exists():
                        db_path = str(current_path)
                    else:
                        return f"Error: Database file not found at {db_path} (checked workspace and current directory)"
            else:
                # Not in sandbox, use relative to current directory
                current_path = Path(db_path)
                if current_path.exists():
                    db_path = str(current_path)
                else:
                    return f"Error: Database file not found at {db_path}"
        
        # Check if database file exists
        if not os.path.exists(db_path):
            return f"Error: Database file not found at {db_path}"
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        cursor = conn.cursor()
        
        try:
            # Execute the query
            cursor.execute(sql_query)
            
            # Handle different types of queries
            if sql_query.strip().upper().startswith(('SELECT', 'WITH', 'PRAGMA')):
                # Query returns results
                results = cursor.fetchall()
                if not results:
                    return []  # Empty list for no results
                
                # Return standard dictionary format
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            else:
                # Query modifies data (INSERT, UPDATE, DELETE, etc.)
                conn.commit()
                rows_affected = cursor.rowcount
                return {"success": True, "rows_affected": rows_affected}
                
        except sqlite3.Error as e:
            return {"error": f"SQL Error: {str(e)}"}
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        return {"error": f"Database Error: {str(e)}"}


# MCP Database Tools - Enhanced database functionality
class MCPDatabase:
    """MCP-style database tools for enhanced reliability and functionality"""
    
    def __init__(self, db_path=None):
        """Initialize with optional default database path"""
        self.default_db_path = db_path or "/Users/nickcdryan/Dev/textevolve/datasets/ticketworld/customer_database.db"
    
    def read_query(self, sql_query, db_path=None):
        """
        Execute a SELECT query with enhanced error handling and formatting
        
        Args:
            sql_query (str): SELECT query to execute
            db_path (str, optional): Database path, uses default if not provided
            
        Returns:
            list: List of dictionaries for results, or error dict
        """
        import sqlite3
        import os
        from pathlib import Path
        
        # Use provided path or default
        target_db = db_path or self.default_db_path
        
        print(f"  [MCP-DB] Read query on {os.path.basename(target_db)}: {sql_query[:100]}...")
        
        try:
            # Validate it's a SELECT query
            query_upper = sql_query.strip().upper()
            if not query_upper.startswith(('SELECT', 'WITH', 'PRAGMA')):
                return {"error": "read_query only accepts SELECT, WITH, or PRAGMA statements"}
            
            # Execute query using existing call_database logic
            result = call_database(target_db, sql_query)
            
            # Enhanced error handling
            if isinstance(result, dict) and 'error' in result:
                return result
            
            # Format results consistently
            if isinstance(result, list):
                print(f"  [MCP-DB] Retrieved {len(result)} rows")
                return result
            else:
                return {"error": "Unexpected result format from database"}
                
        except Exception as e:
            return {"error": f"MCP Database Error: {str(e)}"}
    
    def write_query(self, sql_query, db_path=None):
        """
        Execute an INSERT, UPDATE, or DELETE query
        
        Args:
            sql_query (str): Modification query to execute
            db_path (str, optional): Database path, uses default if not provided
            
        Returns:
            dict: Success status with rows affected, or error dict
        """
        import os
        
        # Use provided path or default
        target_db = db_path or self.default_db_path
        
        print(f"  [MCP-DB] Write query on {os.path.basename(target_db)}: {sql_query[:100]}...")
        
        try:
            # Validate it's a modification query
            query_upper = sql_query.strip().upper()
            if not query_upper.startswith(('INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER')):
                return {"error": "write_query only accepts INSERT, UPDATE, DELETE, CREATE, DROP, or ALTER statements"}
            
            # Execute query using existing call_database logic
            result = call_database(target_db, sql_query)
            
            if isinstance(result, dict):
                if 'error' in result:
                    return result
                elif 'success' in result:
                    print(f"  [MCP-DB] Modified {result.get('rows_affected', 0)} rows")
                    return result
            
            return {"error": "Unexpected result format from database"}
            
        except Exception as e:
            return {"error": f"MCP Database Error: {str(e)}"}
    
    def list_tables(self, db_path=None):
        """
        List all tables in the database
        
        Args:
            db_path (str, optional): Database path, uses default if not provided
            
        Returns:
            list: List of table names, or error dict
        """
        # Use provided path or default
        target_db = db_path or self.default_db_path
        
        query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        result = self.read_query(query, target_db)
        
        if isinstance(result, dict) and 'error' in result:
            return result
        
        # Extract just the table names
        table_names = [row['name'] for row in result if 'name' in row]
        print(f"  [MCP-DB] Found {len(table_names)} tables: {', '.join(table_names)}")
        return table_names
    
    def describe_table(self, table_name, db_path=None):
        """
        Get detailed schema information for a table
        
        Args:
            table_name (str): Name of the table to describe
            db_path (str, optional): Database path, uses default if not provided
            
        Returns:
            dict: Table schema information, or error dict
        """
        # Use provided path or default
        target_db = db_path or self.default_db_path
        
        # Get column information
        query = f"PRAGMA table_info({table_name})"
        result = self.read_query(query, target_db)
        
        if isinstance(result, dict) and 'error' in result:
            return result
        
        if not result:
            return {"error": f"Table '{table_name}' not found"}
        
        # Format schema information
        schema_info = {
            "table_name": table_name,
            "columns": [],
            "column_count": len(result)
        }
        
        for col in result:
            schema_info["columns"].append({
                "name": col.get("name"),
                "type": col.get("type"),
                "not_null": bool(col.get("notnull")),
                "default_value": col.get("dflt_value"),
                "primary_key": bool(col.get("pk"))
            })
        
        print(f"  [MCP-DB] Described table '{table_name}' with {len(result)} columns")
        return schema_info


# Create global MCP database instance for easy use in scripts
mcp_db = MCPDatabase()

# Convenience functions that scripts can use directly
def read_query(sql_query, db_path=None):
    """Execute a SELECT query using MCP database tools"""
    return mcp_db.read_query(sql_query, db_path)

def write_query(sql_query, db_path=None):
    """Execute a modification query using MCP database tools"""
    return mcp_db.write_query(sql_query, db_path)

def list_tables(db_path=None):
    """List all tables in the database"""
    return mcp_db.list_tables(db_path)

def describe_table(table_name, db_path=None):
    """Get schema information for a table"""
    return mcp_db.describe_table(table_name, db_path)


def read_file(filepath, start_line=None, end_line=None):
    """Read entire files or specific line ranges with proper error handling"""
    import os
    from pathlib import Path
    
    print(f"  [FILE] Reading file: {filepath}" + (f" (lines {start_line}-{end_line})" if start_line is not None else ""))
    
    try:
        # Handle path resolution for sandbox environment
        if not os.path.isabs(filepath):
            # If relative path, check workspace first then current directory
            if '/workspace' in os.getcwd() or os.path.exists('/workspace'):
                workspace_path = Path('/workspace') / filepath
                if workspace_path.exists():
                    filepath = str(workspace_path)
                else:
                    # Try current working directory
                    current_path = Path(filepath)
                    if current_path.exists():
                        filepath = str(current_path)
                    else:
                        return f"Error: File not found at {filepath} (checked workspace and current directory)"
            else:
                # Not in sandbox, use relative to current directory
                current_path = Path(filepath)
                if current_path.exists():
                    filepath = str(current_path)
                else:
                    return f"Error: File not found at {filepath}"
        
        # Check if file exists
        if not os.path.exists(filepath):
            return f"Error: File not found at {filepath}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(filepath):
            return f"Error: {filepath} is not a file"
        
        # Read the file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Handle line range selection
            if start_line is not None or end_line is not None:
                total_lines = len(lines)
                
                # Convert to 0-based indexing
                start_idx = (start_line - 1) if start_line is not None else 0
                end_idx = end_line if end_line is not None else total_lines
                
                # Validate line numbers
                if start_idx < 0:
                    start_idx = 0
                if end_idx > total_lines:
                    end_idx = total_lines
                if start_idx >= end_idx:
                    return f"Error: Invalid line range. File has {total_lines} lines."
                
                lines = lines[start_idx:end_idx]
                
                result = f"File content ({filepath}) - Lines {start_idx + 1} to {end_idx} of {total_lines}:\n"
                result += "=" * 50 + "\n"
                
                for i, line in enumerate(lines, start=start_idx + 1):
                    result += f"{i:4d}: {line.rstrip()}\n"
                
                return result
            else:
                # Return entire file
                total_lines = len(lines)
                if total_lines > 200:
                    # For very large files, show warning
                    result = f"File content ({filepath}) - {total_lines} lines (showing all):\n"
                    result += "WARNING: Large file - consider using line ranges for better performance\n"
                    result += "=" * 50 + "\n"
                else:
                    result = f"File content ({filepath}) - {total_lines} lines:\n"
                    result += "=" * 50 + "\n"
                
                for i, line in enumerate(lines, start=1):
                    result += f"{i:4d}: {line.rstrip()}\n"
                
                return result
                
        except UnicodeDecodeError:
            return f"Error: Unable to read {filepath} - file appears to be binary"
        except PermissionError:
            return f"Error: Permission denied reading {filepath}"
            
    except Exception as e:
        return f"File Error: {str(e)}"


def search_file(filepath, pattern, case_sensitive=True, show_line_numbers=True, context_lines=0, max_results=50):
    """Search for patterns in files with grep-like functionality"""
    import os
    import re
    from pathlib import Path
    
    print(f"  [SEARCH] Searching in {filepath} for pattern: '{pattern}'" + 
          ("" if case_sensitive else " (case-insensitive)"))
    
    try:
        # Handle path resolution for sandbox environment
        if not os.path.isabs(filepath):
            # If relative path, check workspace first then current directory
            if '/workspace' in os.getcwd() or os.path.exists('/workspace'):
                workspace_path = Path('/workspace') / filepath
                if workspace_path.exists():
                    filepath = str(workspace_path)
                else:
                    # Try current working directory
                    current_path = Path(filepath)
                    if current_path.exists():
                        filepath = str(current_path)
                    else:
                        return f"Error: File not found at {filepath} (checked workspace and current directory)"
            else:
                # Not in sandbox, use relative to current directory
                current_path = Path(filepath)
                if current_path.exists():
                    filepath = str(current_path)
                else:
                    return f"Error: File not found at {filepath}"
        
        # Check if file exists
        if not os.path.exists(filepath):
            return f"Error: File not found at {filepath}"
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(filepath):
            return f"Error: {filepath} is not a file"
        
        # Read the file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"Error: Unable to read {filepath} - file appears to be binary"
        except PermissionError:
            return f"Error: Permission denied reading {filepath}"
        
        # Prepare regex pattern
        regex_flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled_pattern = re.compile(pattern, regex_flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {str(e)}"
        
        # Find matches
        matches = []
        for line_num, line in enumerate(lines, start=1):
            if compiled_pattern.search(line):
                matches.append(line_num)
                if len(matches) >= max_results:
                    break
        
        if not matches:
            return f"No matches found for pattern '{pattern}' in {filepath}"
        
        # Build result with context
        result = f"Search results for '{pattern}' in {filepath}:\n"
        if len(matches) >= max_results:
            result += f"Found {len(matches)}+ matches (showing first {max_results}):\n"
        else:
            result += f"Found {len(matches)} matches:\n"
        result += "=" * 50 + "\n"
        
        displayed_lines = set()
        
        for match_line in matches:
            # Calculate context range
            start_context = max(1, match_line - context_lines)
            end_context = min(len(lines), match_line + context_lines)
            
            # Add separator if we're not continuing from previous context
            if displayed_lines and start_context > max(displayed_lines) + 1:
                result += "\n" + "-" * 20 + "\n"
            
            # Show context lines
            for line_num in range(start_context, end_context + 1):
                if line_num in displayed_lines:
                    continue
                
                line_content = lines[line_num - 1].rstrip()
                
                if line_num == match_line:
                    # Highlight the matching line
                    prefix = ">>> " if show_line_numbers else ">>> "
                    line_display = f"{line_num:4d}: {line_content}" if show_line_numbers else line_content
                    result += f"{prefix}{line_display}\n"
                else:
                    # Context line
                    prefix = "    " if show_line_numbers else ""
                    line_display = f"{line_num:4d}: {line_content}" if show_line_numbers else line_content
                    result += f"{prefix}{line_display}\n"
                
                displayed_lines.add(line_num)
        
        return result
        
    except Exception as e:
        return f"Search Error: {str(e)}"


def execute_code(code_str, timeout=10):
    """Execute Python code with automatic package installation and proper scoping"""
    import sys
    import re
    import subprocess
    from io import StringIO

    print("  [SYSTEM] Auto-installing execute_code() with scope fix")

    # Clean markdown formatting
    patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```python\s*(.*?)```', 
        r'```\s*\n(.*?)\n```',
        r'```\s*(.*?)```'
    ]

    cleaned_code = code_str.strip()
    for pattern in patterns:
        match = re.search(pattern, code_str, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned_code = match.group(1).strip()
            print("  [CLEANING] Removed markdown")
            break

    # Function to install a package
    def install_package(package_name):
        try:
            print("  [INSTALLING] Installing " + package_name + "...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print("  [SUCCESS] " + package_name + " installed successfully")
                return True
            else:
                print("  [FAILED] Could not install " + package_name + ": " + result.stderr)
                return False
        except Exception as e:
            print("  [ERROR] Installation error: " + str(e))
            return False

    # Execute with proper scoping and auto-installation retry
    max_install_attempts = 3
    attempt = 0

    while attempt <= max_install_attempts:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # CRITICAL FIX: Provide explicit globals and locals
            # This ensures imports are available to functions defined in the code
            exec_namespace = {}
            exec(cleaned_code, exec_namespace, exec_namespace)

            # Success!
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            output = stdout_capture.getvalue().strip()
            return output if output else "Code executed successfully"

        except ModuleNotFoundError as e:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Extract the missing module name
            module_name = str(e).split("'")[1] if "'" in str(e) else None

            if module_name and attempt < max_install_attempts:
                print("  [MISSING] Module '" + module_name + "' not found, attempting to install...")

                # Try to install the missing package
                if install_package(module_name):
                    attempt += 1
                    print("  [RETRY] Retrying code execution (attempt " + str(attempt + 1) + ")...")
                    continue
                else:
                    return "Error: Could not install required package '" + module_name + "'"
            else:
                return "Error: " + str(e)

        except Exception as e:
            sys.stdout = old_stdout  
            sys.stderr = old_stderr
            return "Error: " + str(e)

        attempt += 1

    return "Error: Maximum installation attempts exceeded"


# For testing and validation
if __name__ == "__main__":
    print("system_tools.py - System functions available for injection")
    print("Functions: call_llm, call_database, read_file, search_file, execute_code") 