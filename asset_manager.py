#!/usr/bin/env python
"""
Asset Management System for TextEvolve
Handles dataset assets like databases, filesystems, and APIs through MCP servers
"""

import os
import yaml
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class AssetConfig:
    """Configuration for a dataset asset"""
    name: str
    type: str
    config: Dict[str, Any]
    description: Optional[str] = None

@dataclass
class MCPServer:
    """Information about a running MCP server"""
    name: str
    server_type: str
    process: Any
    port: Optional[int] = None
    config: Optional[Dict] = None
    tools: List[str] = None

class AssetManager:
    """Manages dataset assets and MCP server provisioning"""
    
    # Mapping from asset types to MCP server implementations
    ASSET_TYPE_MAPPING = {
        "filesystem": {
            "server": "mcp-filesystem-server",
            "tools": ["read_file", "search_file", "list_directory"]
        },
        "sqlite": {
            "server": "mcp-sqlite-server", 
            "tools": ["sql_query", "describe_tables", "get_schema"]
        },
        "postgres": {
            "server": "mcp-postgres-server",
            "tools": ["sql_query", "describe_tables", "get_schema"]
        },
        "vector_db": {
            "server": "mcp-chromadb-server",
            "tools": ["vector_search", "add_documents", "list_collections"]
        },
        "rest_api": {
            "server": "mcp-fetch-server",
            "tools": ["api_call", "get_request", "post_request"]
        }
    }
    
    def __init__(self):
        self.active_servers: Dict[str, MCPServer] = {}
        self.temp_dir = tempfile.mkdtemp(prefix="textevolve_assets_")
        
    def load_dataset_config(self, config_path: str) -> Tuple[Dict[str, Any], List[AssetConfig]]:
        """Load dataset configuration and extract asset definitions"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Store the dataset directory for resolving relative paths
        self.dataset_dir = config_path.parent
            
        # Extract dataset metadata
        dataset_info = {
            "name": config.get("name", "unknown"),
            "description": config.get("description", ""),
            "version": config.get("version", "1.0")
        }
        
        # Parse asset configurations
        assets = []
        for asset_data in config.get("assets", []):
            asset = AssetConfig(
                name=asset_data.get("name", f"asset_{len(assets)}"),
                type=asset_data["type"],
                config=asset_data,
                description=asset_data.get("description")
            )
            assets.append(asset)
            
        return dataset_info, assets
    
    def provision_assets(self, assets: List[AssetConfig]) -> Dict[str, List[str]]:
        """Provision MCP servers for all assets and return available tools"""
        all_tools = {}
        
        for asset in assets:
            print(f"Provisioning asset: {asset.name} ({asset.type})")
            
            # Get server info for this asset type
            if asset.type not in self.ASSET_TYPE_MAPPING:
                print(f"Warning: Unknown asset type '{asset.type}', skipping")
                continue
                
            server_info = self.ASSET_TYPE_MAPPING[asset.type]
            
            # Provision the server
            server = self._provision_mcp_server(asset, server_info)
            if server:
                self.active_servers[asset.name] = server
                all_tools[asset.name] = server.tools
                print(f"  ✅ Provisioned {asset.name} with tools: {', '.join(server.tools)}")
            else:
                print(f"  ❌ Failed to provision {asset.name}")
                
        return all_tools
    
    def _provision_mcp_server(self, asset: AssetConfig, server_info: Dict) -> Optional[MCPServer]:
        """Provision a single MCP server based on asset configuration"""
        
        if asset.type == "filesystem":
            return self._provision_filesystem_server(asset)
        elif asset.type == "sqlite":
            return self._provision_sqlite_server(asset)
        else:
            # For now, we'll implement filesystem and sqlite
            # Other types can be added later
            print(f"Asset type '{asset.type}' not yet implemented")
            return None
    
    def _provision_filesystem_server(self, asset: AssetConfig) -> Optional[MCPServer]:
        """Provision a filesystem MCP server"""
        
        # For now, we'll simulate MCP server functionality
        # In a real implementation, this would start actual MCP servers
        
        # Validate filesystem path (resolve relative to dataset directory)
        fs_path = asset.config.get("path", ".")
        if not Path(fs_path).is_absolute() and hasattr(self, 'dataset_dir'):
            fs_path = self.dataset_dir / fs_path
        fs_path = Path(fs_path).resolve()
        
        if not fs_path.exists():
            print(f"Filesystem path does not exist: {fs_path}")
            return None
            
        server = MCPServer(
            name=asset.name,
            server_type="filesystem",
            process=None,  # Simulated for now
            tools=["read_file", "search_file", "list_directory"],
            config={"root_path": fs_path}
        )
        
        return server
    
    def _provision_sqlite_server(self, asset: AssetConfig) -> Optional[MCPServer]:
        """Provision a SQLite MCP server"""
        
        db_path = asset.config.get("path")
        if not db_path:
            print("SQLite asset missing 'path' configuration")
            return None
            
        # Resolve relative to dataset directory
        if not Path(db_path).is_absolute() and hasattr(self, 'dataset_dir'):
            db_path = self.dataset_dir / db_path
        db_path = Path(db_path).resolve()
        
        if not db_path.exists():
            print(f"SQLite database does not exist: {db_path}")
            return None
            
        # Test connection
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            print(f"  Found tables: {', '.join(tables)}")
            
        except Exception as e:
            print(f"Failed to connect to SQLite database: {e}")
            return None
            
        server = MCPServer(
            name=asset.name,
            server_type="sqlite",
            process=None,  # Simulated for now
            tools=["sql_query", "describe_tables", "get_schema"],
            config={"db_path": str(db_path), "tables": tables}
        )
        
        return server
    
    def get_tool_documentation(self) -> str:
        """Generate documentation for all available tools"""
        
        if not self.active_servers:
            return "No tools available."
            
        docs = []
        docs.append("AVAILABLE TOOLS:")
        docs.append("=" * 50)
        
        for asset_name, server in self.active_servers.items():
            docs.append(f"\n{asset_name.upper()} ({server.server_type}):")
            docs.append("-" * 30)
            
            if server.server_type == "filesystem":
                root_path = server.config.get("root_path", ".")
                docs.append(f"Root path: {root_path}")
                docs.append("Tools:")
                docs.append("- read_file(path): Read contents of a file")
                docs.append("- search_file(path, query): Search for text within a file")
                docs.append("- list_directory(path): List contents of a directory")
                
            elif server.server_type == "sqlite":
                db_path = server.config.get("db_path")
                tables = server.config.get("tables", [])
                docs.append(f"Database: {db_path}")
                docs.append(f"Tables: {', '.join(tables)}")
                docs.append("Tools:")
                docs.append("- sql_query(query): Execute SQL query and return results")
                docs.append("- describe_tables(): Get schema information for all tables")
                docs.append("- get_schema(table_name): Get detailed schema for specific table")
                
        return "\n".join(docs)
    
    def get_database_schemas(self) -> str:
        """Get detailed schema information for all databases"""
        
        schemas = []
        
        for asset_name, server in self.active_servers.items():
            if server.server_type == "sqlite":
                schema_info = self._get_sqlite_schema(server)
                if schema_info:
                    schemas.append(f"\n{asset_name.upper()} DATABASE SCHEMA:")
                    schemas.append("=" * 40)
                    schemas.append(schema_info)
                    
        return "\n".join(schemas) if schemas else ""
    
    def _get_sqlite_schema(self, server: MCPServer) -> str:
        """Get detailed schema for SQLite database"""
        
        db_path = server.config.get("db_path")
        if not db_path:
            return ""
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_lines = []
            for table in tables:
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                
                schema_lines.append(f"\nTable: {table}")
                schema_lines.append("-" * (7 + len(table)))
                
                for col in columns:
                    col_name, col_type, not_null, default, pk = col[1], col[2], col[3], col[4], col[5]
                    constraints = []
                    if pk:
                        constraints.append("PRIMARY KEY")
                    if not_null:
                        constraints.append("NOT NULL")
                    if default is not None:
                        constraints.append(f"DEFAULT {default}")
                        
                    constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                    schema_lines.append(f"  {col_name}: {col_type}{constraint_str}")
                    
            conn.close()
            return "\n".join(schema_lines)
            
        except Exception as e:
            print(f"Error getting schema for {server.name}: {e}")
            return ""
    
    def execute_tool(self, asset_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool on a specific asset"""
        
        if asset_name not in self.active_servers:
            return {"success": False, "error": f"Asset '{asset_name}' not found"}
            
        server = self.active_servers[asset_name]
        
        if tool_name not in server.tools:
            return {"success": False, "error": f"Tool '{tool_name}' not available for asset '{asset_name}'"}
            
        # Route to appropriate handler
        if server.server_type == "filesystem":
            return self._execute_filesystem_tool(server, tool_name, **kwargs)
        elif server.server_type == "sqlite":
            return self._execute_sqlite_tool(server, tool_name, **kwargs)
        else:
            return {"success": False, "error": f"Server type '{server.server_type}' not implemented"}
    
    def _execute_filesystem_tool(self, server: MCPServer, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute filesystem tools"""
        
        root_path = Path(server.config.get("root_path", "."))
        
        try:
            if tool_name == "read_file":
                file_path = kwargs.get("path", "")
                full_path = root_path / file_path
                
                if not full_path.exists():
                    return {"success": False, "error": f"File not found: {file_path}"}
                    
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                return {"success": True, "content": content, "path": str(full_path)}
                
            elif tool_name == "search_file":
                file_path = kwargs.get("path", "")
                query = kwargs.get("query", "")
                full_path = root_path / file_path
                
                if not full_path.exists():
                    return {"success": False, "error": f"File not found: {file_path}"}
                    
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple text search
                lines = content.split('\n')
                matches = []
                for i, line in enumerate(lines):
                    if query.lower() in line.lower():
                        matches.append({"line": i + 1, "text": line.strip()})
                        
                return {"success": True, "matches": matches, "query": query, "path": str(full_path)}
                
            elif tool_name == "list_directory":
                dir_path = kwargs.get("path", "")
                full_path = root_path / dir_path
                
                if not full_path.exists():
                    return {"success": False, "error": f"Directory not found: {dir_path}"}
                    
                if not full_path.is_dir():
                    return {"success": False, "error": f"Path is not a directory: {dir_path}"}
                    
                entries = []
                for item in full_path.iterdir():
                    entries.append({
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None
                    })
                    
                return {"success": True, "entries": entries, "path": str(full_path)}
                
        except Exception as e:
            return {"success": False, "error": f"Tool execution failed: {str(e)}"}
            
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def _execute_sqlite_tool(self, server: MCPServer, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute SQLite tools"""
        
        db_path = server.config.get("db_path")
        
        try:
            if tool_name == "sql_query":
                query = kwargs.get("query", "")
                if not query:
                    return {"success": False, "error": "SQL query is required"}
                    
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Execute query
                cursor.execute(query)
                
                # Get results
                if query.strip().upper().startswith("SELECT"):
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()
                    
                    # Convert to list of dictionaries
                    results = []
                    for row in rows:
                        results.append(dict(zip(columns, row)))
                        
                    conn.close()
                    return {"success": True, "results": results, "columns": columns, "row_count": len(results)}
                else:
                    # For non-SELECT queries
                    conn.commit()
                    affected_rows = cursor.rowcount
                    conn.close()
                    return {"success": True, "affected_rows": affected_rows}
                    
            elif tool_name == "describe_tables":
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                table_info = {}
                for table in tables:
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = cursor.fetchall()
                    table_info[table] = [
                        {"name": col[1], "type": col[2], "nullable": not col[3], "primary_key": bool(col[5])}
                        for col in columns
                    ]
                    
                conn.close()
                return {"success": True, "tables": table_info}
                
            elif tool_name == "get_schema":
                table_name = kwargs.get("table_name", "")
                if not table_name:
                    return {"success": False, "error": "table_name is required"}
                    
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
                if not cursor.fetchone():
                    conn.close()
                    return {"success": False, "error": f"Table '{table_name}' not found"}
                    
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                schema = {
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "nullable": not col[3],
                            "default": col[4],
                            "primary_key": bool(col[5])
                        }
                        for col in columns
                    ]
                }
                
                conn.close()
                return {"success": True, "schema": schema}
                
        except Exception as e:
            return {"success": False, "error": f"Database tool execution failed: {str(e)}"}
            
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def cleanup(self):
        """Clean up all running servers and temporary files"""
        
        # In a real implementation, this would terminate MCP server processes
        for server_name, server in self.active_servers.items():
            print(f"Cleaning up server: {server_name}")
            
        self.active_servers.clear()
        
        # Clean up temp directory
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir) 