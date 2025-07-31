# Asset-Based Dataset System for TextEvolve

This implementation extends TextEvolve with support for datasets that require external assets like databases, filesystems, and APIs. The system provides tools to agents, allowing them to dynamically query and interact with these assets during problem-solving.

## Overview

The asset-based system allows datasets to declare external dependencies through a YAML configuration file. The system automatically provisions the required tools and makes them available to reasoning agents.

## Architecture

```
Dataset Directory/
├── dataset.yaml          # Asset configuration
├── evaluation_data.json  # Questions and answers
└── assets/               # External assets
    ├── database.db       # SQLite database
    ├── policy_docs/      # Filesystem assets
    └── README.md         # Documentation
```

## Components

### 1. Asset Manager (`asset_manager.py`)
- Provisions MCP servers based on asset types
- Manages tool execution and lifecycle
- Supports filesystem and SQLite assets (extensible)

### 2. Asset Dataset Loader (`asset_dataset_loader.py`) 
- Extends base DatasetLoader with asset support
- Loads and provisions assets automatically
- Provides tool execution interface

### 3. Enhanced Agent System (`enhanced_agent_system.py`)
- Integrates with asset-enabled datasets
- Generates system prompts with tool documentation
- Provides tool interface to generated scripts

## Supported Asset Types

| Asset Type | Tools Available | Description |
|------------|----------------|-------------|
| `filesystem` | `read_file`, `search_file`, `list_directory` | File system access |
| `sqlite` | `sql_query`, `describe_tables`, `get_schema` | SQLite database queries |

*Additional types (postgres, vector_db, rest_api) can be added easily*

## Dataset Configuration

Create a `dataset.yaml` file in your dataset directory:

```yaml
name: "my_dataset"
description: "Dataset description"
version: "1.0"
assets:
  - name: "knowledge_base"
    type: "filesystem" 
    path: "./assets/docs/"
    description: "Knowledge base documents"
  - name: "user_database"
    type: "sqlite"
    path: "./assets/users.db"
    description: "User and transaction database"
```

## Tool Usage in Generated Scripts

Agents can call tools using the `execute_tool` function:

```python
# Query database
result = execute_tool("user_database", "sql_query", 
                     query="SELECT * FROM users WHERE email = ?", 
                     params=["user@example.com"])

if result["success"]:
    users = result["results"]
    print(f"Found {len(users)} users")

# Read policy document
result = execute_tool("knowledge_base", "read_file", 
                     path="refund_policy.md")

if result["success"]:
    policy_text = result["content"]
    
# Search for specific information
result = execute_tool("knowledge_base", "search_file",
                     path="company_policy.md", 
                     query="shipping error")
```

## Example: Customer Service Dataset

### Step 1: Generate Dataset

```bash
cd synthetic_data
export GEMINI_API_KEY="your_api_key"
python create_customer_service_dataset.py
```

### Step 2: Test the System

```bash
python test_customer_service.py customer_service_dataset
```

This will:
1. Load the dataset with assets
2. Provision filesystem and database tools
3. Test tool execution
4. Create an enhanced agent system
5. Run a sample script with tool access

## Integration with Existing System

The asset system is designed to be backward-compatible:

- Datasets without `dataset.yaml` work as before
- No changes needed to existing datasets
- Asset-enabled datasets get additional tools automatically

## Tool Implementation

Tools are implemented with a consistent interface:

```python
def execute_tool(asset_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool and return standardized result"""
    return {
        "success": bool,     # Whether execution succeeded
        "error": str,        # Error message if failed
        "results": Any,      # Tool-specific results
        # Additional tool-specific fields
    }
```

## Extending with New Asset Types

To add a new asset type:

1. Add mapping in `AssetManager.ASSET_TYPE_MAPPING`
2. Implement provisioning method (`_provision_[type]_server`)
3. Implement tool execution method (`_execute_[type]_tool`)
4. Add tool documentation in `get_tool_documentation`

## Future Enhancements

1. **Real MCP Integration**: Replace simulated servers with actual MCP protocol
2. **Cloud Assets**: Support for hosted databases and APIs
3. **Vector Databases**: Add Chroma, Pinecone, Weaviate support
4. **API Integration**: RESTful and GraphQL API support
5. **Caching**: Tool result caching for performance
6. **Security**: Sandboxed tool execution and permission management

## Benefits

1. **Realistic Testing**: Agents interact with data as humans would
2. **Scalable**: Assets separate from reasoning logic
3. **Dynamic**: Agents query exactly what they need
4. **Extensible**: Easy to add new asset types and tools
5. **Reusable**: Same tools work across different datasets

This asset-based approach enables testing LLM agents on realistic tasks that require external data access, making TextEvolve suitable for real-world application development and evaluation. 