#!/usr/bin/env python
"""
Asset-enabled Dataset Loader for TextEvolve
Extends the base DatasetLoader to support datasets with external assets
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataset_loader import DatasetLoader
from asset_manager import AssetManager, AssetConfig

class AssetDatasetLoader(DatasetLoader):
    """Extended DatasetLoader that supports external assets via MCP servers"""
    
    def __init__(self, 
                 dataset_path: str,
                 load_examples_fn: Callable[[str], List[Any]],
                 get_input_fn: Callable[[Any], Any],
                 get_output_fn: Callable[[Any], Any],
                 shuffle: bool = True,
                 random_seed: int = 42,
                 config_file: str = "dataset.yaml"):
        """
        Initialize asset-enabled dataset loader
        
        Args:
            dataset_path: Path to the dataset file
            load_examples_fn: Function to load examples from the dataset
            get_input_fn: Function to extract input from an example
            get_output_fn: Function to extract output from an example
            shuffle: Whether to shuffle examples
            random_seed: Random seed for shuffling
            config_file: Name of the configuration file (relative to dataset directory)
        """
        
        # Initialize the asset manager
        self.asset_manager = AssetManager()
        self.dataset_info = {}
        self.available_tools = {}
        self.tool_documentation = ""
        self.schema_documentation = ""
        
        # Store the functions for later use
        self._load_examples_fn = load_examples_fn
        self._get_input_fn = get_input_fn  
        self._get_output_fn = get_output_fn
        
        # Load and provision assets before calling parent constructor
        self._load_and_provision_assets(dataset_path, config_file)
        
        # Call parent constructor (this will call _load_examples)
        super().__init__(dataset_path, shuffle, random_seed)
    
    def _load_and_provision_assets(self, dataset_path: str, config_file: str):
        """Load dataset configuration and provision assets"""
        
        # Determine config file path
        dataset_dir = Path(dataset_path).parent if Path(dataset_path).is_file() else Path(dataset_path)
        config_path = dataset_dir / config_file
        
        if config_path.exists():
            print(f"Loading dataset configuration from: {config_path}")
            
            try:
                # Load configuration
                self.dataset_info, assets = self.asset_manager.load_dataset_config(str(config_path))
                
                print(f"Dataset: {self.dataset_info['name']}")
                print(f"Description: {self.dataset_info['description']}")
                
                if assets:
                    print(f"Found {len(assets)} assets to provision...")
                    
                    # Provision all assets
                    self.available_tools = self.asset_manager.provision_assets(assets)
                    
                    # Generate tool documentation
                    self.tool_documentation = self.asset_manager.get_tool_documentation()
                    self.schema_documentation = self.asset_manager.get_database_schemas()
                    
                    print("Asset provisioning completed!")
                else:
                    print("No assets defined in configuration")
                    
            except Exception as e:
                print(f"Warning: Failed to load assets: {e}")
                print("Continuing without assets...")
        else:
            print(f"No configuration file found at: {config_path}")
            print("Dataset will run without external assets")
    
    def _load_examples(self):
        """Load examples using the provided function"""
        self.examples = self._load_examples_fn(self.dataset_path)
        
    def get_example_input(self, example: Any) -> Any:
        """Extract input from example with context formatting if assets are available"""
        raw_input = self._get_input_fn(example)
        
        # Add context formatting if we have asset information
        if self.dataset_info and self.available_tools:
            return self._format_question_with_context(raw_input)
        
        # Return raw input if no assets or dataset info
        return raw_input
        
    def get_example_output(self, example: Any) -> Any:  
        """Extract output from example using provided function"""
        return self._get_output_fn(example)
    
    def get_asset_info(self) -> Dict[str, Any]:
        """Get information about loaded assets and available tools"""
        return {
            "dataset_info": self.dataset_info,
            "available_tools": self.available_tools,
            "tool_documentation": self.tool_documentation,
            "schema_documentation": self.schema_documentation,
            "active_servers": list(self.asset_manager.active_servers.keys())
        }
    
    def execute_tool(self, asset_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool on a specific asset"""
        return self.asset_manager.execute_tool(asset_name, tool_name, **kwargs)
    
    def _format_question_with_context(self, raw_question: str) -> str:
        """Format question with role, task, and resource context based on dataset configuration"""
        
        # Extract dataset information
        dataset_name = self.dataset_info.get('name', 'Unknown Dataset')
        dataset_description = self.dataset_info.get('description', '')
        
        # Infer role and task from dataset description and name
        role = self._infer_role_from_dataset()
        task = self._infer_task_from_dataset()
        
        # Generate resource descriptions based on available assets
        resources = self._generate_resource_descriptions()
        
        # Format the complete question with context
        formatted_question = f"""ROLE: {role}

TASK: {task}

RESOURCES AVAILABLE:
{resources}

CUSTOMER SUPPORT TICKET:
{raw_question}

Analyze this customer support ticket and generate a complete resolution plan. Use the available tools to:
1. Look up customer information in the database 
2. Reference relevant company policies
3. Determine appropriate actions and compensation
4. Decide on escalation needs
5. Set appropriate response tone and priority

Generate a structured resolution plan in JSON format."""
        
        return formatted_question
    
    def _infer_role_from_dataset(self) -> str:
        """Infer appropriate role from dataset information"""
        name = self.dataset_info.get('name', '').lower()
        description = self.dataset_info.get('description', '').lower()
        
        if 'customer service' in name or 'customer service' in description or 'support' in description:
            return "You are a customer service agent."
        elif 'legal' in name or 'legal' in description or 'law' in description:
            return "You are a legal researcher."
        elif 'medical' in name or 'medical' in description or 'diagnosis' in description:
            return "You are a medical professional."
        elif 'code' in name or 'programming' in description or 'software' in description:
            return "You are a software engineer."
        else:
            return f"You are working with the {self.dataset_info.get('name', 'dataset')}."
    
    def _infer_task_from_dataset(self) -> str:
        """Infer appropriate task from dataset information"""
        description = self.dataset_info.get('description', '').lower()
        
        if 'ticket' in description or 'support' in description:
            return "Analyze the given scenario and provide appropriate resolution or response."
        elif 'question' in description or 'query' in description:
            return "Answer the question using available resources and information."
        elif 'diagnosis' in description or 'analysis' in description:
            return "Analyze the given information and provide your professional assessment."
        else:
            return "Process the given input and provide an appropriate response."
    
    def _generate_resource_descriptions(self) -> str:
        """Generate descriptions of available resources based on assets"""
        if not self.available_tools:
            return "- No external resources configured for this dataset"
        
        resource_lines = []
        for asset_name in self.available_tools:
            # Get asset info from dataset configuration
            asset_info = None
            if self.dataset_info and 'assets' in self.dataset_info:
                for asset in self.dataset_info['assets']:
                    if asset['name'] == asset_name:
                        asset_info = asset
                        break
            
            if asset_info:
                asset_type = asset_info.get('type', 'unknown')
                asset_desc = asset_info.get('description', f'{asset_name} resource')
                
                if asset_type == 'sqlite':
                    resource_lines.append(f"- {asset_desc}")
                    resource_lines.append(f"  (Use database query tools to search and retrieve information)")
                elif asset_type == 'filesystem':
                    resource_lines.append(f"- {asset_desc}")
                    resource_lines.append(f"  (Use file search and read tools to access documents)")
                else:
                    resource_lines.append(f"- {asset_desc}")
                    resource_lines.append(f"  (Use appropriate tools to access this resource)")
            else:
                resource_lines.append(f"- {asset_name} (available via tools)")
        
        return '\n'.join(resource_lines) if resource_lines else "- No external resources configured"
    
    def cleanup(self):
        """Clean up asset manager and servers"""
        self.asset_manager.cleanup()

def create_customer_service_loader(dataset_path: str, **kwargs) -> AssetDatasetLoader:
    """Create a loader specifically for customer service datasets"""
    
    def load_examples(path: str) -> List[Dict]:
        """Load customer service evaluation examples"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert to list of examples
        examples = []
        for ticket_id, ticket_data in data.items():
            example = {
                "id": ticket_id,
                "question": ticket_data["question"],
                "answer": ticket_data["answer"]
            }
            examples.append(example)
        
        return examples
    
    def get_input(example: Dict) -> str:
        """Extract input (question) from example with context formatting"""
        return example["question"]  # Will be enhanced by the loader after instantiation
    
    def get_output(example: Dict) -> str:
        """Extract output (answer) from example"""
        return example["answer"]
    
    return AssetDatasetLoader(
        dataset_path=dataset_path,
        load_examples_fn=load_examples,
        get_input_fn=get_input,
        get_output_fn=get_output,
        **kwargs
    )

def create_asset_loader(dataset_path: str, 
                       loader_type: str = "json",
                       input_field: str = "question",
                       output_field: str = "answer",
                       **kwargs) -> AssetDatasetLoader:
    """Create a generic asset-enabled loader for various dataset types"""
    
    def load_examples(path: str) -> List[Dict]:
        """Load examples based on loader type"""
        if loader_type == "json":
            with open(path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else list(data.values())
        
        elif loader_type == "jsonl":
            examples = []
            with open(path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            return examples
        
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
    
    def get_input(example: Dict) -> str:
        """Extract input field from example"""
        return example.get(input_field, "")
    
    def get_output(example: Dict) -> str:
        """Extract output field from example"""
        return example.get(output_field, "")
    
    return AssetDatasetLoader(
        dataset_path=dataset_path,
        load_examples_fn=load_examples,
        get_input_fn=get_input,
        get_output_fn=get_output,
        **kwargs
    ) 