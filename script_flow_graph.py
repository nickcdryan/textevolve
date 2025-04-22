#!/usr/bin/env python
"""
simplified_flow_graph.py - Generate simplified high-level computation flow graphs

This script creates a clean, simplified visualization of the main computation flow
of a Python script, focusing only on the key functional components.

Usage:
    python simplified_flow_graph.py <path_to_script> [options]

Options:
    --no-llm         : Don't use LLM for function descriptions
    --output=PATH    : Specify output path (without extension)
    --node-size=N    : Set node size (default: 6000)
    --font-size=N    : Set font size (default: 14)
"""

import ast
import sys
import os
import subprocess
import importlib.util
import re

def check_and_install_dependencies():
    """Check if required dependencies are installed, and install them if needed"""
    required_packages = ['networkx', 'matplotlib']

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            print(f"Package '{package}' is not installed. Attempting to install...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install it manually: pip install {package}")
                return False

    return True

def call_llm(prompt):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai

        # Check for API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY environment variable not set.")
            print("Please set it with: export GEMINI_API_KEY=your_key_here")
            return "No API key available."

        # Initialize the Gemini client
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text
    except ImportError:
        print("Google Generative AI package not installed.")
        print("To install: pip install google-generativeai")
        return "Unable to call LLM: google-generativeai package not installed."
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"

class MainFlowAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze the main function and its flow"""
    def __init__(self):
        self.functions = {}  # Dictionary mapping function names to their docstrings
        self.main_flow = []  # List of function calls in the main function
        self.branches = []   # List of conditional branches
        self.current_function = None
        self.in_main = False
        self.utility_functions = {"print", "str", "int", "float", "bool", "list", "dict", "set", "tuple"}

    def visit_FunctionDef(self, node):
        # Extract docstring if available
        docstring = ast.get_docstring(node)
        self.functions[node.name] = docstring

        # Check if this is the main function
        if node.name == "main":
            self.in_main = True
            self.generic_visit(node)
            self.in_main = False
        else:
            prev_function = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = prev_function

    def visit_Call(self, node):
        # Only track calls in the main function
        if self.in_main:
            if isinstance(node.func, ast.Name):
                # Skip utility functions
                if node.func.id not in self.utility_functions:
                    self.main_flow.append(node.func.id)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                # Skip utility functions and attributes
                if node.func.value.id not in self.utility_functions:
                    self.main_flow.append(f"{node.func.value.id}.{node.func.attr}")

        self.generic_visit(node)

    def visit_If(self, node):
        # Track conditional branches in main
        if self.in_main:
            # Extract the condition for the branch label
            condition = ast.unparse(node.test).strip()

            # Extract function calls in the true branch
            true_branch_calls = []
            for item in node.body:
                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                    if isinstance(item.value.func, ast.Name):
                        if item.value.func.id not in self.utility_functions:
                            true_branch_calls.append(item.value.func.id)
                elif isinstance(item, ast.Return) and isinstance(item.value, ast.Call):
                    if isinstance(item.value.func, ast.Name):
                        if item.value.func.id not in self.utility_functions:
                            true_branch_calls.append(item.value.func.id)

            # Extract function calls in the false branch
            false_branch_calls = []
            for item in node.orelse:
                if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                    if isinstance(item.value.func, ast.Name):
                        if item.value.func.id not in self.utility_functions:
                            false_branch_calls.append(item.value.func.id)
                elif isinstance(item, ast.Return) and isinstance(item.value, ast.Call):
                    if isinstance(item.value.func, ast.Name):
                        if item.value.func.id not in self.utility_functions:
                            false_branch_calls.append(item.value.func.id)

            # Record the branch
            if true_branch_calls or false_branch_calls:
                self.branches.append({
                    'condition': condition,
                    'true_branch': true_branch_calls,
                    'false_branch': false_branch_calls
                })

        self.generic_visit(node)

    def get_simplified_flow(self):
        """Return a simplified flow by removing duplicates while preserving order"""
        result = []
        seen = set()

        for func in self.main_flow:
            if func not in seen:
                result.append(func)
                seen.add(func)

        return result

def get_function_descriptions(main_flow, functions, code):
    """Use LLM to get clear, concise descriptions for functions in the main flow"""
    # Collect functions to describe
    funcs_to_describe = [func for func in main_flow if func in functions]

    if not funcs_to_describe:
        return {}

    # Prepare LLM prompt
    prompt = f"""
    For the following Python functions, provide a clear, concise description (5-7 words maximum) of what each function does.

    ```python
    {code}
    ```

    Provide very short descriptions (5-7 words max) for these functions:
    {', '.join(funcs_to_describe)}

    Format your response as:
    function_name1: Very short description
    function_name2: Very short description
    ...
    """

    # Call LLM
    response = call_llm(prompt)

    # Parse LLM response
    descriptions = {}
    for line in response.strip().split("\n"):
        if ":" in line:
            parts = line.split(":", 1)
            func_name = parts[0].strip()
            description = parts[1].strip()
            if func_name in functions:
                descriptions[func_name] = description

    return descriptions

def get_function_category(func_name):
    """Categorize function based on its name"""
    if func_name == 'main':
        return 'main'
    elif any(term in func_name.lower() for term in ["llm", "call_"]):
        return 'llm'
    elif any(term in func_name.lower() for term in ["extract", "parse", "analyze"]):
        return 'extract'
    elif any(term in func_name.lower() for term in ["verify", "validate", "check"]):
        return 'verify'
    elif any(term in func_name.lower() for term in ["format", "output", "response"]):
        return 'format'
    elif any(term in func_name.lower() for term in ["find", "get", "fetch", "search"]):
        return 'find'
    else:
        return 'other'

def simplify_condition(condition):
    """Simplify a condition for display in the graph"""
    # Replace common patterns with simpler text
    condition = re.sub(r'^\s*"([^"]+)"\s+in\s+(\w+)\s*$', r'if \2 contains "\1"', condition)
    condition = re.sub(r'^\s*(\w+)\s+==\s+"([^"]+)"\s*$', r'if \1 is "\2"', condition)
    condition = re.sub(r'^\s*(\w+)\s+!=\s+"([^"]+)"\s*$', r'if \1 is not "\2"', condition)

    # Keep it short
    if len(condition) > 20:
        condition = condition[:17] + "..."

    return condition

def generate_simplified_flow_graph(script_path, output_path=None, use_llm=True, 
                                  node_size=6000, font_size=14):
    """Generate a simplified high-level flow graph for a Python script"""
    # Import required libraries
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        print("Required packages not installed. Please install them with:")
        print("pip install networkx matplotlib")
        return None

    # Read the script
    with open(script_path, 'r') as f:
        code = f.read()

    # Parse the AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in script: {e}")
        return None

    # Extract main function flow
    analyzer = MainFlowAnalyzer()
    analyzer.visit(tree)

    # Get simplified flow
    main_flow = analyzer.get_simplified_flow()
    branches = analyzer.branches

    if not main_flow:
        print("Could not extract main computation flow. Make sure the script has a 'main' function.")
        return None

    print(f"Extracted flow: {' -> '.join(main_flow)}")
    print(f"Found {len(branches)} branches in the flow.")

    # Get function descriptions
    descriptions = {}
    if use_llm:
        print("Getting function descriptions from LLM...")
        descriptions = get_function_descriptions(main_flow, analyzer.functions, code)

    # Create a directed graph
    G = nx.DiGraph()

    # Color map for different function categories
    color_map = {
        'main': 'gold',
        'llm': 'lightgreen',
        'extract': 'lightblue',
        'verify': 'lightsalmon',
        'format': 'lightcyan',
        'find': 'lavender',
        'other': 'lightgrey',
        'branch': 'white'
    }

    # Add a start node
    G.add_node("input", 
               label="question", 
               category="other",
               color="lightgrey",
               shape="ellipse")

    # Add nodes for each function in the main flow
    prev_node = "input"
    for func in main_flow:
        # Skip if already added
        if func in G.nodes:
            continue

        # Get description and label
        desc = descriptions.get(func, "")
        label = f"{func}"
        if desc:
            label = f"{func}\n{desc}"

        # Get category
        category = get_function_category(func)

        # Add the node
        G.add_node(func, 
                  label=label, 
                  category=category,
                  color=color_map[category],
                  shape="box")

        # Connect to previous node in the flow
        G.add_edge(prev_node, func)
        prev_node = func

    # Add a terminal node
    G.add_node("output", 
               label="answer", 
               category="other",
               color="lightgrey",
               shape="ellipse")

    # Find the last real function node
    last_func = main_flow[-1] if main_flow else None

    # Process branches if any
    if branches:
        for i, branch in enumerate(branches):
            condition = simplify_condition(branch['condition'])
            branch_id = f"branch_{i}"

            # Find the function before this branch
            # This is a simplification - in a real case we'd need to determine
            # where exactly this branch occurs in the flow

            # Add a decision node
            G.add_node(branch_id,
                      label=condition,
                      category="branch",
                      color=color_map["branch"],
                      shape="diamond")

            # Try to figure out where this branch should go in the flow
            # Simplified approach: look for function calls that match
            branch_point = None
            for i, func in enumerate(main_flow):
                # If we find a function that's in either branch, assume the branch
                # happens after the previous function
                if (func in branch['true_branch'] or 
                    func in branch['false_branch']) and i > 0:
                    branch_point = main_flow[i-1]
                    break

            # If we couldn't figure it out, put it after the last function in the flow
            if not branch_point:
                branch_point = last_func

            if branch_point and branch_point in G.nodes:
                # Connect the branch node after the branch point
                for succ in list(G.successors(branch_point)):
                    G.remove_edge(branch_point, succ)
                    G.add_edge(branch_id, succ)
                G.add_edge(branch_point, branch_id)
    else:
        # No branches - connect the last function to output
        if last_func:
            G.add_edge(last_func, "output")

    # Generate the output file name if not provided
    if output_path is None:
        script_name = os.path.basename(script_path).split('.')[0]
        output_path = f"viz/{script_name}_viz"

    # Set up the plot
    plt.figure(figsize=(18, 8))

    # Create a horizontal layout
    pos = nx.spring_layout(G, k=0.9)

    # Adjust positions to be more horizontal
    # Find input and output node positions
    max_x = max(pos.values(), key=lambda p: p[0])[0]
    min_x = min(pos.values(), key=lambda p: p[0])[0]
    range_x = max_x - min_x

    if "input" in pos and "output" in pos:
        # Fix input and output positions at the left and right
        pos["input"] = (min_x - range_x * 0.1, 0)
        pos["output"] = (max_x + range_x * 0.1, 0)

        # Spread other nodes evenly between input and output
        nodes = [n for n in G.nodes if n not in ["input", "output"]]
        if nodes:
            step = range_x * 1.2 / (len(nodes) + 1)
            for i, node in enumerate(nodes):
                x = min_x + (i + 1) * step
                y = pos[node][1] * 0.2  # Reduce y variation
                pos[node] = (x, y)

    # Draw nodes with different shapes and colors
    for shape in ['box', 'ellipse', 'diamond']:
        nodelist = [n for n, data in G.nodes(data=True) if data.get('shape') == shape]
        if not nodelist:
            continue

        node_colors = [G.nodes[n]['color'] for n in nodelist]

        if shape == 'box':
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=nodelist,
                                  node_color=node_colors,
                                  node_size=node_size,
                                  alpha=0.8,
                                  edgecolors='black',
                                  node_shape='s')  # Square shape
        elif shape == 'ellipse':
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=nodelist,
                                  node_color=node_colors,
                                  node_size=node_size,
                                  alpha=0.8,
                                  edgecolors='black',
                                  node_shape='o')  # Circle shape
        elif shape == 'diamond':
            nx.draw_networkx_nodes(G, pos, 
                                  nodelist=nodelist,
                                  node_color=node_colors,
                                  node_size=node_size,
                                  alpha=0.8,
                                  edgecolors='black',
                                  node_shape='d')  # Diamond shape

    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                         width=2.0,
                         alpha=1.0,
                         arrowsize=30,
                         node_size=node_size,
                         arrowstyle='->')

    # Add labels to nodes
    labels = {}
    for node, data in G.nodes(data=True):
        if 'label' in data:
            labels[node] = data['label']
        else:
            labels[node] = node

    # Draw labels with larger font
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_family='sans-serif')

    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Extraction/Parsing',
                  markerfacecolor='lightblue', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Find/Search',
                  markerfacecolor='lavender', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Verification',
                  markerfacecolor='lightsalmon', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='Formatting/Output',
                  markerfacecolor='lightcyan', markersize=15),
        plt.Line2D([0], [0], marker='s', color='w', label='LLM Functions',
                  markerfacecolor='lightgreen', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Input/Output',
                  markerfacecolor='lightgrey', markersize=15),
        plt.Line2D([0], [0], marker='d', color='w', label='Decision Branch',
                  markerfacecolor='white', markersize=15, markeredgecolor='black'),
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=4)

    plt.axis('off')
    plt.tight_layout()

    # Save the plot
    try:
        output_file = f"{output_path}.png"
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
        print(f"Graph saved to {output_file}")
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None

    return output_path

def main():
    # Check dependencies
    if not check_and_install_dependencies():
        print("Required dependencies are missing. Please install them manually.")
        return

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python simplified_flow_graph.py <path_to_script> [options]")
        print("Options:")
        print("  --no-llm      : Don't use LLM for function descriptions")
        print("  --output=PATH : Specify output path (without extension)")
        print("  --node-size=N : Set node size (default: 6000)")
        print("  --font-size=N : Set font size (default: 14)")
        return

    script_path = sys.argv[1]

    # Parse options
    use_llm = "--no-llm" not in sys.argv

    output_path = None
    for arg in sys.argv:
        if arg.startswith("--output="):
            output_path = arg.split("=")[1]

    node_size = 6000  # default
    for arg in sys.argv:
        if arg.startswith("--node-size="):
            try:
                node_size = int(arg.split("=")[1])
            except ValueError:
                print(f"Invalid node size: {arg}. Using default: 6000")

    font_size = 14  # default
    for arg in sys.argv:
        if arg.startswith("--font-size="):
            try:
                font_size = int(arg.split("=")[1])
            except ValueError:
                print(f"Invalid font size: {arg}. Using default: 14")

    # Check if the script file exists
    if not os.path.isfile(script_path):
        print(f"Error: Script file '{script_path}' not found.")
        return

    try:
        generate_simplified_flow_graph(
            script_path, 
            output_path, 
            use_llm,
            node_size,
            font_size
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()