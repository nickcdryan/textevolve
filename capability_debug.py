
import os
import json
from pathlib import Path
from agent_system import AgentSystem, CapabilityTracker

def print_section(title):
    print("\n" + "=" * 40)
    print(f"  {title}  ")
    print("=" * 40)

def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + f"{key}: {value}")
        else:
            print("  " * indent + f"{key}: {value}")

# Enhanced version of CapabilityTracker with additional logging
class DebugCapabilityTracker(CapabilityTracker):
    def update_from_evaluations(self, evaluations, capability_insights, error_analysis):
        print_section("DEBUG: update_from_evaluations CALLED")
        print(f"Number of evaluations: {len(evaluations)}")
        print(f"Error analysis keys: {error_analysis.keys() if error_analysis else 'None'}")
        
        if error_analysis:
            print("\nError Analysis Content Preview:")
            for key in ['strengths', 'weaknesses', 'bottlenecks', 'improvement_areas', 'improvement_suggestions']:
                if key in error_analysis:
                    print(f"  {key}: {error_analysis[key]}")
                else:
                    print(f"  {key}: Not found in error_analysis")
        
        # Call the original method and capture its result
        result = super().update_from_evaluations(evaluations, capability_insights, error_analysis)
        
        print("\nAssessment after update:")
        print_dict(self.current_assessment)
        
        return result
    
    def generate_report(self):
        print_section("DEBUG: generate_report CALLED")
        report = super().generate_report()
        print("Report generated:")
        print_dict(report)
        return report

# Initialize agent with the debug tracker
def initialize_debug_agent(dataset_path="calendar_scheduling.json", example_prefix="calendar_scheduling_example_"):
    # Create an agent system
    agent = AgentSystem(dataset_path=dataset_path, example_prefix=example_prefix)
    
    # Replace the capability tracker with our debug version
    agent.capability_tracker = DebugCapabilityTracker()
    
    print_section("AGENT INITIALIZED WITH DEBUG CAPABILITY TRACKER")
    
    return agent

def inspect_archive_files():
    print_section("INSPECTING ARCHIVE FILES")
    archive_dir = Path("archive")
    
    # Check if directory exists
    if not archive_dir.exists():
        print("Archive directory doesn't exist.")
        return
    
    # Look for iterations with capability reports
    for file in archive_dir.glob("iteration_*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if "capability_report" in data:
                print(f"\nFile: {file.name}")
                print("Capability Report:")
                if data["capability_report"]:
                    print_dict(data["capability_report"])
                else:
                    print("  Empty report")
                
                # Check error analysis
                if "performance" in data and "error_analysis" in data["performance"]:
                    print("\nError Analysis in this iteration:")
                    error_analysis = data["performance"]["error_analysis"]
                    for key in ['strengths', 'weaknesses', 'primary_issue']:
                        if key in error_analysis:
                            print(f"  {key}: {error_analysis[key]}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

def main():
    print_section("CAPABILITY TRACKER DEBUG UTILITY")
    
    # Inspect existing archive files
    inspect_archive_files()
    
    # Set the API key for testing (assuming this is already in environment)
    if not os.environ.get("GEMINI_API_KEY"):
        print("\nWarning: GEMINI_API_KEY not set in environment. Set it before running actual tests.")
    
    print("\nThis script has analyzed existing archives. To run a test iteration with debugging:")
    print("1. Add 'from capability_debug import initialize_debug_agent' to run_script.py")
    print("2. Replace 'agent = AgentSystem(...)' with 'agent = initialize_debug_agent(...)'")
    print("3. Run a test iteration with the debugged capability tracker")

if __name__ == "__main__":
    main()
