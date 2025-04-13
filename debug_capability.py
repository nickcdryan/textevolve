
#!/usr/bin/env python
"""
debug_capability.py - Detailed debugging for the capability tracking system
"""

import os
import json
import traceback
from pathlib import Path
from agent_system import AgentSystem, CapabilityTracker

def print_banner(text):
    """Print a banner with the given text."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80 + "\n")

def print_dict(d, indent=0):
    """Pretty print a dictionary."""
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + f"{key}:")
            if value:
                for item in value:
                    if isinstance(item, dict):
                        print_dict(item, indent + 1)
                    else:
                        print("  " * (indent + 1) + f"- {item}")
            else:
                print("  " * (indent + 1) + "[]")
        else:
            print("  " * indent + f"{key}: {value}")

class DebuggingCapabilityTracker(CapabilityTracker):
    """Enhanced CapabilityTracker with detailed logging for debugging."""
    
    def __init__(self):
        print_banner("Initializing DebuggingCapabilityTracker")
        super().__init__()
        print("Core capabilities:", self.capabilities)
        print("Initial history:", self.history)
        print("Initial assessment:", self.current_assessment)
    
    def update_from_evaluations(self, evaluations, capability_insights, error_analysis):
        """Add detailed logging to track data flow in update_from_evaluations."""
        print_banner("DEBUG: update_from_evaluations CALLED")
        
        # Log inputs
        print(f"Number of evaluations: {len(evaluations)}")
        print(f"Has capability_insights: {bool(capability_insights)}")
        print(f"Has error_analysis: {bool(error_analysis)}")
        
        if error_analysis:
            print("\nError Analysis Keys:", list(error_analysis.keys()))
            
            for key in ['strengths', 'weaknesses', 'bottlenecks', 'primary_issue', 'improvement_suggestions', 
                       'improvement_areas', 'capability_mapping']:
                if key in error_analysis:
                    print(f"\n{key.upper()}:")
                    if isinstance(error_analysis[key], dict):
                        print_dict(error_analysis[key])
                    elif isinstance(error_analysis[key], list):
                        for item in error_analysis[key]:
                            print(f"  - {item}")
                    else:
                        print(f"  {error_analysis[key]}")
                else:
                    print(f"\n{key.upper()}: Not found in error_analysis")
        
        # Log evaluation structure
        if evaluations:
            print("\nSample evaluation structure:")
            sample_eval = evaluations[0]
            print_dict(sample_eval)
            
            # Check for capability failures
            failures_present = any("capability_failures" in eval_data for eval_data in evaluations)
            print(f"\nCapability failures present in evaluations: {failures_present}")
            
            if failures_present:
                for i, eval_data in enumerate(evaluations):
                    if "capability_failures" in eval_data:
                        print(f"Evaluation {i} has capability failures: {eval_data.get('capability_failures', [])}")
        
        # Call original method and capture result
        try:
            print("\nCalling original update_from_evaluations method...")
            result = super().update_from_evaluations(evaluations, capability_insights, error_analysis)
            print("\nMethod completed successfully")
        except Exception as e:
            print(f"\nEXCEPTION in update_from_evaluations: {e}")
            print(traceback.format_exc())
            # Create a minimal result to avoid further errors
            result = {
                "strengths": ["Error in update_from_evaluations"],
                "weaknesses": [f"Exception: {str(e)}"],
                "bottlenecks": ["Error processing data"],
                "improvement_areas": ["error_handling"],
                "improvement_suggestions": ["Fix exception in capability tracking"]
            }
        
        # Log the result
        print("\nResulting assessment:")
        print_dict(self.current_assessment)
        
        return result
    
    def generate_report(self):
        """Add detailed logging to track data flow in generate_report."""
        print_banner("DEBUG: generate_report CALLED")
        
        print("Current assessment before report generation:")
        print_dict(self.current_assessment)
        
        try:
            improvement_focus = self.get_improvement_focus()
            print(f"\nImprovement focus: {improvement_focus}")
        except Exception as e:
            print(f"\nEXCEPTION getting improvement focus: {e}")
            print(traceback.format_exc())
        
        try:
            print("\nCalling original generate_report method...")
            report = super().generate_report()
            print("\nReport generated successfully")
        except Exception as e:
            print(f"\nEXCEPTION in generate_report: {e}")
            print(traceback.format_exc())
            # Create a minimal report to avoid further errors
            report = {
                "strengths": ["Error in generate_report"],
                "weaknesses": [f"Exception: {str(e)}"],
                "bottlenecks": ["Error generating report"],
                "improvement_focus": "error_handling",
                "trend": "error"
            }
        
        print("\nFinal report content:")
        print_dict(report)
        
        return report
    
    def get_improvement_focus(self):
        """Add detailed logging to track data flow in get_improvement_focus."""
        print_banner("DEBUG: get_improvement_focus CALLED")
        
        print("Current assessment for determining improvement focus:")
        print(f"improvement_areas: {self.current_assessment.get('improvement_areas', [])}")
        print(f"weaknesses: {self.current_assessment.get('weaknesses', [])}")
        
        try:
            print("\nCalling original get_improvement_focus method...")
            focus = super().get_improvement_focus()
            print(f"\nImprovement focus determined: {focus}")
            return focus
        except Exception as e:
            print(f"\nEXCEPTION in get_improvement_focus: {e}")
            print(traceback.format_exc())
            return "error_handling"  # Fallback
    
    def _analyze_trend(self):
        """Add detailed logging to track data flow in _analyze_trend."""
        print_banner("DEBUG: _analyze_trend CALLED")
        
        print(f"History length: {len(self.history)}")
        
        if len(self.history) < 2:
            print("Insufficient history for trend analysis")
            return "insufficient_data"
        
        try:
            print("\nCalling original _analyze_trend method...")
            trend = super()._analyze_trend()
            print("\nTrend analysis completed successfully")
            print_dict(trend)
            return trend
        except Exception as e:
            print(f"\nEXCEPTION in _analyze_trend: {e}")
            print(traceback.format_exc())
            return "error"  # Fallback

def patch_agent_system():
    """Monkey patch the AgentSystem to use our debugging tracker."""
    print_banner("Patching AgentSystem to use DebuggingCapabilityTracker")
    
    original_init = AgentSystem.__init__
    
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        print("Replacing standard CapabilityTracker with DebuggingCapabilityTracker")
        self.capability_tracker = DebuggingCapabilityTracker()
    
    AgentSystem.__init__ = patched_init
    print("Patching complete")

def inspect_error_analysis_in_archive():
    """Examine archived iterations for error analysis data."""
    print_banner("Inspecting Error Analysis in Archive")
    
    archive_dir = Path("archive")
    if not archive_dir.exists():
        print("Archive directory not found")
        return
    
    for file in archive_dir.glob("iteration_*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            iteration = data.get("iteration", "unknown")
            print(f"\nFile: {file.name} (Iteration {iteration})")
            
            if "performance" in data and "error_analysis" in data["performance"]:
                error_analysis = data["performance"]["error_analysis"]
                
                print("Error Analysis Keys:", list(error_analysis.keys()))
                
                # Check for key fields
                for field in ['strengths', 'weaknesses', 'bottlenecks', 'primary_issue', 
                             'improvement_suggestions', 'capability_mapping']:
                    if field in error_analysis:
                        print(f"\n{field.upper()}:")
                        if isinstance(error_analysis[field], dict):
                            print_dict(error_analysis[field])
                        elif isinstance(error_analysis[field], list):
                            for item in error_analysis[field]:
                                print(f"  - {item}")
                        else:
                            print(f"  {error_analysis[field]}")
                    else:
                        print(f"\n{field.upper()}: Not found")
                
                # Check for any errors in the error analysis
                if "root_causes" in error_analysis:
                    print("\nROOT CAUSES:")
                    for cause in error_analysis["root_causes"]:
                        print(f"  - {cause}")
            else:
                print("No error analysis found in this iteration")
            
            # Check if capability report exists
            if "capability_report" in data:
                print("\nCapability Report:")
                if data["capability_report"]:
                    print_dict(data["capability_report"])
                else:
                    print("  Empty report")
            else:
                print("\nNo capability report found")
            
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

def main():
    """Main function to run the debugging utility."""
    print_banner("CAPABILITY TRACKING DEBUGGING UTILITY")
    
    # Examine archive files for clues
    inspect_error_analysis_in_archive()
    
    # Patch the agent system for future runs
    patch_agent_system()
    
    print_banner("DEBUGGING SETUP COMPLETE")
    print("To use this debugging, import and run the patch_agent_system function before creating an agent:")
    print("\nfrom debug_capability import patch_agent_system")
    print("patch_agent_system()")
    print("agent = AgentSystem(...)")
    
    print("\nOr modify run_script.py to use this utility directly.")

if __name__ == "__main__":
    main()
