import os
import json
import re
import math

def main(question):
    """
    Orchestrates meeting scheduling using a dynamic ReAct-based approach, 
    focusing on iterative extraction, constraint application, and verification with targeted refinements.
    """
    try:
        # Initialize the ReAct agent
        react_agent = MeetingSchedulingAgent()
        solution = react_agent.solve(question)
        return solution
    except Exception as e:
        return f"Error in main: {str(e)}"

class MeetingSchedulingAgent:
    """
    An agent that uses the ReAct pattern to schedule meetings.
    """
    def __init__(self):
        self.system_instruction = "You are an expert meeting scheduling agent."

    def solve(self, problem, max_iterations=5):
        """Solves the meeting scheduling problem using ReAct."""
        thought = "Let's start by extracting all key information from the problem description."
        actions = []
        for i in range(max_iterations):
            prompt = f"""
            Problem: {problem}
            {thought}
            Possible actions: ExtractInformation, GenerateSchedule, VerifySchedule, Finish
            What is your next thought and what action will you take?
            
            Example 1:
            Problem: You need to schedule a meeting for John and Jane for 30 minutes on Monday. John is busy 1-2pm. Jane prefers to meet before noon.
            Thought: First, I need to extract the key information like participants, duration and constraints.
            Action: ExtractInformation[question="{problem}"]

            Example 2:
            Problem: Extracted info: {{"participants": ["John", "Jane"], "duration": "30 minutes", "available_days": ["Monday"], "time_constraints": "John is busy 1-2pm. Jane prefers to meet before noon."}}
            Thought: Now that I have the information, I can create a candidate meeting schedule.
            Action: GenerateSchedule[extracted_info="{{\\"participants\\": [\\"John\\", \\"Jane\\"], \\"duration\\": \\"30 minutes\\", \\"available_days\\": [\\"Monday\\"], \\"time_constraints\\": \\"John is busy 1-2pm. Jane prefers to meet before noon.\\" }}""]
            """
            response = call_llm(prompt, self.system_instruction)
            try:
                thought, action = response.split("Action: ")
                thought = "Thought: " + thought.split("Thought: ")[-1].strip()
                action_type, action_details = action.split("[")
                action_details = action_details[:-1]
                actions.append({"type": action_type.strip(), "details": action_details.strip()})
                # Handle actions
                if action_type.strip() == "ExtractInformation":
                    extracted_info = self._extract_information(action_details)
                    thought = "Thought: I have extracted the information. Now I can generate a schedule."
                elif action_type.strip() == "GenerateSchedule":
                    schedule = self._generate_schedule(action_details)
                    thought = "Thought: I have generated a schedule. I need to verify it."
                elif action_type.strip() == "VerifySchedule":
                    verification_result = self._verify_schedule(action_details)
                    if verification_result == "Valid":
                        return schedule
                    else:
                        thought = f"Thought: The schedule is invalid. I need to generate a new schedule. {verification_result}"
                        schedule = self._generate_schedule(extracted_info)
                elif action_type.strip() == "Finish":
                    return action_details
            except Exception as e:
                return f"Error processing ReAct step: {str(e)}"
        return "Could not find a valid solution."

    def _extract_information(self, details):
        """Extracts meeting information with a specific schema and examples."""
        system_instruction = "You are an expert at extracting structured information from text."
        prompt = f"""
        Extract the following information from the text: participants, duration, available days, time constraints. Return a JSON.
        
        Example:
        Input: Schedule a meeting for John, Jane, and Peter for 1 hour on Monday or Tuesday between 9am and 5pm. John is busy from 10am-11am on Monday. Jane is unavailable from 2pm-3pm on Tuesday.
        Output:
        {{
          "participants": ["John", "Jane", "Peter"],
          "duration": "1 hour",
          "available_days": ["Monday", "Tuesday"],
          "time_constraints": "between 9am and 5pm. John is busy from 10am-11am on Monday. Jane is unavailable from 2pm-3pm on Tuesday."
        }}
        Now extract the same from:
        {details}
        """
        return call_llm(prompt, system_instruction)

    def _generate_schedule(self, extracted_info):
        """Generates a candidate schedule based on extracted information."""
        system_instruction = "You are an expert meeting scheduler."
        prompt = f"""
        Generate a candidate meeting schedule that satisfies the constraints in JSON.
        
        Example:
        Input:
        {{
          "participants": ["John", "Jane"],
          "duration": "30 minutes",
          "available_days": ["Monday"],
          "time_constraints": "John is busy from 10am-11am."
        }}
        Output: Monday, 9:00 - 9:30
        Now generate the same from:
        {extracted_info}
        """
        return call_llm(prompt, system_instruction)

    def _verify_schedule(self, details):
        """Verifies the candidate schedule against the extracted information."""
        system_instruction = "You are a meeting schedule validator."
        prompt = f"""
        Verify the following schedule is valid based on extracted information, return "Valid" or "Invalid" with reason.
        
        Example:
        Extracted Information:
        {{
          "participants": ["John", "Jane"],
          "duration": "30 minutes",
          "available_days": ["Monday"],
          "time_constraints": "John is busy from 10am-11am."
        }}
        Candidate schedule: Monday, 10:30 - 11:00
        Output: Invalid, John is busy.
        
        Now verify the same from:
        {details}
        """
        return call_llm(prompt, system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response"""
    try:
        from google import genai
        from google.genai import types

        # Initialize the Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        # Call the API with system instruction if provided
        if system_instruction:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                ),
                contents=prompt
            )
        else:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"