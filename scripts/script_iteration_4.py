import os
import re

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

def main(question):
    """Main function to schedule meetings based on constraints, using a ReAct approach with verification."""

    def schedule_react_agent(task, max_iterations=5):
        """Solve scheduling problems using ReAct."""
        system_instruction = "You are a scheduling expert using ReAct to find meeting times."
        context = f"""
        You are an expert scheduler. Use ReAct to find the best meeting time.
        Example:
        Problem: Schedule a meeting for John and Mary for 30 minutes on Monday between 9am and 5pm. John is busy 10am-11am. Mary is busy 2pm-3pm.
        Thought 1: I need to understand the constraints.
        Action 1: Parse constraints.
        Observation 1: Participants: John, Mary. Duration: 30 minutes. Day: Monday. Time: 9am-5pm. John busy: 10am-11am. Mary busy: 2pm-3pm.
        Thought 2: I need to find a free slot for both.
        Action 2: Find available slots.
        Observation 2: Available slot: 9:00-9:30.
        Thought 3: I have a potential solution.
        Action 3: Verify solution.
        Observation 3: The solution is valid.
        Thought 4: I am done.
        Action 4: Finish[Monday 9:00-9:30]

        Now solve this:
        {task}
        """
        for i in range(max_iterations):
            prompt = context + f"\nThought {i+1}:"
            response = call_llm(prompt, system_instruction)
            context += response

            if "Finish[" in response:
                final_answer = response.split("Finish[")[1].split("]")[0]
                return "Here is the proposed time: " + final_answer
            if "Error:" in response:
                return "Error finding a solution."
        return "Could not find a solution within the iteration limit."

    # Call the ReAct agent to find the solution.
    return schedule_react_agent(question)