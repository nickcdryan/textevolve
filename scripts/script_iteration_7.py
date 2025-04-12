import os

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

def schedule_meeting(question):
    """Schedules a meeting based on the provided question and constraints."""

    # Agent 1: Constraint Extractor & Summarizer
    def extract_and_summarize_constraints(problem):
        system_instruction = "You are an expert at extracting and summarizing constraints for scheduling meetings. Focus on participant availability, meeting duration, and time preferences."
        prompt = f"Extract and summarize the key constraints from this scheduling problem: {problem}"
        return call_llm(prompt, system_instruction)

    # Agent 2: Time Slot Generator
    def generate_time_slots(constraints_summary):
        system_instruction = "You are a creative meeting scheduler who can generate plausible time slots that satisfy the given constraints. Suggest only a *single*, specific time slot."
        prompt = f"Generate ONE possible meeting time slot based on these constraints: {constraints_summary}"
        return call_llm(prompt, system_instruction)

    # Agent 3: Time Slot Checker
    def verify_time_slot(proposed_time, original_problem):
        system_instruction = "You are a rigorous time slot checker. Carefully verify if the proposed time slot satisfies all constraints in the original problem."
        prompt = f"Carefully verify if the proposed time '{proposed_time}' is valid given all constraints in the following scheduling problem: {original_problem}. Explain your reasoning step by step and state definitively whether the time slot is valid or invalid."
        return call_llm(prompt, system_instruction)
    
    # Agent 4: Earliest Availability Checker (New!)
    def check_earliest_availability(valid_time, all_schedules):
        system_instruction = "You are an earliest availability expert. Given a meeting time, make absolutely sure that it's the earliest possible one considering all the other schedules. Check calculations meticulously."
        prompt = f"Given a valid meeting time: '{valid_time}' and considering all the schedules: '{all_schedules}', is this the earliest possible time? Debate your calculation for 10 rounds to make sure you're right."
        return call_llm(prompt, system_instruction)

    try:
        # 1. Extract and Summarize Constraints
        constraints_summary = extract_and_summarize_constraints(question)

        # 2. Generate a Potential Time Slot
        proposed_time = generate_time_slots(constraints_summary)

        # 3. Verify the Time Slot
        verification_result = verify_time_slot(proposed_time, question)

        if "valid" in verification_result.lower():
            #4. Check Earliest Availability
            earliest_check = check_earliest_availability(proposed_time, question)
            
            if "earliest" in earliest_check.lower():
              return f"Here is the proposed time: {proposed_time}"
            else:
              return "Error: Time is not earliest possible. Further processing needed."
        else:
            return "Error: Proposed time is invalid."

    except Exception as e:
        return f"An error occurred: {str(e)}"

def main(question):
    """Main function to schedule a meeting."""
    return schedule_meeting(question)