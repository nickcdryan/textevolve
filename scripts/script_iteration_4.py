
    import os
    import json
    from google import genai
    from google.genai import types

    def call_llm(prompt, system_instruction=None):
        """Call the Gemini LLM with a prompt and return the response"""
        try:
            # Initialize the Gemini client
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

            # Call the API with system instruction if provided
            if system_instruction:
                response = client.models.generate_content(
                    model="gemini-2.0-flash", 
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    ),
                    contents=prompt
                )
            else:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    ),
                    contents=prompt
                )

            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return f"Error: {str(e)}"

    def extract_information(text):
        """Extract key information from the input text"""
        system_instruction = "You are an information extraction specialist."

        prompt = f"""
        Extract key information from this text. Focus on identifying important elements and relationships.

        Example:
        Input: The project must be completed by June 15th and requires collaboration between the engineering and design teams.
        Output: {"deadline": "June 15th", "teams_involved": ["engineering", "design"], "requirement": "collaboration"}

        Now extract information from this input:
        {text}
        """

        return call_llm(prompt, system_instruction)

    def generate_solution(problem):
        """Generate a solution to the problem"""
        system_instruction = "You are a problem-solving expert."

        prompt = f"""
        Generate a detailed solution for this problem:

        Example:
        Problem: Design a simple notification system that sends alerts when a temperature sensor exceeds 30°C.
        Solution: Create a monitoring service that polls the temperature sensor every minute. When a reading exceeds 30°C, trigger the notification system to send an alert via email and SMS to registered users, including the current temperature value and timestamp.

        Now solve this problem:
        {problem}
        """

        return call_llm(prompt, system_instruction)

    def main(question):
        """Main function to solve problems"""
        try:
            # Step 1: Extract key information
            information = extract_information(question)

            # Step 2: Generate a solution
            solution = generate_solution(question)

            # Return the solution
            return solution
        except Exception as e:
            print(f"Error in main: {str(e)}")
            return "I couldn't generate a solution due to an error."
    