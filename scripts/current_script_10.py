import os
import re
import math

def main(question):
    """
    This script uses a "Holistic Reading & Arithmetic Reasoner" agent.
    Instead of strict decomposition, it combines reading comprehension with
    arithmetic problem-solving into a single, unified step. The hypothesis is that
    by encouraging the LLM to reason holistically about both text and numbers, it
    can avoid the errors associated with decomposition or separate stages. This directly
    addresses previous struggles with arithmetic and misinterpretation of intent.
    This also uses more than one example in different parts of the code.
    """
    try:
        holistic_reasoner = HolisticReadingArithmeticReasoner()
        answer = holistic_reasoner.answer_question(question)
        return answer
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

class HolisticReadingArithmeticReasoner:
    """
    A class that embodies the Holistic Reading & Arithmetic Reasoner.
    """
    def __init__(self):
        self.system_instruction = """You are a Holistic Reading & Arithmetic Reasoner. You analyze the passage and question to understand both text and arithmetic aspects and provide a well-reasoned answer."""

    def answer_question(self, question, max_attempts=3):
        """
        Answers the question using a holistic understanding and reasoning approach.
        """
        for attempt in range(max_attempts):
            reasoning_result = self._reason_about_question(question)

            # Verification stage: check if the reasoning is a valid response
            verification_prompt = f"""
            Verify if the reasoning result is a valid and well reasoned answer to the original problem.

            Original Question: {question}

            Reasoning result: {reasoning_result}

            Example 1:
            Original Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
            Reasoning result: Chris Johnson's first touchdown was 6 yards and Jason Hanson's first field goal was 53 yards, therefore the answer is 59 yards.
            Valid

            Example 2:
            Original Question: Who caught the final touchdown of the game?
            Reasoning result: Wes Welker caught the final touchdown of the game, therefore the answer is Wes Welker.
            Valid
            """

            validation_result = call_llm(verification_prompt, self.system_instruction)

            if "valid" in validation_result.lower():
                # Return the answer portion only
                answer_match = re.search(r'answer is (.*)', reasoning_result)
                if answer_match:
                    return answer_match.group(1).strip()
                else:
                    return reasoning_result
            else:
                print(f"Result failed to validate on attempt {attempt + 1}")

        return "The holistic reasoner did not arrive at a conclusive and valid answer."

    def _reason_about_question(self, question):
        """Reason about the question using a holistic understanding of passage."""
        reasoning_prompt = f"""
        Reason about the question and passage to formulate a direct and comprehensive answer. Extract any relevant numerical quantities and perform calculations if necessary.

        Question: {question}

        Example 1:
        Question: How many yards did Chris Johnson's first touchdown and Jason Hanson's first field goal combine for?
        Reasoning: Chris Johnson's first touchdown was 6 yards and Jason Hanson's first field goal was 53 yards, therefore the answer is 59 yards.

        Example 2:
        Question: Who caught the final touchdown of the game?
        Reasoning: Wes Welker caught the final touchdown of the game, therefore the answer is Wes Welker.

        Reasoning:
        """
        return call_llm(reasoning_prompt, self.system_instruction)

def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response. DO NOT deviate from this example template or invent configuration options. This is how you call the LLM."""
    try:
        from google import genai
        from google.genai import types
        import os  # Import the os module

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