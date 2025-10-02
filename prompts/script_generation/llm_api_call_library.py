# OPENAI

import openai
from openai import OpenAI

def call_llm(prompt, system_instruction=None):


    # Set your API key (keep this safe and secure)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Call the chat completion endpoint
    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
    )

    # Print the response content
    print(response.output[0].content[0].text)
    return response.output[0].content[0].text


# LLM API CALL - SYSTEM PROVIDED

def call_llm(prompt, system_instruction=None):
    """
    Call the LLM with a prompt and return the response.
    This function is provided by the system and handles all LLM interactions.
    DO NOT redefine this function or invent configuration options.
    
    Args:
        prompt: The prompt to send to the LLM
        system_instruction: Optional system instruction to guide the LLM's behavior
    
    Returns:
        str: The LLM's response
    
    Usage:
        response = call_llm("What is 2+2?")
        response = call_llm("Solve this problem", system_instruction="You are a math expert")
    """
    # This function is automatically available in generated scripts
    # The system handles the LLM client initialization and API calls
    # Just call it directly - implementation is injected at runtime
    pass


# Temperature with gemini:

# from google import genai
# from google.genai import types

# client = genai.Client(api_key="GEMINI_API_KEY")

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=["Explain how AI works"],
#     config=types.GenerateContentConfig(
#         max_output_tokens=500,
#         temperature=0.1
#     )
# )
# print(response.text)