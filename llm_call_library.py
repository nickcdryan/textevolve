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


# GEMINI

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