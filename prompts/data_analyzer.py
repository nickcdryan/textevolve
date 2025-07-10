import json

def get_dataset_analysis_prompt(formatted_examples):
    """
    Generate prompt and system instruction for dataset analysis.

    Args:
        formatted_examples: List of example dictionaries to analyze

    Returns:
        tuple: (prompt, system_instruction)
    """
    system_instruction = """You are a Data Pattern Analyst specialized in identifying patterns, 
        structures, and challenges in datasets. Your goal is to provide deep 
        insights and creative strategies for approaching problems."""


    prompt = f"""
You're analyzing a new dataset to identify patterns, structures, and potential problem-solving approaches.

Here are some representative examples from the dataset:
{json.dumps(formatted_examples, indent=2)}

Think deeply about these examples and provide a comprehensive analysis with the following sections:

## DATASET CHARACTERISTICS
- What patterns do you observe in the questions?
- What patterns do you observe in the answers?
- What is the structure and format of both inputs and outputs?
- What domain knowledge might be required?
- Are all questions of the same type, or are there multiple types?
- Do all questions require the same type of reasoning?

## DATA CHALLENGES
- What makes these problems difficult?
- What potential edge cases or complexities should be considered?
- What types of reasoning are required to solve these problems?

## POTENTIAL APPROACHES
- What solution strategies might work well for this type of problem?
- What decomposition of the problem would be most effective?
- What validation techniques would help ensure correct solutions?
- How would you handle unusual or edge cases?

## CREATIVE INSIGHTS
- What non-obvious patterns or shortcuts might exist?
- What unique perspectives might help solve these problems?
- What analogies to other problem domains might be useful?

## IMPLEMENTATION RECOMMENDATIONS
- What verification steps would be crucial?
- What intermediate steps or representations would be helpful?
- Assuming you will work mostly with text as inputs and outputs, what specific techniques would be most effective? Remember, overreliance on JSON parsing and complex code generation often leads to errors and low performance. Instead, focus on leveraging the LLM's natural reasoning abilities.

Be specific and concrete in your analysis. Focus on actionable insights that would help develop effective solutions.
"""

    return prompt, system_instruction