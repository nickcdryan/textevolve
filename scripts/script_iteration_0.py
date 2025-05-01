def main(question):
    """Transform the test input grid according to patterns shown in training examples, enhanced with multi-example prompting and validation."""
    try:
        # Enhanced prompt to extract transformation rule with multiple examples
        prompt = f"""
        You are an expert at identifying grid transformation rules from examples.
        Analyze the training examples and describe the transformation rule in simple terms.
        Then, apply this rule to the test input to generate the transformed grid.

        Example 1:
        Input Grid:
        [[0, 7, 7], [7, 7, 7], [0, 7, 7]]
        Output Grid:
        [[0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7, 7], [0, 7, 7, 0, 7, 7, 0, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7], [0, 0, 0, 7, 7, 7, 7, 7, 7], [0, 0, 0, 0, 7, 7, 0, 7, 7]]
        Transformation Rule: Each original cell is expanded to a 3x3 block.

        Example 2:
        Input Grid:
        [[4, 0, 4], [0, 0, 0], [0, 4, 0]]
        Output Grid:
        [[4, 0, 4, 0, 0, 0, 4, 0, 4], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 0, 4, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0]]
        Transformation Rule: Each original cell is expanded to a 3x3 block of the same value.

        Example 3:
        Input Grid:
        [[0, 0, 0], [0, 0, 2], [2, 0, 2]]
        Output Grid:
        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 2, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 2], [2, 0, 2, 0, 0, 0, 2, 0, 2]]
        Transformation Rule: Each original cell is expanded to a 3x3 block of the same value.

        Question: {question}

        Let's analyze the training examples and determine the transformation rule,
        then apply it to the test input.

        Describe the transformation rule in one sentence. Then apply that rule to the test grid.

        """
        result = call_llm(prompt, system_instruction="You are an expert grid transformer.")

        # Implement a fallback mechanism in case of failure or unexpected output.
        if result and "Transformation Rule:" in result:
            return result.split("Transformation Rule:")[1].strip()
        else:
            return "Failed to transform grid. Please check the input and try again."

    except Exception as e:
        return f"An error occurred: {str(e)}"