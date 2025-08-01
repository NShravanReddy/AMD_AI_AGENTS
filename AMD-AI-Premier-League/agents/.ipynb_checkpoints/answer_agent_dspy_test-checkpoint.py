import dspy
import os

lm = dspy.LM('azure/gpt-4o', api_key=os.getenv("AZURE_OPENAI_API_KEY"), api_base=os.getenv("AZURE_OPENAI_ENDPOINT"))
dspy.configure(lm=lm)

loaded_dspy_program = dspy.load("Evaluation/dspy/optimizer_scripts/dspy_programs/mcq_solver")

# 2) Call it just like you would the original optimized_react
result = loaded_dspy_program(mcq_prompt="""
You are an expert at solving multiple choice questions. You must respond with EXACT JSON format.

Here are examples of the input format and expected output:

Input Example 1:
{
  "topic": "Logical Reasoning",
  "question": "Find the next number in the sequence: 2, 5, 10, 17, ?",
  "choices": [
    "A) 24",
    "B) 26", 
    "C) 25",
    "D) 23"
  ],
  "answer": "C"
}

Expected Output:
{
  "answer": "C",
  "reasoning": "The sequence follows the pattern: +3, +5, +7, +9. Each difference increases by 2. So 17 + 8 = 25."
}

Input Example 2:
{
  "topic": "Truth-teller and Liar",
  "question": "Alice says, 'Bob is a liar.' Bob says, 'Alice tells the truth.' Who is liar?",
  "choices": [
    "A) Alice",
    "B) Bob",
    "C) Both", 
    "D) Neither"
  ],
  "answer": "C"
}

Expected Output:
{
  "answer": "C",
  "reasoning": "If Alice tells truth, Bob is liar, but Bob says Alice tells truth (contradiction). If Alice lies, Bob tells truth, but then Alice's statement about Bob being liar is false, making Bob truthful (contradiction). Both must be liars."
}

Input Example 3:
{
  "topic": "Blood Relations and Family Tree",
  "question": "John is Amy's father. Amy is Lisa's mother. What is John to Lisa?",
  "choices": [
    "A) Grandfather",
    "B) Uncle",
    "C) Father",
    "D) Brother"
  ],
  "answer": "A"
}

Expected Output:
{
  "answer": "A", 
  "reasoning": "John is Amy's father, and Amy is Lisa's mother. This makes John the father of Lisa's mother, which means John is Lisa's grandfather."
}

Now solve this question and respond with EXACT JSON format:

{
  "topic": "Logical Reasoning",
  "question": "Find the next number in the sequence: 2, 5, 10, 17, ?",
  "choices": [
    "A) 24",
    "B) 26",
    "C) 25",
    "D) 23"
  ],
  "answer": "C"
}

Your response must be valid JSON in this exact format:
{
  "answer": "<correct choice letter only>",
  "reasoning": "brief reasoning within 100 words for why the answer is correct"
}
""")

# 3) Print out the answer
print("Answer:", result.expected_json_output)
