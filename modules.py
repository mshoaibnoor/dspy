# https://dspy.ai/learn/programming/modules/

import dspy
import os

lm = dspy.LM('openai/gpt-4o-mini', api_key="OPENAI_API_KEY")
dspy.configure(lm=lm)



sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict("sentence -> sentiment: bool")

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

print(response)


question = "What's something great about the ColBERT retrieval model?"
classify = dspy.ChainOfThought("question -> answer", n=5)

response = classify(question=question)
print(response.completions.answer)

print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")