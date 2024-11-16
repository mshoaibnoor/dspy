
import dspy
import os

lm = dspy.LM('openai/gpt-4o-mini', api_key="OPENAI_API_KEY")
dspy.configure(lm=lm)


# math = dspy.ChainOfThought("question -> answer: float")
# answer = math(question="Two dice are tossed. What is the probability that the sum equals two?")

# print(answer)

# def search(query: str) -> list[str]:
#     """Retrieves abstracts from Wikipedia."""
#     results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
#     return [x['text'] for x in results]

# rag = dspy.ChainOfThought('context, question -> response')

# question = "What's the name of the castle that David Gregory inherited?"
# prediction = rag(context=search(question), question=question)

# print(prediction)

from typing import Literal

# class Classify(dspy.Signature):
#     """Classify sentiment of a given sentence."""

#     sentence: str = dspy.InputField()
#     sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
#     confidence: float = dspy.OutputField()

# classify = dspy.Predict(Classify)
# prediction = classify(sentence="This book was super fun to read, though not the last chapter.")

# print(prediction)


# text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features in a press release."

# module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
# response = module(text=text)

# print(response.title)
# print(response.headings)
# print(response.entities_and_metadata)


def evaluate_math(expression: str) -> float:
    return dspy.PythonInterpreter({}).execute(expression)

def search_wikipedia(query: str) -> str:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
print(pred.answer)