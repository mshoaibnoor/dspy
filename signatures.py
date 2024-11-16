# https://dspy.ai/learn/programming/signatures/

import dspy
import os


# Example A: Sentiment Classification
# example from the SST-2 dataset.
sentence = "it's a tough and often tiring journey."  # example from the SST-2 dataset.
classify = dspy.Predict("sentence -> sentiment: bool")
decision = classify(sentence=sentence).sentiment
print(decision)


# Example B: Summarization
# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""
summary = dspy.ChainOfThought("document -> summary")
response = summary(document=document)
print(response.summary)
print(response.reasoning)


## Class-based DSPy Signatures
# Example C: Classification
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotions."""
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']=dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion
classify = dspy.Predict(Emotion)
decision = classify(sentence=sentence)
print(decision)

# Example D: A metric that evaluates faithfulness to citations
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")


context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
result = faithfulness(context=context, text=text)
print(result)