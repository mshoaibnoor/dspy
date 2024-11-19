# https://adasci.org/dspy-streamlining-llm-prompt-optimization/

import dspy
import dspy.evaluate
import os

from dotenv import load_dotenv
load_dotenv()

API_KEY=os.getenv('OPENAI_API_KEY')
# print(API_KEY)

lm = dspy.LM('openai/gpt-4o-mini', api_key=API_KEY)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Configuring LM and RM
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
dspy.configure(lm=lm)

# Loading data set
from dspy.datasets import HotPotQA

dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
print(len(trainset),len(devset))

# Building signatures
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


# Building the pipeline
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context,answer=prediction.answer)
        # return prediction

# Optimizing the Pipeline
from dspy.teleprompt import BootstrapFewShot
def validate_context_and_answer(example,pred,trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example,pred)
    answer_PM = dspy.evaluate.answer_passage_match(example,pred)
    return answer_EM and answer_PM

teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
compile_rag = teleprompter.compile(RAG(),trainset=trainset)

# Executing the Pipeline
my_question = "What castle did David Gregory inhert?"
pred = compile_rag(my_question)
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")