# https://medium.com/@leighphil4/a-dspy-optimized-ai-interview-assistant-c90a35c8f3da

# Import modules
import os
import dspy
import streamlit as st
from dspy import Signature, Module, InputField, OutputField, Predict, Prediction, Example
from dspy.teleprompt import BootstrapFewShot
from llama_parse import LlamaParse
from dspy.evaluate import Evaluate
import dspy
import os

from dotenv import load_dotenv
load_dotenv()


lm = dspy.LM('openai/gpt-4o-mini', api_key="OPENAI_API_KEY")
dspy.configure(lm=lm)

# Set up Resume and Job Description Parsing Functions
def parse_resume(resume_file):
    parser = LlamaParse(
        api_key="llx-EXmz9Z6NZRKTBGLsPBYG7bxxxxxxxxxxxxxxxnGDK",
        result_type="text",
        verbose=True,
    )
    resume_document = parser.load_data(resume_file)
    resume_text = resume_document[0].text
    return resume_text
print("Resume parsed...")
