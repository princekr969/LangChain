# Local Model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from local_model import get_model

model =  get_model()
# Prompt
from langchain_core.prompts import PromptTemplate

#output parser
from langchain_core.output_parsers import StrOutputParser


template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = "Write a 5 line summary on the following text. \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

# Chain
chain = template1 | model | parser | template2 | model | parser

result  = chain.invoke({'topic': 'black-hole'})

print(result)