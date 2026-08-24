import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = get_model()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variable=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variable=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()
