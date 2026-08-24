import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = get_model()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variable=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'cricket'})
print(result)

chain.get_graph().print_ascii()