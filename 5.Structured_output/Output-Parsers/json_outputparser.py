# Local Model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from local_model import get_model

model =  get_model()
# Prompt
from langchain_core.prompts import PromptTemplate

#output parser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

parser = JsonOutputParser()

template = PromptTemplate(
    template = "Give me the name, age and city of fictional person \n {format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser


result  = chain.invoke({})
print(result)
