import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

model = get_model()

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        'Classify the sentiment of the following feedback text into positive or negative.\n'
        'Respond with ONLY a JSON object like this: {{"sentiment": "positive"}}\n'
        'Do not repeat the schema. Do not explain.\n'
        'Feedback: {feedback}\n{format_instructions}'
    ),
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

result = classifier_chain.invoke({'feedback': 'This is a wonderful smartphone'}).sentiment

prompt2 = PromptTemplate(
    template='Write an appropriate respone to thsi positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='Write an appropriate respone to thsi negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive', prompt2 | model | parser1),
    (lambda x:x['sentiment'] == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain
print(chain.invoke('feedback': 'This is a terrible phone'))
chain.get_graph().print_ascii()



