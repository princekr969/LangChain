import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

model = get_model()
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}', 
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a linkedin post about {topic}', 
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result =  parallel_chain.invoke({'topic':'AI'})

print(result)