import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

model = get_model()
prompt1 = PromptTemplate(
    template='Write a joke about {topic}', 
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    'Joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})
chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(chain.invoke({'topic': 'Cricket'}))


    


