import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

model = get_model()
parser = StrOutputParser()
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}', 
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}', 
    input_variables=['text']
)

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic':'Russia vs Ukraine'})
print(result)

final_chain.get_graph().print_ascii()




