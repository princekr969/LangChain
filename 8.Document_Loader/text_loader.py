import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

from langchain_community.document_loaders import TextLoader

model = get_model()
prompt = PromptTemplate(
    template='Write a summary for the following text - \n {text}',
    input_variable=['text']
)

parser = StrOutputParser()
loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({'text':docs[0].page_content})
print(result)
print(len(docs))
print("Docs Content: ",docs[0].page_content)
print("Docs metaData: ",docs[0].metadata)
print(docs)
print(docs[0])