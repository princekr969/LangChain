import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

from langchain_community.document_loaders import TextLoader, PyPDFLoader

model = get_model()
loader = PyPDFLoader('virtualization-notes.pdf')
docs = loader.load( )
print(len(docs))
print(docs[0].page_content)
print(docs[1].metadata)