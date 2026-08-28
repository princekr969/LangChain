import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

model = get_model()
loader = DirectoryLoader(
    path='books',
    glob = '.pdf',
    loader_cl = PyPDFLoader
)

docs = loader.load()
print(len(doc))
print(doc[0].page_content)
print(doc[1].metadata)