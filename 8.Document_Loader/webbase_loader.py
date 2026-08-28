import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel

from langchain_community.document_loaders import TextLoader, WebBaseLoader

model = get_model()
prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text poem - \n {text}",
    input_variable=['question', 'text']
)

parser = StrOutputParser()
url = "https://www.amazon.com/CyberPowerPC-Xtreme-GeForce-Windows-GXiVR8080A41/dp/B0DXVFWSS7/?_encoding=UTF8"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser
chain.invoke({'question':'What is the product we are talking about?', 'text':docs[0]})

print(len(docs))
print(docs[0].page_content)
