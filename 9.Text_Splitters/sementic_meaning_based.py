import local model using below code
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model


from langchain_experimental.text_splitter import SementicChunker

text_splitter = SementicChunker(

)