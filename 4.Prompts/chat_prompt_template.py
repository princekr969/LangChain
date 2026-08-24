import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model

model = get_model()

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain {topic} in simple terms')
])

prompt = chat_template.invoke({'domain': 'engineering', 'topic': 'Clould Computing'})
print(prompt)
