import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

model = get_model()

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []
with open('chat_history.txt') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        role, _, text = line.partition(':')
        role = role.strip().lower()
        text = text.strip()
        if role == 'human':
            chat_history.append(HumanMessage(content=text))
        elif role == 'ai':
            chat_history.append(AIMessage(content=text))

prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'where is my refund'})

response = model.invoke(prompt)
print(response.content)