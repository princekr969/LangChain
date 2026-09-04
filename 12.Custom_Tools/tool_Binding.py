import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests


llm = get_model()
# tool create
@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))
print(multiply.name)
print(multiply.description)

llm_with_tool = llm.bind_tools({multiply})

llm_with_tool.inovoke('Can You Multiply 3 with 10').tool_calls

# tool execution
query = HumanMessage('can you multiply 3 with 1000')
messages = [query]
result = llm_with_tools.invoke(messages)
messages.append(result)
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)
output = llm_with_tools.invoke(messages).content
print(output)