from langchain_core.tools import tool

@tool
def multiply(a, b):
    """multiply two Number"""
    return a*b

result = multiply.invoke({"a":3, "b":5})

print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)