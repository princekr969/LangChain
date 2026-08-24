from lanchain_openai import OpenAI
from dotenv import load_dotenv

# Load the env file varible to this file
load_dotenv()

# We create an object of openai and then we say which model we want to used
# THis OpenAi API is paid we need to add 5 dollar to use it api.
llm = OpenAI(model='gpt-3.5-turbo-instruct')

# Invoke is the very import function in langchain, it is avaliable in each componenet of langchain model, prompt, chain etc.
# here with the help of this function we communicate with this gpt model
# So this invoke function  hit the gpt model and the user query to the gpt model and return the result/output
result = llm.invoke("What is the capital of India")

print(result)