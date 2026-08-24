# The code of langchain is very consistant
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# To know about the models visit platform.openai.com 
# Here we can give more parameter to the model:
# temperature vary from 0-2, it is creative parameter mean how creative reponse you want.
# max_completion_tokens it tells the model, how much token (roughy we can consider like word) you want in output, it helpfull in paid models you need to pay on the basic of tokens
model = ChatOpenAI(model='gpt-4', temperature=1.8, max_completion_tokens=10)

result = model.invoke("what is th capital of india")

# Here in result it return a object which content all the things like token_usage, output, metadata etc.
# to return only the
print(result)