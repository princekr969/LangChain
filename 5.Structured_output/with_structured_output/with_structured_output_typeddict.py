from LangChain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypeDict, Annotated

load_dotenv()
model = ChatOpenAI()

# schema
# we add annotated to my code so the llm understand the structure in more better way because only TypedDict is simplest way the llm may get create ambiguity 
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive, or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

# when we used with_structured_output behind the scene it generate a prompt and give it to the model with the user input
structured_model = model.with_structured_output(Review)

result = model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)
print(result['summary'])
print(result['sentiment'])