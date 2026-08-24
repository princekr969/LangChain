from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

import os

# We running the model local by default it save at your C drive but my drive is full that why i want the model is save at my D drive
os.environ['HF_HOME'] = 'D:/huggingface_cache'


llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        do_sample=True,
        max_new_tokens=100,
        return_full_text=False
    )
)
model = ChatHuggingFace(llm=llm)

message = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(message)

message.append(AIMessage(content=result.content))

print(message)