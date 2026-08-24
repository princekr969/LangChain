from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# We running the model local by default it save at your C drive but my drive is full that why i want the model is save at my D drive
os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the Capital of India?")

print(result.content)

