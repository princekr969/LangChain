# import local model using below code
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from local_model import get_model


import os

# Set HF_HOME before any huggingface/transformers import happens,
# so the cache goes to D drive instead of C drive
os.environ['HF_HOME'] = 'D:/huggingface_cache'

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

_model = None  # cache so we don't reload the model every time get_model() is called

def get_model(
    model_id: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    temperature: float = 0.5,
    max_new_tokens: int = 100
) -> ChatHuggingFace:
    """
    Returns a cached ChatHuggingFace model instance.
    Loads the model only once per process, even if called multiple times.
    """
    global _model
    if _model is None:
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task='text-generation',
            pipeline_kwargs=dict(
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                return_full_text=False
            )
        )
        _model = ChatHuggingFace(llm=llm)
    return _model