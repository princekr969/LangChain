from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
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

st.header('Research Tool')

# Static Prompt:
# user_input = st.text_input('Enter Your Prompt:')



paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# Template
template = load_prompt('template.json')

# Dynamic Prompt
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button('Summarize'):
    with st.spinner('Generating...'):
        result = model.invoke(prompt)
    st.write(result.content)
