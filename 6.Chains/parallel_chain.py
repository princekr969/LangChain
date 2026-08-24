import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from local_model import get_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel

model1 = get_model()
model2 = get_model()

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variable=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variable=['text']
)
prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variable=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

sample_text = """
Photosynthesis is the process by which green plants, algae, and some bacteria convert 
light energy into chemical energy. This process occurs mainly in the chloroplasts of 
plant cells, using a green pigment called chlorophyll. During photosynthesis, plants 
take in carbon dioxide from the air and water from the soil. Using sunlight as an 
energy source, they convert these raw materials into glucose, a type of sugar that 
serves as food for the plant. Oxygen is released as a byproduct of this reaction. 
The overall chemical equation for photosynthesis is: 
6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. 
Photosynthesis is essential for life on Earth because it produces the oxygen most 
living organisms need to breathe, and it forms the base of the food chain by 
converting solar energy into a form that other organisms can consume.
"""

result = chain.invoke({'text': sample_text})

print(result)

chain.get_graph().print_ascii()
