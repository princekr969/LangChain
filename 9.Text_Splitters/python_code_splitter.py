from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """

def calculate_sum(a, b):
    result = a + b
    return result


def calculate_product(a, b):
    result = a * b
    return result


def greet(name):
    message = f"Hello, {name}!"
    print(message)


class Calculator:
    def __init__(self, value):
        self.value = value

    def square(self):
        return self.value * self.value


numbers = [1, 2, 3, 4, 5]

for number in numbers:
    print(number)

sum_result = calculate_sum(10, 20)
product_result = calculate_product(5, 6)

print("Sum:", sum_result)
print("Product:", product_result)

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))

print(chunks)
