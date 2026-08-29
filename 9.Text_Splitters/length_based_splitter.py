from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """
The VMM uses a fixed top-4 MB region of the guest's address space. This region is reserved for the VMM and remains mapped while the guest is running. Because the VMM region exists inside the same address space that the guest is already using, the system does not need to perform a complete world switch whenever the VMM needs to execute.
In a full world switch, the processor would have to move from the guest environment to a separate VMM environment, which can involve changing address-space mappings and saving or restoring execution state. This adds overhead.
With the fixed top-4 MB approach, the VMM can access its own code and data through the already-mapped region. In simple terms, the VMM does not need to leave the current address-space context; it can directly use the reserved region and then return to the guest. This makes transitions between the guest and VMM faster and reduces the overhead associated with a full context or world switch.
"""

# Load PDF
loader = PyPDFLoader("virtualization-notes.pdf")
docs = loader.load()

# Create splitter
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

# Split normal text
result1 = splitter.split_text(text)

print("Chunks from text:")
print(result1)

# Split PDF documents
result2 = splitter.split_documents(docs)

print("\nChunks from PDF:")
print(result2)