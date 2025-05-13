from clone_repo import clone_repo
from embedder import embed_and_store
from retriever import retrieve_code
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
load_dotenv()

# Get credentials and config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "code-index")

# Create Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=PINECONE_ENVIRONMENT or 'us-east-1'
        )
    )
    print(f"Created index: {PINECONE_INDEX_NAME}")
else:
    print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

# Connect to Pinecone index
index = pc.Index(PINECONE_INDEX_NAME)
print("Connected to Pinecone index:", PINECONE_INDEX_NAME)

if __name__ == "__main__":
    url = input("\U0001F517 Enter GitHub repo URL to clone: ")
    path = clone_repo(url)
    embed_and_store(path)

    while True:
        q = input("\n💬 Ask a code question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        results = retrieve_code(q)

        print("\n🔍 Top Matches:\n")
        for i, res in enumerate(results):
            print(f"[{i+1}] {res[:300]}...\n")
