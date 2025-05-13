import os
import uuid
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Load HuggingFace embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Pinecone API config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # should be "code-index"
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT")     # e.g., "us-east-1"

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec={"cloud": "aws", "region": PINECONE_REGION}
        )
    return pc.Index(PINECONE_INDEX_NAME)

def get_code_chunks(repo_path):
    chunks = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:  # only store non-empty
                        chunks.append(content)
    return chunks

def embed_and_store(repo_path):
    index = init_pinecone()
    code_chunks = get_code_chunks(repo_path)
    
    embeddings = model.encode(code_chunks)
    
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": emb.tolist(),
            "metadata": {"text": chunk}
        }
        for emb, chunk in zip(embeddings, code_chunks)
    ]
    
    index.upsert(vectors=vectors)
    print(f"✅ Stored {len(code_chunks)} chunks in Pinecone.")
