import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Constants from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT")

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

def retrieve_code(query, top_k=3):
    index = init_pinecone()

    # Embed the query
    query_embedding = model.encode(query).tolist()

    # Perform vector search
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # Extract matched text chunks from metadata
    results = []
    for match in response['matches']:
        text = match.get('metadata', {}).get('text')
        if text:
            results.append(text)

    return results
