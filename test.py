import faiss
import numpy as np

def inspect_faiss_index(index_path: str) -> None:
    """Loads a Faiss index and prints its dimensions.

    Args:
        index_path: The path to the Faiss index file.
    """
    try:
        index = faiss.read_index(index_path)
        print(f"Successfully loaded Faiss index from: {index_path}")
        print(f"Number of vectors in the index: {index.ntotal}")
        print(f"Dimension of vectors in the index: {index.d}")
    except Exception as e:
        print(f"Error loading or inspecting Faiss index: {e}")

if __name__ == "__main__":
    # Assuming your script is in the 'lean-explore' directory
    # and the index is in 'data/main_faiss.index'
    inspect_faiss_index("data/main_faiss.index")
