import csv
import chromadb
from chromadb.utils import embedding_functions

# Loading data and creating ChromaDB collection
with open("items.csv") as file:
    lines = csv.reader(file)
    documents = []
    metadatas = []
    ids = []
    id = 1

    for i, line in enumerate(lines):
        if i == 0:
            continue
        documents.append(line[1])
        metadatas.append({"item_id": line[0]})
        ids.append(str(id))
        id += 1

chroma_client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

def search_and_print_results(search_query):
    results = collection.query(
        query_texts=[search_query],
        n_results=5,
        include=['documents']
    )
    print(results['documents'])

if __name__ == '__main__':
    search_query = input("borek")
    search_and_print_results(search_query)
