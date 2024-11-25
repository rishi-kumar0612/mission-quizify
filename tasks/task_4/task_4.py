import vertexai
from vertexai.preview.language_models import TextEmbeddingModel


class EmbeddingClient:
    """
    Initializes an embedding client to connect to Google Cloud's Vertex AI for text embeddings.
    """

    def __init__(self, model_name, project, location):
        # Initialize the Vertex AI SDK
        vertexai.init(project=project, location=location)
        
        # Load the embedding model
        self.client = TextEmbeddingModel.from_pretrained(model_name)

    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.

        :param query: The text query to embed.
        :return: The embeddings for the query.
        """
        embeddings = self.client.get_embeddings([query])
        return embeddings[0].values  # Return the embedding vector

    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.

        :param documents: A list of text documents to embed.
        :return: A list of embeddings for the given documents.
        """
        embeddings_list = self.client.get_embeddings(documents)
        return [embedding.values for embedding in embeddings_list]

if __name__ == "__main__":
    model_name = "textembedding-gecko@003"
    project = "velvety-ray-437319-a6"  # Replace with your actual project ID
    location = "us-central1"

    embedding_client = EmbeddingClient(model_name, project, location)
    vectors = embedding_client.embed_query("Hello World!")
    if vectors:
        print(vectors)
        print("Successfully used the embedding client!")
