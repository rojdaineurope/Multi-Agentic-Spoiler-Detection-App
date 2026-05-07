# retriever_agent.py

class RetrieverAgent:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def get_context(self, comment, movie_id, k=3):
        results = self.vector_db.similarity_search(
            comment,
            k=k,
            filter={"movie_id": movie_id}
        )

        if not results:
            return None

        seen = set()
        unique_chunks = []

        for r in results:
            if r.page_content not in seen:
                seen.add(r.page_content)
                unique_chunks.append(r.page_content)

        return "\n---\n".join(unique_chunks)