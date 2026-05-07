# main.py

from base import client, vector_db
from analyzer_agent import AnalyzerAgent
from retriever_agent import RetrieverAgent
from classifier_agent import ClassifierAgent
from critic_agent import CriticAgent
from orchestrator import OrchestratorAgent

analyzer = AnalyzerAgent(client)
retriever = RetrieverAgent(vector_db)
classifier = ClassifierAgent(client)
critic = CriticAgent(client)

orchestrator = OrchestratorAgent(
    analyzer,
    retriever,
    classifier,
    critic
)

if __name__ == "__main__":
    comment = "The main character dies at the end but overall great acting."
    movie_id = "tt0111161"

    result = orchestrator.run(comment, movie_id)
    print(result)