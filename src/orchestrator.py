# orchestrator.py

class OrchestratorAgent:
    def __init__(self, analyzer, retriever, classifier, critic):
        self.analyzer = analyzer
        self.retriever = retriever
        self.classifier = classifier
        self.critic = critic

    def run(self, comment, movie_id):
        # 1. Analyzer → şüpheli cümleleri bul
        claims = self.analyzer.extract_claims(comment)

        if not claims or claims.strip() == "":
            return "KARAR: NORMAL\nReason: No suspicious sentences found."

        # 2. Retriever → context getir
        context = self.retriever.get_context(comment, movie_id)

        if context is None:
            return "HATA: Context bulunamadı"

        # 3. Classifier → karar ver
        initial_result = self.classifier.classify(
            comment=claims,   # 🔥 sadece şüpheli cümleleri veriyoruz
            context=context
        )

        # 4. Critic → kontrol
        final_result = self.critic.review(
            comment=claims,
            classifier_output=initial_result
        )

        return final_result