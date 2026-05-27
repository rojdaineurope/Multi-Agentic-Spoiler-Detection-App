# orchestrator.py

class OrchestratorAgent:
    def __init__(self, analyzer, retriever, classifier, critic):
        self.analyzer = analyzer
        self.retriever = retriever
        self.classifier = classifier
        self.critic = critic

    def run(self, comment, movie_id):
        # 1. Analyzer find suspicious letters
        claims = self.analyzer.extract_claims(comment)

        # If Analyzer turns 'NO_CLAIMS_FOUND' it gives it with its reason 
        if "NO_SUSPICIOUS_FOUND" in claims:
            return claims.replace("NO_SUSPICIOUS_FOUND", "KARAR: NORMAL")

        

        # 2. Retriever brings context
        context = self.retriever.get_context(comment, movie_id)

        if context is None:
            return "HATA: Context bulunamadı"

        # 3. Classifier takes a decision
        initial_result = self.classifier.classify(
            comment=claims,   
            context=context
        )

        # 4. Critic control/validate
        final_result = self.critic.review(
            comment=claims,
            classifier_output=initial_result
        )

        return final_result