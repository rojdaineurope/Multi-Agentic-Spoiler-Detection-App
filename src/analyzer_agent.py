# analyzer_agent.py

class AnalyzerAgent:
    def __init__(self, client):
        self.client = client

    def extract_claims(self, comment):
        prompt = f"""
You are a Linguistic Analyzer Agent specializing in movie reviews.
Your task is to break down the review and extract ONLY the sentences that contain potential spoilers.

If you find suspicious sentences, format your output as:
Claim: [sentence]
Reasoning: [why it's suspicious]

If the review is completely safe (e.g., only talks about acting, emotions, music, or general opinions), you MUST output exactly in this format:
NO_SUSPICIOUS_FOUND
Reason: [Explain exactly WHY it is NORMAL. Example: 'The review only expresses personal feelings and praises the cinematography without revealing any plot details.']

Review:
{comment}
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1, 
            max_tokens=150, 
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content