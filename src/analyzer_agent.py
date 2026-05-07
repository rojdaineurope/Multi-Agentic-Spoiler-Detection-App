# analyzer_agent.py

class AnalyzerAgent:
    def __init__(self, client):
        self.client = client

    def extract_claims(self, comment):
        prompt = f"""
Break the review into sentences.

Identify sentences that may reveal plot points.

Return ONLY suspicious sentences.

Review:
{comment}
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content