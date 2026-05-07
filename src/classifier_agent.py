# classifier_agent.py

class ClassifierAgent:
    def __init__(self, client):
        self.client = client

    def classify(self, comment, context):
        prompt = f"""
You are a spoiler detection agent.

CONTEXT:
{context}

REVIEW:
{comment}

Decide:
KARAR: SPOILER or KARAR: NORMAL
Reason: max 8 words
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content