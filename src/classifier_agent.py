# classifier_agent.py

class ClassifierAgent:
    def __init__(self, client):
        self.client = client

    def classify(self, comment, context):
        prompt = f"""
You are a strict spoiler detection judge. 
Your task is to compare the user's extracted claims against the official movie context.

CONTEXT (Official Plot Summary):
{context}

USER CLAIMS:
{comment}

SPOILER = reveals ending, character death/fate/kill/escape/villain, plot twist, specific scene outcome.
NORMAL = general opinion, actor or director praise, genre or theme comments, pacing and soundtrack discussion, or general production quality, without revealing plot developments or outcomes.

KARAR: [SPOILER or NORMAL]
Reason: [Provide a highly specific reason based on the context. If it is a SPOILER, explain exactly WHAT it reveals (e.g., 'Reveals the ultimate fate of the main character at the ending'). If it is NORMAL, explain exactly WHY it is safe (e.g., 'The claim only discusses the premise without giving away the ending').]
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",#Our model
            temperature=0, # We reduce the temperature to increase accuracy
            max_tokens=150, # We increase the token limit for the detailed explanation
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content