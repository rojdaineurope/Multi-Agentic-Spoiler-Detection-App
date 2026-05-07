# critic_agent.py

SPOILER_KEYWORDS = [
    "dies", "killed", "death", "ending",
    "twist", "betrays", "murdered"
]

class CriticAgent:
    def __init__(self, client):
        self.client = client

    def review(self, comment, classifier_output):
        # rule-based override
        if "KARAR: NORMAL" in classifier_output:
            if any(k in comment.lower() for k in SPOILER_KEYWORDS):
                return "KARAR: SPOILER\nReason: keyword override"

        # LLM-based second opinion
        prompt = f"""
You are a strict reviewer.

Initial decision:
{classifier_output}

Comment:
{comment}

If wrong → correct it.
Else → repeat same.

Output format:
KARAR: ...
Reason: ...
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=60,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content