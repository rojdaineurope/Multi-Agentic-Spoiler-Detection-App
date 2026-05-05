import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

SPOILER_KEYWORDS = [
    "dies", "killed", "death", "escapes", "ending", "twist",
    "turns out", "revealed", "becomes", "betrays", "sacrifices",
    "murdered", "survives", "final scene", "last scene"
]
NEGATION_WORDS = [
    "nobody", "no one", "doesn't", "never", "not", "without",
    "none", "nothing", "nor", "no"
]

def get_all_movie_ids():
    results = vector_db.get(include=['metadatas'])
    if results is None:
        return []
    metadatas = results.get('metadatas', [])
    if metadatas is None:
        return []
    unique_ids = set()
    for meta in metadatas:
        if meta and 'movie_id' in meta:
            unique_ids.add(str(meta['movie_id']).strip())
    return list(unique_ids)

def keyword_check(comment: str):
    words = comment.lower().split()
    for i, word in enumerate(words):
        clean_word = word.strip(".,!?\"'")
        if clean_word in SPOILER_KEYWORDS:
            window = [w.strip(".,!?\"'") for w in words[max(0, i - 3):i]]
            if not any(neg in window for neg in NEGATION_WORDS):
                return "KARAR: SPOILER\nReason: Keyword match — explicit plot signal."
    return None

def get_k(comment: str) -> int:
    # 5'e çıkmak yerine bilgi kirliliğini engellemek için k'yı maksimum 3 ile sınırlandırıyoruz.
    word_count = len(comment.split())
    if word_count < 15:
        return 2
    else:
        return 3
def identify_spoiler(comment: str, movie_id: str) -> str:
    movie_id = str(movie_id).strip()
    available_ids = get_all_movie_ids()

    if movie_id not in available_ids:
        return f"KARAR: HATA - Girilen '{movie_id}' ID'sine ait veri bulunamadı."

    # Keyword Override önce çalışır
    keyword_result = keyword_check(comment)
    if keyword_result is not None:
        return keyword_result

    results = vector_db.similarity_search(
        comment,
        k=get_k(comment),
        filter={"movie_id": movie_id}
    )

    if not results:
        return "Film özeti hafızada bulunamadı."

    # Duplicate chunk'ları temizle
    seen = set()
    unique_results = []
    for r in results:
        if r.page_content not in seen:
            seen.add(r.page_content)
            unique_results.append(r)

    movie_context = "\n---\n".join([r.page_content for r in unique_results])

    prompt = f"""
You are a spoiler detector. Classify the review as SPOILER or NORMAL.

MOVIE CONTEXT:
{movie_context}

REVIEW:
{comment}


SPOILER = reveals ending, character death/fate/kill/escape/villain, plot twist, specific scene outcome.
NORMAL = general opinion, actor or director praise, genre or theme comments, pacing and soundtrack discussion, or general production quality, without revealing plot developments or outcomes.

Examples:
Review: "The main character dies at the end." -> KARAR: SPOILER
Review: "Great acting by the lead role and the pacing was perfect." -> KARAR: NORMAL
Review: "I loved the cinematography, though the story was slow." -> KARAR: NORMAL

Rules:
- The NORMAL class includes discussions about acting, directing, technical aspects, and general enjoyment of the movie.
- If the review discusses a character but does not reveal what happens to them or how the story resolves, it is NORMAL.
- Any explicit plot reveal, ending description, or specific plot outcome must be classified as SPOILER.
- Cleaned text has no punctuation — read carefully.
- When uncertain → NORMAL

Reply with ONLY:
KARAR: SPOILER
or
KARAR: NORMAL
Reason: [max 8 words]
"""

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=60,
    )

    result = chat_completion.choices[0].message.content

    # Safety net: model NORMAL dedi ama açık keyword var
    if "KARAR: NORMAL" in result:
        comment_lower = comment.lower()
        if any(kw in comment_lower for kw in SPOILER_KEYWORDS):
            return "KARAR: SPOILER\nReason: Keyword override — explicit plot reveal detected."

    return result


if __name__ == "__main__":
    test_cases = [
        ("tt0111161", "The main character manages to escape from prison through a tunnel he dug for years."),
        ("tt0111161", "Morgan Freeman and Tim Robbins deliver outstanding performances in this film."),
        ("tt0468569", "Heath Ledger's joker is the greatest villain performance in cinema history."),
        ("tt0068646", "Michael becomes the new godfather and orders the killing of all rival family heads."),
        ("tt0110912", "The briefcase is never opened and its contents are never revealed to the audience."),
        ("tt0050083", "In the end, the jury votes not guilty and the boy is acquitted."),
    ]

    print("=" * 50)
    print("AGENT TEST BAŞLIYOR")
    print("=" * 50)

    for movie_id, comment in test_cases:
        print(f"\nFilm ID : {movie_id}")
        print(f"Yorum   : {comment}")
        result = identify_spoiler(comment, movie_id)
        print(f"Sonuç   : {result}")
        print("-" * 50)