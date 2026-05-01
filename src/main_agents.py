import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. .env dosyasındaki değişkenleri yükle
load_dotenv() 

# 2. Anahtarı sistemden çek
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 3. Hafızayı Bağla (ChromaDB klasörünü okur)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def get_all_movie_ids():
    """Hafızadaki (ChromaDB) tüm benzersiz film ID'lerini eksiksiz ve temizlenmiş döndürür."""
    results = vector_db.get(include=['metadatas'])
    metadatas = results.get('metadatas', [])
    unique_ids = set()
    for meta in metadatas:
        if 'movie_id' in meta:
            movie_id_val = str(meta['movie_id']).strip()
            unique_ids.add(movie_id_val)
    return list(unique_ids)

def identify_spoiler(comment, movie_id):
    movie_id = str(movie_id).strip()
    available_ids = get_all_movie_ids()
    
    if movie_id not in available_ids:
        return f"KARAR: HATA - Girilen '{movie_id}' ID'sine ait veri bulunamadı."

    # Bağlam miktarını artırıyoruz (k=4)
    results = vector_db.similarity_search(comment, k=4, filter={"movie_id": movie_id})
    
    if not results:
        return "Film özeti hafızada bulunamadı."
    
    movie_context = "\n".join([res.page_content for res in results])

    # Modelin daha tutarlı karar vermesi için Few-Shot (Örnekli) Prompt
    prompt = f"""
    Sen bir Film Spoiler Tespit Uzmanısın. Amacın, kullanıcının yorumunun film özetine göre spoiler içerip içermediğini bulmaktır.
    Yorum Türkçe veya İngilizce olabilir. Dilden bağımsız olarak cümlenin anlamına odaklan.

    FİLM ÖZETİ (GERÇEK BİLGİ):
    {movie_context}

    KULLANICI YORUMU:
    {comment}

    ÖRNEKLER:
    --------------------
    Yorum: "The main character manages to escape from prison."
    KARAR: SPOILER
    Çünkü yorum, ana karakterin hapisten kaçtığını ve filmin sonucunu/olay örgüsünü ifşa etmektedir.
    --------------------
    Yorum: "The acting was very good and the music is great."
    KARAR: NORMAL
    Çünkü yorum herhangi bir spoiler içermeyen, oyunculuk ve müzik üzerine genel bir değerlendirmedir.
    --------------------

    KURALLAR:
    Eğer yorum aşağıdaki durumları içeriyorsa kesinlikle 'KARAR: SPOILER' ver:
    - Filmin sonu, finali veya finalde ne olduğu.
    - Filmin sürpriz gelişmeleri (plot twist).
    - Karakterlerin kaderiyle ilgili kritik bilgiler.

    Eğer yorum sadece genel bir düşünce, oyunculuk değerlendirmesi veya spoiler olmayan yüzeysel bir ifadeyse 'KARAR: NORMAL' ver.

    GÖREV:
    1. Cevabına 'KARAR: SPOILER' veya 'KARAR: NORMAL' diyerek başla.
    2. Nedenini kısa ve net bir cümleyle açıkla.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",  # Modelimizi daha yüksek limitli olanla değiştiriyoruz
        temperature=0.1,
    )
    
    return chat_completion.choices[0].message.content

# --- TEST BÖLÜMÜ ---
if __name__ == "__main__":
    test_movie_id = "tt0111161" 
    test_comment = "The main character manages to escape from prison."
    
    print("Ajan analizi başlatılıyor...")
    sonuc = identify_spoiler(test_comment, test_movie_id)
    print("\n--- AJAN ANALİZİ ---")
    print(sonuc)