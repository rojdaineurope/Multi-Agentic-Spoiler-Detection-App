import os
import shutil
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_reviews_to_chroma():
    # 1. Dosya Yolu Kontrolü
    csv_path = "cleaned_reviews.csv"
        
    if not os.path.exists(csv_path):
        print(f"Hata: Veri dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return

    print("Veri seti yükleniyor...")
    df = pd.read_csv(csv_path)

    # 2. Embedding Modelini Yükle
    print("Embedding modeli (all-MiniLM-L6-v2) yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. ChromaDB Bağlantısı ve Sıfırlama
    if os.path.exists("./chroma_db"):
        print("Eski veritabanı temizleniyor...")
        shutil.rmtree("./chroma_db")

    print("Yeni ChromaDB oluşturuluyor...")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 4. Dökümanları Oluştur
    docs = []
    for index, row in df.iterrows():
        if pd.isna(row['cleaned_review']) or pd.isna(row['movie_id']):
            continue
            
        review_text = str(row['cleaned_review']).strip()
        movie_id_val = str(row['movie_id']).strip()
        
        docs.append(
            Document(
                page_content=review_text,
                metadata={"movie_id": movie_id_val}
            )
        )

    print(f"Toplam {len(docs)} döküman hazırlanıyor...")
    
    # 5. Dökümanları Batch (Paket) Halinde Ekleme
    batch_size = 4000  # Sınırın altında güvenli bir boyut
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(f"Yükleniyor: {i} ile {i + len(batch)} arasındaki dökümanlar...")
        vector_db.add_documents(batch)
        
    print("✅ İşlem tamamlandı! Veritabanı başarıyla oluşturuldu.")

if __name__ == "__main__":
    load_reviews_to_chroma()