import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from main_agents import identify_spoiler  # Senin yazdığın fonksiyon

# 1. TEST VERİSİNİ HAZIRLA (Elimizdeki temizlenmiş veriden örnek alıyoruz)
# Not: 'cleaned_reviews.csv' dosyasında 'is_spoiler' ve 'movie_id' sütunları olmalı
df = pd.read_csv("cleaned_reviews.csv")
test_set = df.sample(20)  # API limitlerini zorlamamak için şimdilik 20 örnekle test edelim

y_true = []  # Gerçek etiketler
y_pred = []  # RAG'ın tahminleri

print("RAG Değerlendirmesi Başladı... Lütfen bekleyin.")

for index, row in test_set.iterrows():
    yorum = row['cleaned_review']
    gercek_etiket = row['is_spoiler'] # 1 (Spoiler) veya 0 (Normal)
    m_id = row['movie_id']
    
    # RAG sistemine soruyoruz
    try:
        ajan_cevabi = identify_spoiler(yorum, m_id)
        tahmin = 1 if "KARAR: SPOILER" in ajan_cevabi else 0
        
        y_true.append(gercek_etiket)
        y_pred.append(tahmin)
        print(f"İşleniyor: {index} | Gerçek: {gercek_etiket} | Tahmin: {tahmin}")
    except Exception as e:
        print(f"Hata oluştu: {e}")

# 2. CONFUSION MATRIX OLUŞTURMA
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal', 'Spoiler'], 
            yticklabels=['Normal', 'Spoiler'])

plt.title('Multi-Agent RAG Confusion Matrix')
plt.xlabel('RAG Tahmini')
plt.ylabel('Gerçek Durum')
plt.savefig("rag_confusion_matrix.png")
plt.show()

# 3. RAPORLAMA
print("\n--- RAG PERFORMANS RAPORU ---")
print(classification_report(y_true, y_pred))