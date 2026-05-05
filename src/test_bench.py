import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from updatedversionofBABA import identify_spoiler  # Mevcut ajan fonksiyonunu içe aktarıyoruz

def run_benchmark_in_batches():
    # Test veri setini yükle
    test_df = pd.read_csv("test_dataset3.csv")
    
    total_time = 0
    correct_predictions = 0
    
    # Metrikleri hesaplamak için listeler
    y_true = []
    y_pred = []
    
    # Blok parametreleri
    batch_size = 10  # Her seferinde 10'ar yorum işlenecek
    sleep_time = 10   # Her bloktan sonra 6 dan 10 a cektik saniye beklenecek
    
    total_docs = len(test_df)
    
    print(f"🚀 Kıyaslama (Benchmark) testi {batch_size}'li bloklar halinde başlatılıyor...\n")
    
    for i in range(0, total_docs, batch_size):
        batch_df = test_df.iloc[i:i + batch_size]
        print(f"--- BİLGİLENDİRME: Dökümanlar işleniyor [{i} ile {i + len(batch_df)} arası] ---")
        
        for index, row in batch_df.iterrows():
            comment = str(row['cleaned_review'])
            movie_id = str(row['movie_id'])
            
            # Orijinal etiketi belirle
            original_is_spoiler = row['is_spoiler']
            
            # y_true listesini dolduruyoruz (1: Spoiler, 0: Normal)
            true_label_val = 1 if original_is_spoiler == True else 0
            y_true.append(true_label_val)
            
            expected_label = "KARAR: SPOILER" if original_is_spoiler == True else "KARAR: NORMAL"
            
            start_time = time.time()
            result = identify_spoiler(comment, movie_id)
            elapsed_time = time.time() - start_time
            
            total_time += elapsed_time
            
            # Tahmini y_pred listesine ekliyoruz
            if "KARAR: SPOILER" in result:
                y_pred.append(1)
            else:
                y_pred.append(0)
                
            if expected_label in result:
                correct_predictions += 1
                
            print(f"ID: {movie_id} | Süre: {elapsed_time:.2f} sn | Beklenen: {expected_label} | Sonuç: {'Başarılı ✅' if expected_label in result else 'Başarısız ❌'}")
        
        # Son blok değilse bekleme süresini uygula
        if i + batch_size < total_docs:
            print(f"⏳ Groq token limitini korumak için {sleep_time} saniye bekleniyor...\n")
            time.sleep(sleep_time)
            
    accuracy = (correct_predictions / total_docs) * 100
    avg_latency = total_time / total_docs
    
    # Sklearn metriklerini hesaplıyoruz
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*40)
    print(" TEST SONUÇLARI VE METRİKLER")
    print("="*40)
    print(f"Toplam Test Edilen Döküman: {total_docs}")
    print(f"Doğruluk Oranı (Accuracy): %{accuracy:.2f}")
    print(f"Ortalama Yanıt Süresi: {avg_latency:.2f} saniye")
    print("\n EK BAŞARI METRİKLERİ:")
    print(f"- Precision   : {precision:.4f}")
    print(f"- Recall      : {recall:.4f}")
    print(f"- F1-Score    : {f1:.4f}")
    print("\n Confusion Matrix:")
    print(cm)
    print("="*40)
    
    # Görselleştirme (Matplotlib Confusion Matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Spoiler"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    # Görüntüyü dosyaya kaydetme
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion Matrix 'confusion_matrix.png' olarak kaydedildi.")
    
    # Ekranı açıp gösterme
    plt.show()

if __name__ == "__main__":
    run_benchmark_in_batches()