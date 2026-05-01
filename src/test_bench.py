import time
import pandas as pd
from main_agents import identify_spoiler  # Mevcut ajan fonksiyonunu içe aktarıyoruz

def run_benchmark():
    # Test veri setini yükle
    test_df = pd.read_csv("test_dataset.csv")
    
    total_time = 0
    correct_predictions = 0
    
    print("🚀 Kıyaslama (Benchmark) testi başlatılıyor...\n")
    
    for index, row in test_df.iterrows():
        comment = str(row['cleaned_review'])
        movie_id = str(row['movie_id'])
        
        # Orijinal etiketi belirle
        original_is_spoiler = row['is_spoiler']
        expected_label = "KARAR: SPOILER" if original_is_spoiler == True else "KARAR: NORMAL"
        
        start_time = time.time()
        result = identify_spoiler(comment, movie_id)
        elapsed_time = time.time() - start_time
        
        total_time += elapsed_time
        
        # Sonucu kontrol et
        is_success = False
        if expected_label in result:
            is_success = True
            correct_predictions += 1
            
        print(f"ID: {movie_id} | Süre: {elapsed_time:.2f} sn | Beklenen: {expected_label} | Sonuç: {'Başarılı ✅' if is_success else 'Başarısız ❌'}")
        
    accuracy = (correct_predictions / len(test_df)) * 100
    avg_latency = total_time / len(test_df)
    
    print("\n" + "="*40)
    print("🎯 TEST SONUÇLARI")
    print("="*40)
    print(f"Toplam Test Edilen Döküman: {len(test_df)}")
    print(f"Doğruluk Oranı (Accuracy): %{accuracy:.2f}")
    print(f"Ortalama Yanıt Süresi: {avg_latency:.2f} saniye")
    print("="*40)

if __name__ == "__main__":
    run_benchmark()