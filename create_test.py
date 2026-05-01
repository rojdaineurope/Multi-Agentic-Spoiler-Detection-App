import pandas as pd
import os

def create_gold_standard_dataset():
    # 1. Dosya Yolu Kontrolü
    csv_path = "cleaned_reviews.csv"
    if not os.path.exists(csv_path):
        print("Hata: cleaned_reviews.csv dosyası proje dizininde bulunamadı.")
        return

    print("Veri seti yükleniyor...")
    df = pd.read_csv(csv_path)

    # Sütun isimlerinin doğruluğunu kontrol et
    if 'is_spoiler' not in df.columns or 'cleaned_review' not in df.columns:
        print("Hata: Gerekli sütunlar veri setinde bulunamadı. Sütun isimlerini kontrol et.")
        return

    # 2. True (Spoiler) ve False (Normal) olarak filtreleme
    spoiler_df = df[df['is_spoiler'] == True]
    normal_df = df[df['is_spoiler'] == False]

    print(f"Toplam Spoiler Yorum Sayısı: {len(spoiler_df)}")
    print(f"Toplam Normal Yorum Sayısı: {len(normal_df)}")

    # 3. Her bir gruptan rastgele 50'şer satır seçme (random_state ile sabitliyoruz)
    spoiler_sample = spoiler_df.sample(n=50, random_state=42)
    normal_sample = normal_df.sample(n=50, random_state=42)

    # 4. Verileri birleştirme
    final_test_df = pd.concat([spoiler_sample, normal_sample], ignore_index=True)

    # 5. Dosya olarak kaydetme
    final_test_df.to_csv("test_dataset.csv", index=False)
    print(f"✅ İşlem tamamlandı! 100 satırlık test veri seti 'test_dataset.csv' olarak kaydedildi.")

if __name__ == "__main__":
    create_gold_standard_dataset()