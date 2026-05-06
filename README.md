Kurulum ve Veri Akışı
Sistemi yeni bir bilgisayarda (örneğin Sena'nın bilgisayarı) kurmak ve çalıştırmak için aşağıdaki adımları sırasıyla takip edebilirsiniz.

1. Adım: Bağımlılıkların Kurulumu
   Gerekli kütüphaneleri yüklemek için terminalde şu komutu çalıştırın:

Bash
pip install -r requirements.txt
Eğer requirements.txt dosyasını kullanmak yerine tek tek kurmak isterseniz:

Bash
pip install langchain-chroma chromadb langchain-huggingface groq python-dotenv pandas scikit-learn

2. Adım: Ortam Değişkenleri (.env)
   Proje ana dizininde bir .env dosyası oluşturun ve Groq API anahtarınızı ekleyin(bana wpden gönderdiğin anahtarları ekleyebilirsin birini eklesen yetr)

Kod snippet'i
GROQ*API_KEY=gsk*...

3. Adım: Veri Ön İşleme (Data Preprocessing)
   IMDb veri setindeki yorumları temizlemek (noktalama işaretlerini ve gereksiz karakterleri kaldırmak) için preprocess.py dosyasını çalıştırın. Bu adım, verileri işleyerek cleaned_reviews.csv dosyasını oluşturur:

Bash
python src/preprocess.py

4. Adım: ChromaDB Veritabanını Kurma ve Yükleme
   Temizlenen verileri (vektörleştirerek) ChromaDB'ye aktarmak ve ./chroma_db dizinini oluşturmak için vector_storage.py dosyasını çalıştırın:

Bash
python src/vector_storage.py

bu kadar şimdilik
