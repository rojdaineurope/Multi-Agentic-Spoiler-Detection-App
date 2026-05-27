import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Language packet
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. HTML labels cleaned
    text = re.sub(r'<.*?>', '', text)
    # 2. Sjust hold letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. lowercase
    text = text.lower().strip()
    # 4. remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# Load Data
print(" Data reading")

df = pd.read_json('IMDB_reviews.json', lines=True, chunksize=20000)
sample_df = next(df)

print(f"Toplam {len(sample_df)}  rows are cleaning...")
tqdm.pandas() 
sample_df['cleaned_review'] = sample_df['review_text'].progress_apply(clean_text)


# 'is_spoiler' target label
processed_df = sample_df[['movie_id', 'is_spoiler', 'cleaned_review']]

# record cleaned dataset
processed_df.to_csv('cleaned_reviews.csv', index=False)
print(" 'cleaned_reviews.csv' ")

import os

output_file = 'cleaned_reviews.csv'

#if not os.path.exists(output_file):
#   print("Temizlenmiş dosya bulunamadı. İşlem başlatılıyor...")

#    processed_df.to_csv(output_file, index=False)
#else:
#    print(f"'{output_file}' zaten mevcut, direkt yükleniyor...")
#    processed_df = pd.read_csv(output_file)

# For see how much spoiler we have but not necessary right now
#print("\n--- Veri Seti Spoiler Dağılımı ---")
#print(processed_df['is_spoiler'].value_counts())