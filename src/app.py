import streamlit as st
from main_agents import identify_spoiler

# Sayfa ayarlarını en başa alalım
st.set_page_config(page_title="Spoiler Agent", layout="centered")

st.title("Spoiler Detection Agent")
st.write("Film ID ve yorumunuzu girerek analiz başlatın.")
st.markdown("Yapay zeka ajanlarımız film özetlerini okur ve yorumunuzun spoiler içerip içermediğini analiz eder.")

# Giriş alanları
movie_id = st.text_input("Film ID (Örn: tt0105112)", "tt0105112")
user_comment = st.text_area("Film Hakkındaki Yorumun:", placeholder="Buraya yorumunu yaz...")

if st.button("Analiz Et"):
    if user_comment:
        with st.spinner('Ajanlar tartışıyor, lütfen bekle...'):
            try:
                result = identify_spoiler(user_comment, movie_id)
                
                # Karara göre renkli kutu göster
                if "KARAR: SPOILER" in result:
                    st.error(result)
                else:
                    st.success(result)
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
    else:
        st.warning("Lütfen önce bir yorum yaz!")