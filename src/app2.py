import streamlit as st

from base import client, vector_db
from analyzer_agent import AnalyzerAgent
from retriever_agent import RetrieverAgent
from classifier_agent import ClassifierAgent
from critic_agent import CriticAgent
from orchestrator import OrchestratorAgent


# 🔹 Agents
analyzer = AnalyzerAgent(client)
retriever = RetrieverAgent(vector_db)
classifier = ClassifierAgent(client)
critic = CriticAgent(client)

orchestrator = OrchestratorAgent(
    analyzer,
    retriever,
    classifier,
    critic
)

# UI
st.set_page_config(page_title="Spoiler Agent", layout="centered")

st.title(" Spoiler Detection Agent ")
st.write("Yorumunuzun spoiler içerip içermediğini ajanlarımız birlikte analiz eder.")

# Inputs
movie_id = st.text_input("Film ID (Örn: tt0105112)", "tt0105112")
user_comment = st.text_area("Film Hakkındaki Yorumun:", placeholder="Buraya yorumunu yaz...")

if st.button("Analiz Et"):
    if user_comment:
        with st.spinner('Ajanlar analiz yapıyor...'):
            try:
                # 🔍 Analyzer
                claims = analyzer.extract_claims(user_comment)

                # 🎯 Final pipeline
                result = orchestrator.run(user_comment, movie_id)

                st.divider()

                # 🔍 Extracted Claims
                st.subheader(" Suspicious Sentences (Linguistic-Analyzer Agent)")
                if claims and claims.strip():
                    st.info(claims)
                else:
                    st.warning("Şüpheli cümle bulunamadı.")

                # 🎯 Final Decision
                st.subheader("Final Decision")

                if "KARAR: SPOILER" in result:
                    st.error(result)
                else:
                    st.success(result)

            except Exception as e:
                st.error(f"Hata: {e}")
    else:
        st.warning("Lütfen önce bir yorum yaz!")