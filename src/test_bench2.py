import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

# 🔥 YENİ IMPORTLAR
from base import client, vector_db
from analyzer_agent import AnalyzerAgent
from retriever_agent import RetrieverAgent
from classifier_agent import ClassifierAgent
from critic_agent import CriticAgent
from orchestrator import OrchestratorAgent


def run_benchmark_in_batches():

    # 🔹 Agents initialize
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

    # Test veri seti
    test_df = pd.read_csv("test_dataset3.csv")

    total_time = 0
    correct_predictions = 0

    y_true = []
    y_pred = []

    batch_size = 10
    sleep_time = 10
    #eski hali 10 a 10 du

    total_docs = len(test_df)

    print(f"🚀 Multi-Agent Benchmark başlatıldı...\n")

    for i in range(0, total_docs, batch_size):
        batch_df = test_df.iloc[i:i + batch_size]

        print(f"--- [{i} - {i + len(batch_df)}] arası işleniyor ---")

        for index, row in batch_df.iterrows():
            comment = str(row['cleaned_review'])
            movie_id = str(row['movie_id'])

            original_is_spoiler = row['is_spoiler']
            true_label_val = 1 if original_is_spoiler else 0
            y_true.append(true_label_val)

            expected_label = "KARAR: SPOILER" if original_is_spoiler else "KARAR: NORMAL"

            start_time = time.time()

            # 🔥 YENİ SİSTEM
            result = orchestrator.run(comment, movie_id)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # prediction
            if "KARAR: SPOILER" in result:
                y_pred.append(1)
            else:
                y_pred.append(0)

            if expected_label in result:
                correct_predictions += 1

            print(f"ID: {movie_id} | {elapsed_time:.2f}s | {'✅' if expected_label in result else '❌'}")

        if i + batch_size < total_docs:
            print(f"⏳ {sleep_time} sn bekleniyor...\n")
            time.sleep(sleep_time)

    # 📊 METRİKLER
    accuracy = (correct_predictions / total_docs)
    avg_latency = total_time / total_docs

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "="*40)
    print(" MULTI-AGENT SONUÇLARI")
    print("="*40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Latency  : {avg_latency:.2f} sec")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

    # 📉 GÖRSEL
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Spoiler"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Multi-Agent Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_multi_agent.png")
    plt.show()

    print("✅ confusion_matrix_multi_agent.png kaydedildi")


if __name__ == "__main__":
    run_benchmark_in_batches()