import os
import json
import logging
import shutil
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 嵌入與向量資料庫套件（新版本）
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------- Logging 設定 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- 建立向量資料庫（新版自動持久化） ----------
def build_new_vectordb(documents, embeddings, persist_dir="5_5test"): #4_23test
    if os.path.exists(persist_dir):
        logging.info(f"🧹 清除原有資料夾：{persist_dir}")
        shutil.rmtree(persist_dir, ignore_errors=True)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    logging.info(f"✅ 向量資料庫已建立並自動儲存至：{persist_dir}")
    return vectordb

# ---------- 載入 JSON 文件 ----------
def load_documents_from_json_files():
    file_paths = [
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\文獻資料修正版.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\修正後的空汙qa.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\新聞資料修正版三家合併.json",
        #r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\內部文件測試檔含metadata.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\gov_policy_2024_2025.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\政府資料QA.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\環境部QA.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\air_pollution_enforcement_2025.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\air_pollution_enforcement_2025QA.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\air_quality_monitoring_2023.json",
        # 4/27新增
        r"C:\Users\USER\Desktop\資料庫目前資料\清理後_內部文件測試檔.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\air_quality_report_110_clean.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\air_quality_report_111_clean.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\air_quality_report_112_clean.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\air_quality_report_113_clean.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\gpt整理成的usr計畫.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\grok整理usr計畫.json",
        r"C:\Users\USER\Desktop\清理後確定加入資料\2、3、一至三.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\gpt整理qa_with_docref.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\GROK整理qa_with_docref.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\2-4qa_gemini_with_docref.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\110年qa_with_doc_ref.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\111年qa_with_docref.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\112年qa_with_docref.json",
        r"C:\Users\USER\Desktop\清理處理後的qa\113qa_with_docref.json",
        #4/29新增
        r"C:\Users\USER\Desktop\429清理\113-116QA.json",
        r"C:\Users\USER\Desktop\429清理\odt_all.json",
        r"C:\Users\USER\Desktop\429清理\ODT_QA_fixed.json",
        r"C:\Users\USER\Desktop\429清理\pollution-control.json",
        r"C:\Users\USER\Desktop\429清理\pollution-control_QA.JSON",
        r"C:\Users\USER\Desktop\429清理\113-116_2_metadata_completed.json",
        r"C:\Users\USER\Desktop\504高雄fqa\kaohsiung_air_QA.json"
    ]

    long_text_documents = []
    qa_documents = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info(f"📥 已成功讀取：{os.path.basename(file_path)}")

            for entry in data:
                content = entry.get("page_content") or entry.get("content") or entry.get("text") or ""
                if not content.strip():
                    continue
                metadata = entry.get("metadata", {})
                metadata["source"] = metadata.get("source", os.path.basename(file_path))
                metadata = {
                    k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                    for k, v in metadata.items()
                }

                doc = Document(page_content=f"passage: {content.strip()}", metadata=metadata)

                if any(key in file_path.lower() for key in ["qa", "問答"]):
                    qa_documents.append(doc)
                else:
                    long_text_documents.append(doc)

        except Exception as e:
            logging.error(f"❌ {os.path.basename(file_path)} 讀取失敗：" + str(e))

    if long_text_documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        long_text_documents = splitter.split_documents(long_text_documents)
        logging.info(f"📚 長文 chunking 完成，共 {len(long_text_documents)} 段")

    all_docs = qa_documents + long_text_documents
    logging.info(f"📦 全部 JSON 文件合併完成，共 {len(all_docs)} 筆")
    return all_docs

# ---------- 載入 PDF 文件 ----------
def load_documents_from_pdfs(pdf_paths):
    all_documents = []
    for path in pdf_paths:
        try:
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                if not text:
                    continue
                metadata = {
                    "source": os.path.basename(path),
                    "page": page_num + 1,
                    "doc_id": os.path.splitext(os.path.basename(path))[0]
                }
                all_documents.append(Document(page_content=f"passage: {text}", metadata=metadata))
            logging.info(f"📄 已處理 PDF：{os.path.basename(path)}，共 {len(doc)} 頁")
        except Exception as e:
            logging.error(f"❌ 讀取 PDF 失敗：{os.path.basename(path)} - {e}")
    return all_documents

# ---------- 主程式 ----------
if __name__ == "__main__":
    model_name = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logging.info(f"✅ 嵌入模型已載入：{model_name}")

    # 載入 JSON 與 PDF 文件
    json_documents = load_documents_from_json_files()
    pdf_paths = [
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Long-term exposure to NO2 and O3 and all-cause and respiratory mortality A systematic review and meta-analysis.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\MegaSense_Cyber-PhysicalSystemforReal-timeUrbanAirQualityMonitoring.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Optimized machine learning model.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Predicting Concentration Levels of Air Pollutants by Transfer.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Quantifying the potential effects of air pollution reduction on population health and health expenditure in Taiwan.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\The association between airborne particulate matter (PM2.5) exposure level.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\The last decade of air pollution epidemiology and the challenges of quantitative risk assessment.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Air Quality Prediction with Physics-Guided Dual Neural ODEs in Open Systems.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\AirCast Improving Air Pollution Forecasting Through Multi-Variable Data Alignment.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\AirRadar Inferring Nationwide Air Quality in China with Deep Neural Networks.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Ambient air pollution and cardiovascular diseases An umbrella review of systematic reviews and meta-analyses.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Association between exposure to air pollution and increased ischaemic stroke incidence a retrospective population-based cohort study (EP-PARTICLES study).pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Forecasting Air Quality in Taiwan by Using Machine Learning.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Forecasting Smog Clouds With Deep Learning A Proof-Of-Concept.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Global, national, and urban burdens of paediatric asthma.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Impact of long-term exposure to ambient air pollution on the incidence of chronic obstructive pulmonary disease A systematic review and meta-analysis.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Long-term air pollution exposure and incident physical disability.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\批量處理\Long-term evaluation of a low-cost air sensor network for monitoring.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\Air Pollution Control Act Enforcement Rules.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\大資料\Guide on Ambient Air Quality Legislation - Air Pollution Series.pdf",
        r"C:\Users\USER\Desktop\暫定外部資料庫文獻\大資料\WHO global.pdf"
    ]
    pdf_documents = load_documents_from_pdfs(pdf_paths)

    all_documents = json_documents + pdf_documents
    logging.info(f"🧩 總文件數量：{len(all_documents)}")

    vectordb = build_new_vectordb(all_documents, embeddings, persist_dir="5_5test") #4_23test #4_27test#4_28test
    logging.info("🎉 向量資料庫建置完成！")
