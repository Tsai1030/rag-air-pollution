import os
import json
import logging
import shutil
import fitz  # PyMuPDF
import uuid
import traceback
from io import BytesIO

# ✨ 新增的 import ✨
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from unstructured.partition.docx import partition_docx
# --------------------

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# ---------- Logging 設定 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- 常數設定 ----------
PERSIST_DIRECTORY = "6_12"  # 強烈建議每次修改後都用新的目錄名
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# ---------- 建立向量資料庫函式 ----------
def build_new_vectordb(documents, embeddings, persist_dir):
    if os.path.exists(persist_dir):
        logging.info(f"🧹 清除原有資料夾：{persist_dir}")
        shutil.rmtree(persist_dir, ignore_errors=True)

    if not documents:
        logging.warning("⚠️ 沒有文件可以建立向量資料庫。")
        return None

    valid_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if len(valid_documents) < len(documents):
        logging.warning(f"⚠️ 過濾掉 {len(documents) - len(valid_documents)} 個內容為空的文件區塊。")

    if not valid_documents:
        logging.warning("⚠️ 過濾後沒有有效的文件可以建立向量資料庫。")
        return None

    logging.info(f"🏗️ 準備將 {len(valid_documents)} 個有效文件區塊寫入向量資料庫...")
    try:
        logging.info("🛡️ 執行最終的元資料複雜類型過濾...")
        cleaned_documents_for_chroma = filter_complex_metadata(valid_documents)
        logging.info(f"   過濾後，準備寫入 {len(cleaned_documents_for_chroma)} 個文件區塊。")
        
        vectordb = Chroma.from_documents(
            documents=cleaned_documents_for_chroma,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        logging.info(f"✅ 向量資料庫已建立並自動儲存至：{persist_dir}")
        return vectordb
    except Exception as e:
        logging.error(f"❌ 建立向量資料庫時出錯: {e}")
        logging.error(traceback.format_exc())
        return None

# ---------- 載入並處理 JSON 文件函式 ----------
def load_documents_from_json_files(file_paths):
    final_documents = []
    docs_to_split = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
    )

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.warning(f"⚠️ 文件路徑不存在，將跳過：{file_path}")
            continue

        file_basename = os.path.basename(file_path)
        file_basename_lower = file_basename.lower()
        is_kmu_usr_file = "kmu_usr" in file_basename_lower
        is_qa_file = any(key in file_basename_lower for key in ["qa", "問答"])
        should_chunk_this_file = not (is_kmu_usr_file or is_qa_file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logging.warning(f"⚠️ 文件 {file_basename} 不是 JSON 列表格式，將跳過。")
                continue
            
            log_tags = []
            if is_qa_file: log_tags.append("QA (不切割, 優化處理)")
            elif is_kmu_usr_file: log_tags.append("KMU USR (不切割)")
            else: log_tags.append("長文本 (將進行切割)")
            logging.info(f"📥 開始處理：{file_basename} ({', '.join(log_tags)})")

            for i, entry in enumerate(data):
                if not isinstance(entry, dict):
                    logging.warning(f"   - 在 {file_basename} 中找到非字典格式的項目，索引 {i}，將跳過。")
                    continue

                page_content_text = ""
                current_metadata = {"source": file_basename}
                
                if is_qa_file: current_metadata["document_type"] = "QA_Pair"
                elif is_kmu_usr_file: current_metadata["document_type"] = "KMU_USR_Report"
                else: current_metadata["document_type"] = "General_Text"

                metadata_from_entry = entry.get("metadata", {})
                if isinstance(metadata_from_entry, dict):
                    for k, v in metadata_from_entry.items():
                        if isinstance(v, list): current_metadata[k] = ", ".join(map(str, v))
                        elif isinstance(v, dict):
                            try: current_metadata[k] = json.dumps(v, ensure_ascii=False)
                            except TypeError: current_metadata[k] = str(v)
                        elif v is None: current_metadata[k] = ""
                        elif isinstance(v, (str, int, float, bool)): current_metadata[k] = v
                        else: current_metadata[k] = str(v)
                else:
                    logging.warning(f"   - 在 {file_basename} 索引 {i} 的 metadata 不是字典，已忽略其內容。")

                if "doc_id" not in current_metadata:
                    current_metadata["doc_id"] = str(uuid.uuid4())

                if is_qa_file:
                    question = entry.get("question", "").strip()
                    answer = entry.get("answer", "").strip()
                    if question and answer:
                        page_content_text = question
                        current_metadata["full_answer"] = answer
                    else:
                        page_content_text = entry.get("page_content", "").strip()
                else:
                    page_content_text = entry.get("page_content", entry.get("content", entry.get("text", ""))).strip()

                if not page_content_text:
                    logging.warning(f"   - 在文件 {file_basename} 索引 {i} (doc_id: {current_metadata.get('doc_id', 'N/A')}) 找不到有效內容，將跳過。")
                    continue
                
                doc = Document(page_content=str(page_content_text), metadata=current_metadata)
                if should_chunk_this_file: docs_to_split.append(doc)
                else: final_documents.append(doc)

        except Exception as e:
            logging.error(f"❌ 處理文件 {file_basename} 時發生嚴重錯誤：{e}")
            logging.error(traceback.format_exc())

    if docs_to_split:
        logging.info(f"📚 準備切分 {len(docs_to_split)} 份長文檔...")
        try:
            split_documents = splitter.split_documents(docs_to_split)
            logging.info(f"   ✅ 長文切分完成，生成 {len(split_documents)} 個區塊。")
            final_documents.extend(split_documents)
        except Exception as e:
            logging.error(f"❌ 切分長文本時發生錯誤：{e}")
            logging.error(traceback.format_exc())

    logging.info(f"📦 所有 JSON 文件處理完成，共生成 {len(final_documents)} 個文件區塊。")
    return final_documents

# ---------- 載入並處理 PDF 文件函式 ----------
def load_documents_from_pdfs(pdf_paths, text_splitter):
    all_pdf_chunks = []
    for path in pdf_paths:
        if not os.path.exists(path):
            logging.warning(f"⚠️ PDF 文件路徑不存在，將跳過：{path}")
            continue
        
        file_basename = os.path.basename(path)
        logging.info(f"📄 開始處理 PDF：{file_basename}")
        
        try:
            doc_fitz = fitz.open(path)
            full_text = "\n".join(page.get_text() for page in doc_fitz)
            
            if full_text:
                base_metadata = {
                    "source": file_basename,
                    "doc_id": f"{os.path.splitext(file_basename)[0]}_pdf_{str(uuid.uuid4())[:8]}",
                    "document_type": "PDF_Document"
                }
                doc_for_splitting = Document(page_content=full_text, metadata=base_metadata)
                split_chunks = text_splitter.split_documents([doc_for_splitting])
                all_pdf_chunks.extend(split_chunks)
                logging.info(f"   ✅ PDF {file_basename} 切分完成，生成 {len(split_chunks)} 個區塊。")
            else:
                logging.warning(f"   ⚠️ PDF {file_basename} 未能提取到任何有效文本。")
        except Exception as e:
            logging.error(f"❌ 讀取或處理 PDF {file_basename} 失敗：{e}")
            logging.error(traceback.format_exc())
    
    logging.info(f"📄 所有 PDF 文件處理完成，共生成 {len(all_pdf_chunks)} 個文件區塊。")
    return all_pdf_chunks

# ✨ ==============================================================================
# ✨ 【新增】處理 Word 文件的進階函式
# ✨ ==============================================================================
def initialize_caption_model():
    """初始化並返回圖片描述模型和處理器，會自動檢測 GPU。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Salesforce/blip-image-captioning-large"
    logging.info(f"🖼️ 正在 {device} 上初始化圖片描述模型 ({model_id})...")
    try:
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        logging.info("   ✅ 圖片描述模型初始化成功。")
        return processor, model, device
    except Exception as e:
        logging.error(f"❌ 初始化圖片描述模型失敗: {e}")
        logging.warning("   ⚠️ 將無法處理 Word 文件中的圖片。")
        return None, None, None

def process_word_documents(word_paths, text_splitter, caption_model_components):
    """
    處理 Word (.docx) 文件，提取文本、表格和圖片描述。
    """
    all_word_docs = []
    processor, caption_model, device = caption_model_components
    
    for path in word_paths:
        if not os.path.exists(path):
            logging.warning(f"⚠️ Word 文件路徑不存在，將跳過：{path}")
            continue
            
        file_basename = os.path.basename(path)
        logging.info(f"📄 開始處理 Word 文件：{file_basename}")

        try:
            elements = partition_docx(
                filename=path, 
                infer_table_structure=True,
                include_page_breaks=False,
                strategy="hi_res"
            )

            text_buffer = []
            
            for i, element in enumerate(elements):
                base_metadata = {
                    "source": "kao_usr",  # ✨【核心要求】根據您的要求設定
                    "original_filename": file_basename,
                    "doc_id": str(uuid.uuid4()),
                }

                if "unstructured.documents.elements.Table" in str(type(element)):
                    if text_buffer:
                        content = "\n\n".join(text_buffer)
                        doc_for_splitting = Document(page_content=content, metadata=base_metadata.copy())
                        all_word_docs.extend(text_splitter.split_documents([doc_for_splitting]))
                        text_buffer = []
                    
                    table_html = getattr(element.metadata, 'text_as_html', str(element))
                    table_metadata = base_metadata.copy()
                    table_metadata["chunk_type"] = "table"
                    table_metadata["element_id"] = element.id
                    table_doc = Document(page_content=table_html, metadata=table_metadata)
                    all_word_docs.append(table_doc)
                    logging.info(f"   - 找到並轉換了一個表格 (HTML 格式)")
                
                elif "unstructured.documents.elements.Image" in str(type(element)) and caption_model:
                    if text_buffer:
                        content = "\n\n".join(text_buffer)
                        doc_for_splitting = Document(page_content=content, metadata=base_metadata.copy())
                        all_word_docs.extend(text_splitter.split_documents([doc_for_splitting]))
                        text_buffer = []

                    try:
                        img_bytes = getattr(element.metadata, 'image_bytes', None)
                        if img_bytes:
                            img = Image.open(BytesIO(img_bytes)).convert("RGB")
                            inputs = processor(images=img, return_tensors="pt").to(device)
                            outputs = caption_model.generate(**inputs, max_new_tokens=75) # 增加描述長度
                            caption = processor.decode(outputs[0], skip_special_tokens=True)
                            
                            image_metadata = base_metadata.copy()
                            image_metadata["chunk_type"] = "image"
                            image_metadata["element_id"] = element.id
                            image_doc = Document(page_content=f"圖片內容描述：{caption}", metadata=image_metadata)
                            all_word_docs.append(image_doc)
                            logging.info(f"   - 找到並生成了一張圖片的描述：'{caption[:60]}...'")
                        else:
                            logging.warning("   - 找到圖片元素但無法提取圖片數據。")
                    except Exception as img_e:
                        logging.warning(f"   - 處理 Word 中的一張圖片時出錯: {img_e}")
                
                else:
                    text_buffer.append(element.text)

            if text_buffer:
                content = "\n\n".join(text_buffer)
                doc_for_splitting = Document(page_content=content, metadata=base_metadata.copy())
                all_word_docs.extend(text_splitter.split_documents([doc_for_splitting]))

        except Exception as e:
            logging.error(f"❌ 處理 Word 文件 {file_basename} 時發生嚴重錯誤：{e}")
            logging.error(traceback.format_exc())

    logging.info(f"📄 所有 Word 文件處理完成，共生成 {len(all_word_docs)} 個文件區塊。")
    return all_word_docs

# ---------- 主程式執行區 ----------
if __name__ == "__main__":
    # 步驟 1: 設定嵌入模型
    logging.info(f"🧠 準備載入嵌入模型：{EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logging.info(f"✅ 嵌入模型已成功載入到 GPU。")
    except Exception as e:
        logging.error(f"❌ 無法載入嵌入模型到 GPU：{e}")
        logging.info("   嘗試在 CPU 上載入...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logging.info(f"✅ 嵌入模型已成功載入到 CPU。")
        except Exception as e_cpu:
            logging.error(f"❌ 在 CPU 上載入嵌入模型也失敗：{e_cpu}")
            exit()

    # ✨ 新增：初始化多模態模型 (僅需一次)
    caption_model_components = initialize_caption_model()

    json_file_paths = [
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\文獻資料修正版.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\新聞資料修正版三家合併.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\gov_policy_2024_2025.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\政府資料QA.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\環境部QA.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\air_pollution_enforcement_2025.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\air_pollution_enforcement_2025QA.json",
        r"C:\Users\USER\Desktop\資料庫目前資料\第一次測試資料\尚未加入4_19\air_quality_monitoring_2023.json",
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
        r"C:\Users\USER\Desktop\429清理\113-116QA.json",
        r"C:\Users\USER\Desktop\429清理\odt_all.json",
        r"C:\Users\USER\Desktop\429清理\ODT_QA_fixed.json",
        r"C:\Users\USER\Desktop\429清理\pollution-control.json",
        r"C:\Users\USER\Desktop\429清理\pollution-control_QA.JSON",
        r"C:\Users\USER\Desktop\429清理\113-116_2_metadata_completed.json",
        r"C:\Users\USER\Desktop\504高雄fqa\kaohsiung_air_QA.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr9.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr10.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr11.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr12.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr13.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr14.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr15.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr16.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr17.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr18.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr19.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr20.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr21.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr22.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr23.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr24.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr25.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr26.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr27.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr28.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr29.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr30.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr31.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr32.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr33.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr34.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr35.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr1.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr2.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr3.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr4.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr5.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr6.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr7.json",
        r"C:\Users\USER\Desktop\kmu_usr_5_5要匯入的資料\kmu_usr8.json",
        r"C:\Users\USER\Desktop\6_3修正USR計劃加上高醫大\修正後的空汙qa_v2.json"
    ]
    
    pdf_file_paths = [
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


    # ✨ 新增：定義 Word 文件路徑 (請確保檔案已轉為 .docx)
    word_file_paths = [
        r"C:\Users\USER\Desktop\usr計畫空汙data\高雄醫學大學 114年-116年 空污下的大學社會責任-環境教育與健康促進永續發展計畫 提案計.docx"
    ]

    # 步驟 3: 載入並處理文件
    
    # 初始化一個通用的文本分割器
    general_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )
    
    # 處理 JSON 文件
    logging.info("🚀 開始載入 JSON 文件...")
    json_documents = load_documents_from_json_files(json_file_paths)
    logging.info(f"✔️ JSON 文件處理完成，共處理 {len(json_documents)} 個 Document 物件。")

    # 處理 PDF 文件
    logging.info("🚀 開始載入 PDF 文件...")
    pdf_documents = load_documents_from_pdfs(pdf_file_paths, general_text_splitter)
    logging.info(f"✔️ PDF 文件處理完成，共處理 {len(pdf_documents)} 個 Document 物件。")

    # ✨ 新增：載入並處理 Word 文件
    logging.info("🚀 開始載入 Word 文件...")
    word_documents = process_word_documents(word_file_paths, general_text_splitter, caption_model_components)
    logging.info(f"✔️ Word 文件處理完成，共處理 {len(word_documents)} 個 Document 物件。")

    # 步驟 4: 合併所有來源的文件
    all_processed_documents = json_documents + pdf_documents + word_documents
    if not all_processed_documents:
        logging.warning("⚠️ 所有文件來源處理後均為空，無法建立資料庫。請檢查文件路徑和內容。")
        exit()
    logging.info(f"🧩 最終合併所有來源的文件/區塊總數：{len(all_processed_documents)}")

    # 步驟 5: 建立向量資料庫
    vectordb = build_new_vectordb(all_processed_documents, embeddings, persist_dir=PERSIST_DIRECTORY)

    if vectordb:
        logging.info(f"🎉 向量資料庫建置完成！儲存於 '{PERSIST_DIRECTORY}'")
        logging.info("💡 提醒：請記得更新您的 RAG 主應用程式，使其從新的資料庫目錄載入。")
    else:
        logging.error("❌ 向量資料庫建置失敗，請檢查上面的錯誤日誌。")
