# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, Depends, Request, HTTPException, APIRouter, Header
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from datetime import datetime
import logging
import time
import json
import os
import torch
from pathlib import Path
from typing import Optional, List, Dict, Annotated
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from functools import lru_cache
import random
import re
import shutil
from fastapi import Security, status
from fastapi.security import APIKeyHeader



# --- API Key Configuration ---
API_KEY_NAME = "X-API-Key"
api_key_header_auth = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

VALID_API_KEYS = {
    "your-super-secret-api-key-1": "api_consumer_alpha",
    "another-very-secure-key-for-beta": "api_consumer_beta",
    "pxyuxXUMG3m7SdzShMIBG8mVEerE0LakY5YEapG84H4": "kmu_image_team",
    "7mEwt6uLnaiqmTq5af3aPvrGYDIZ2oCdp59UiiNh3yw": "kmu_image_team",
    "o03xq_jku1cerv-LeV21MB7wL4QArmEGim2TGQhUcuI": "kmu_image_team",
    "wBT6XJiN1l5Vhc2Tx_d8r41sAFMsEgw-efCO6Y_fvKg": "kmu_image_team",
}

# --- Logging Configuration ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
# Ensure log level is a valid level string for basicConfig
valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if log_level not in valid_log_levels:
    print(f"Warning: Invalid LOG_LEVEL '{log_level}'. Defaulting to INFO.")
    log_level = "INFO"

logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def get_api_key_user(api_key: str = Security(api_key_header_auth)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated: X-API-Key header missing."
        )
    if api_key in VALID_API_KEYS:
        user_identifier = VALID_API_KEYS[api_key]
        logger.info(f"API Key validated for user: {user_identifier}")
        user_data_dir = get_user_chat_data_dir(user_identifier)
        user_data_dir.mkdir(parents=True, exist_ok=True)
        (user_data_dir / "chat_messages").mkdir(parents=True, exist_ok=True)
        user_feedback_dir = get_user_feedback_save_path(user_identifier)
        user_feedback_dir.mkdir(parents=True, exist_ok=True)
        user_qa_dir = get_user_qa_log_path(user_identifier)
        user_qa_dir.mkdir(parents=True, exist_ok=True)
        return user_identifier
    else:
        logger.warning(f"Invalid API Key received: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials: Invalid API Key."
        )

app = FastAPI(
    title="KMU Air Pollution RAG API",
    description="""
Welcome to the KMU Air Pollution Retrieval Augmented Generation (RAG) System API.
This API allows you to interact with our RAG system to get answers about air pollution topics
based on our knowledge base.

**Authentication:**
To use the public API endpoints (prefixed with `/api/v1/public`), you need an API Key.
Please include this key in the `X-API-Key` header of your requests.
If you need an API Key, please contact the system administrator.

**Session Management:**
For multi-turn conversations, the API will return a `session_id` in the response.
Include this `session_id` in subsequent requests to maintain context.
If no `session_id` is provided in a request, a new session will be initiated for the API user.

**Note:** The system is currently in testing and may experience instability. Please review output results carefully.
    """,
    version="1.0.0"
)
api_router = APIRouter(prefix="/api")

def detect_format_mode(question: str) -> str:
    format_triggers = [
        "請用一段話", "摘要", "表格", "表列", "條列式", "清單形式", "一句話", "說明就好",
        "summarize", "as a table", "one paragraph", "bullet points", "list format",
        "格式", "指定的格式", "指定格式", "以下格式", "下列格式", "這個格式", "這種格式",
        "我要的格式", "請用格式", "請用以下", "用以下格式"
    ]
    if any(re.search(r'\b' + re.escape(kw) + r'\b', question, re.IGNORECASE) for kw in format_triggers):
        logger.info(f"Format mode 'custom' triggered by keyword for question: '{question[:100]}'")
        return "custom"
    if re.search(r"(請用|使用|採用|依照|依據|照著|給我|我要).*格式", question, re.IGNORECASE):
        logger.info(f"Format mode 'custom' triggered by explicit format instruction for question: '{question[:100]}'")
        return "custom"
    if re.search(r"Question\s*:", question, re.IGNORECASE) and \
       re.search(r"Answers?\s*:", question, re.IGNORECASE) and \
       ("格式" in question or "請用" in question or "幫我" in question or "給我" in question or "我要的是" in question):
        logger.info(f"Format mode 'custom' triggered by 'Question:/Answers:' pattern with instructive verb for question: '{question[:100]}'")
        return "custom"
    if "簡單說明" in question:
        logger.info(f"Format mode 'custom' triggered by '簡單說明' for question: '{question[:100]}'")
        return "custom"
    logger.info(f"Format mode 'default' for question: '{question[:100]}'")
    return "default"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-Username", "X-API-Key"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

MAX_HISTORY_PER_SESSION = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding = None
vectordb = None
request_counters = {}
FEEDBACK_SAVE_PATH_BASE = Path("user_specific_feedback")
QA_LOG_PATH_BASE = Path("user_specific_qa_logs")
SAVE_QA = True
MAX_LLM_RETRIES = 1
USER_DATA_BASE_DIR = Path("user_specific_chat_data")

def sanitize_username(username: str) -> str:
    if not username: return "default_user"
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', username).lower()
    return sanitized if sanitized else "invalid_user"

def get_user_chat_data_dir(username: str) -> Path: return USER_DATA_BASE_DIR / sanitize_username(username)
def get_user_chats_metadata_file(username: str) -> Path: return get_user_chat_data_dir(username) / "chats_metadata.json"
def get_user_chat_messages_dir(username: str) -> Path: return get_user_chat_data_dir(username) / "chat_messages"
def get_user_feedback_save_path(username: str) -> Path: return FEEDBACK_SAVE_PATH_BASE / sanitize_username(username)
def get_user_qa_log_path(username: str) -> Path: return QA_LOG_PATH_BASE / sanitize_username(username)

async def get_current_username(x_username: Optional[str] = Header(None)) -> str:
    if x_username is None:
        logger.warning("請求中未找到 X-Username Header")
        raise HTTPException(status_code=400, detail="請求標頭中缺少 X-Username")
    sanitized = sanitize_username(x_username)
    if not sanitized or sanitized == "invalid_user":
        logger.warning(f"提供的用戶名 '{x_username}' 清理後無效")
        raise HTTPException(status_code=400, detail="提供的用戶名無效")
    get_user_chat_data_dir(sanitized).mkdir(parents=True, exist_ok=True)
    (get_user_chat_data_dir(sanitized) / "chat_messages").mkdir(parents=True, exist_ok=True)
    get_user_feedback_save_path(sanitized).mkdir(parents=True, exist_ok=True)
    get_user_qa_log_path(sanitized).mkdir(parents=True, exist_ok=True)
    return sanitized

embedding_model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
try:
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    _ = embedding.embed_query("測試嵌入模型")
except Exception as e:
    logger.error(f"❌ 嵌入模型載入失敗: {str(e)}", exc_info=True)
    embedding = None

@app.on_event("startup")
def startup_event():
    global vectordb, embedding_model_name
    logger.info(f"日誌級別設定為: {log_level}")
    logger.info(f"使用設備: {device}")
    if embedding is None:
        logger.error("❌ 嵌入模型未載入，無法初始化向量資料庫。")
    else:
        logger.info(f"正在載入嵌入模型: {embedding_model_name}")
        logger.info("✅ 嵌入模型載入並測試成功")
        try:
            persist_dir = os.environ.get("VECTORDB_PATH", "6_12")
            logger.info(f"正在從 '{persist_dir}' 載入向量資料庫...")
            if not os.path.exists(persist_dir):
                logger.error(f"❌ 向量資料庫目錄 '{persist_dir}' 不存在。")
            else:
                vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
                _ = vectordb.similarity_search("系統預熱", k=1)
                logger.info(f"✅ 向量資料庫從 '{persist_dir}' 載入並預熱完成")
        except Exception as e:
            logger.error(f"❌ 向量資料庫載入失敗: {str(e)}", exc_info=True)
            vectordb = None
    USER_DATA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_SAVE_PATH_BASE.mkdir(parents=True, exist_ok=True)
    QA_LOG_PATH_BASE.mkdir(parents=True, exist_ok=True)
    logger.info(f"用戶特定數據的基礎目錄已準備就緒: {USER_DATA_BASE_DIR}, {FEEDBACK_SAVE_PATH_BASE}, {QA_LOG_PATH_BASE}")


# --- Prompt Templates ---

# 主要版本
# STRUCTURED_LIST_PROMPT = PromptTemplate(
#     input_variables=["question", "context", "history", "format_mode"],
#     template="""
# You are a helpful assistant providing clear and structured information in **Traditional Chinese**. Your task is to answer the user's question based on the retrieved context below, adhering strictly to the specified format based on the `format_mode`.

# 📌 **Format Mode:** {format_mode}

# 🚨 **格式指令 (Format Instructions - 嚴格遵守):**

# *   **If `format_mode` is `custom`:** 代表使用者在問題中指定了格式。請 **完全遵循使用者要求的格式** 進行回答。如果使用者要求隱含了 Markdown 結構 (如列表)，請使用標準 Markdown (`* `, `1. `)。確保 `**粗體**` 使用兩個星號。
# *   **If `format_mode` is `default`:** 代表系統隨機選中了下述的「結構化列表風格」。你 **必須** 嚴格使用此風格，任何偏差都視為錯誤。
#     *   **最關鍵要求 (CRITICAL for default mode):** 章節標題 **必須** 使用 **兩個星號** 包裹 (格式為 `**一、中文標題**`)。**絕對禁止** 使用多餘的星號 (如 `***標題***` 或 `* **標題** *`) 或單個星號。**標題後的內容文字絕不能加粗。**

# 🎯 **預設回應格式 (Default Response Format - 結構化列表風格 - 僅在 `format_mode` 為 'default' 時使用):**

# 1.  **引言 (Introduction):** 在開頭提供一個簡短的引言 (1-2 句話)。
# 2.  **編號章節 (Numbered Sections):** 包含 2 至 5 個章節，使用 `**一、**`, `**二、**` 等作為標題前綴，標題本身需加粗，完整格式為 `**一、中文標題**`。
# 3.  **章節內容 (Content):** 每個章節標題下方撰寫 1-3 句話的內容。**內容必須是純文字 (plain text)**，不得加粗或使用斜體。語言需清晰易懂。可以在內容文字中 **少量** 使用相關的表情符號 (見下方建議)，但 **標題中禁止使用任何表情符號**。
# 4.  **分隔線 (Separator Line):** 在 **每個** 章節的內容文字結束後，**必須** 插入一行由 **100個** 半形句點 (`.`) 組成的分隔線，如下：
#     ....................................................................................................
# 5.  **間距 (Spacing):** 引言和第一個章節標題之間保留一個空行。每個分隔線和下一個章節標題之間也保留一個空行。
# 6.  **語言 (Language):** 全文使用 **繁體中文**。
# 7.  **內容來源 (Context Usage):** 回答內容 **僅能** 根據下方提供的 `Retrieved Context` 生成，不要提及「根據上下文」或直接複製上下文原文。
# 8.  **表情符號建議 (Emojis - 僅用於內容文字):** 💡, ✅, ⚠️, 📊, 👥, 🏫, 🌱, 🤝, ❤️ (或其他與內容相關的符號)

# 📘 **Conversation History:** {history}
# 📄 **Retrieved Context:** {context}
# ❓ **User Question:** {question}

# 👇 **請用繁體中文回答。根據偵測到的 `format_mode` 遵循對應的格式指令。若為 `default` 模式，請極度嚴格地遵守「結構化列表風格」的所有細節，特別是 `**標題**` 格式和純文字內容要求。**

# 📝 **輸出範例 (EXAMPLE OUTPUT FORMAT - when `format_mode` is "default"):**

# 小港空污USR計畫已展現出一定的成效 ✅，主要體現在提升社區居民對空污議題的認知、促進健康行為的改變以及建立社區參與的機制。

# **一、社區參與與教育推廣**
# 計畫團隊深入小港區的學校 🏫 與社區 👥，舉辦各類環境與健康教育活動，例如針對空污敏感族群的兒童氣喘衛教營隊，以及在鳳林國小舉辦的「空污健康站」環境教育嘉年華。這些活動吸引了數百名小港地區居民參與，成功將空污知識轉化為社區行動。
# .................................................................................................................................................................

# **二、針對特定族群的健康促進**
# 計畫針對空污敏感族群的兒童氣喘進行衛教，並擴及高齡族群，例如舉辦高齡者社區健康促進講座和長照據點合作的呼吸保健課程。這些活動提升了不同年齡層對空氣品質與健康風險的認知，並促進了健康行為的改變 ❤️。
# .................................................................................................................................................................

# **三、與企業合作的亮點**
# 計畫與小港醫院及地方企業合作 🤝 推動空污監測系統和ESG健康促進方案，顯示計畫在整合資源、促進社區發展方面的努力。
# .................................................................................................................................................................

# **四、計畫目標的落實與成果分享**
# 計畫申請書完整闡述了小港空污議題的背景、目標、執行方案與預期效益 📊，並定期提交進度與成果報告，確保計畫目標的落實，並通過多元方式進行成效評估與分享。
# .................................................................................................................................................................
# """
# )

# 測試v1
# 微調後的版本 (專為 default mode 優化)
STRUCTURED_LIST_PROMPT= PromptTemplate(
    input_variables=["question", "context", "history"], # 注意：移除了 format_mode，因為它已在應用層面處理
    template="""
你是一位嚴謹的資訊整理專家，你的任務是根據提供的資料，使用一種特殊的「結構化列表風格」，以繁體中文回答使用者的問題。

**--- 核心原則 (MUST FOLLOW) ---**
**0.  前置檢查 (Pre-computation Check): 在回答之前，先在腦中判斷提供的 `<CONTEXT>` 是否包含回答 `<QUESTION>` 的直接答案。如果完全沒有，則必須放棄格式，直接回答「根據所提供的資料，無法回答『[此處引用使用者的問題]』這個問題。」**

**--- 核心原則 (MUST FOLLOW) ---**
1.  **絕對忠於事實 (Strictly Grounded in Context):**
    - 你的所有回答、數據和結論 **必須** 且 **只能** 來自下方的 `<CONTEXT>` 區塊。
    - **嚴禁** 虛構、推測或使用任何外部知識。
2.  **坦承資訊不足 (Acknowledge Limits):**
    - 如果 `<CONTEXT>` 中的資訊不足以生成一個完整的章節，**必須** 跳過該章節。**絕不**能為了湊滿章節數量而編造內容。
3.  **直接切入主題 (Be Direct):**
    - **禁止** 任何與回答無關的開場白或結尾語。
    - **禁止** 提及資料來源（如「根據文本...」）。

**--- 輸出格式指南 (OUTPUT FORMATTING - 必須嚴格遵守) ---**
1.  **引言:** 以 1-2 句話的簡短引言開始。
2.  **章節標題:**
    - 使用 `**一、中文標題**` 的格式。標題本身必須用兩個星號 `**` 包裹。
    - **禁止** 在標題中使用任何表情符號。
3.  **章節內容:**
    - 在標題下方撰寫 1-3 句話的純文字內容。
    - **禁止** 對內容文字進行加粗或使用斜體。
    - 可以在內容中少量使用相關的 Emoji。
4.  **分隔線:**
    - 在每個章節內容結束後，**必須** 另起一行，插入由 100 個半形句點組成的分隔線。
    - 分隔線格式: `....................................................................................................`
5.  **間距:** 在引言、章節標題和分隔線之間，保持適當的空行以維持可讀性。

**--- 輸入資料 (INPUT DATA) ---**
<CONTEXT>
{context}
</CONTEXT>

<HISTORY>
{history}
</HISTORY>

<QUESTION>
{question}
</QUESTION>

**--- 輸出範例 (OUTPUT EXAMPLE) ---**
小港空污USR計畫已展現出一定的成效 ✅，主要體現在提升社區居民對空污議題的認知、促進健康行為的改變以及建立社區參與的機制。

**一、社區參與與教育推廣**
計畫團隊深入小港區的學校 🏫 與社區 👥，舉辦各類環境與健康教育活動，成功將空污知識轉化為社區行動。
....................................................................................................

**二、針對特定族群的健康促進**
計畫特別針對兒童與高齡者等敏感族群進行衛教，提升了不同年齡層對空氣品質與健康風險的認知，並促進了健康行為的改變 ❤️。
....................................................................................................

**三、跨域合作亮點**
計畫與小港醫院及地方企業合作 🤝，共同推動空污監測系統和ESG健康促進方案，展現了資源整合的努力。
....................................................................................................

👇 **請嚴格遵循以上所有規則，特別是「**標題**」格式和「100個句點分隔線」，直接開始撰寫你的回答:**
"""
)

# 主要版本
# HIERARCHICAL_BULLETS_PROMPT = PromptTemplate(
#     input_variables=["question", "context", "history", "format_mode"],
#     template="""
# You are a helpful assistant tasked with providing detailed, hierarchically structured information **in Traditional Chinese**. Your task is to answer the user's question based on the retrieved context below using **standard Markdown headings and lists** for structure. This style was randomly selected for the default mode.

# 📌 **Format Mode:** {format_mode} (System detected: 'default' - no specific user request in question)

# 🚨 **Format Instructions (Strictly Adhere to this Style):**

# *   Your response **MUST** be entirely in **Traditional Chinese**, structured using standard Markdown.
# *   **Headings:**
#     *   Use `# ` (one hash + space) for the optional **Main Title**.
#     *   Use `## ` (two hashes + space) for **Main Section Headers** (e.g., `## 一、Main Section Title`).
#     *   Use `### ` (three hashes + space) for **Sub-section Headers** (e.g., `### 1. Sub-section Title`).
# *   **Lists:**
#     *   Below headings, when itemization is needed, use standard Markdown **unordered lists** (`* ` or `- ` followed by a space) or **ordered lists** (`1. `, `2. ` followed by a space).
#     *   List item text must be **plain text starting immediately after the marker and space**. **CRITICAL: Absolutely DO NOT apply bold markdown (`**`) to the text immediately following the list marker (`* / - / 1. `) or any introductory label ending with a colon (like `Label:`).** Use `**bold emphasis**` very sparingly *only* for specific keywords deep *within* the list item text itself.
#     *   Maintain consistent list indentation (typically 2 or 4 spaces).
# *   **Spacing:** Ensure one blank line between the Main Title (if used) and the first Section Header, and one blank line between subsequent `##` Section Headers.
# *   **Critical Requirements:**
#     *   **Strictly adhere** to the Markdown syntax specified above, especially the **required spaces** after heading markers (`#`, `##`, `###`) and list markers (`*`, `-`, `1.`).
#     *   **Absolutely prohibit** non-standard formats or extra symbols.
#     *   **Bold (`**`)** is used *only* for emphasizing specific words within paragraphs or list items. **Do not bold entire headings or list items, and especially not the text immediately following a list marker or label.**
# *   **Context Usage:** Base your answer *only* on the provided `Retrieved Context`. **Absolutely prohibit mentioning** "According to the text," "The text indicates," "The text doesn't mention," or any phrases referencing the context source. State the information derived from the context directly.
# *   **No Preamble / Meta-commentary:** Start the response **directly** with the required formatted content (e.g., the `#` title or first `##` heading). **Do not** add introductory phrases like "Okay, here is the information..." or comments about the format itself like "Here is the response in hierarchical format:".
# *   **No Conversational Closing:** Do not add concluding remarks like "Hope this helps!" or follow-up questions.

# 📘 **Conversation History:**
# {history}

# 📄 **Retrieved Context:**
# {context}

# ❓ **User Question:**
# {question}

# 👇 **Respond entirely in Traditional Chinese.** Strictly follow the 'Hierarchical Markdown' format: use headings with spaces (`# /## /### `) and standard lists (`* / - / 1. `). **Ensure list item text immediately following the marker or a label ending in a colon is plain text, NOT bold.** Start directly with the formatted content.

# 📝 **Example Output Format (when `format_mode` is "default" and this style is chosen):**

# # 空氣污染教育成效評估方法

# 在台灣的大學社會責任（USR）計畫中，評估「空氣污染教育成效」是一項多面向的任務，應結合量化與質化指標，以全面了解教育活動對社區、學生與政策層面的影響。以下是幾種常見且建議使用的評估方式：

# ## 一、量化指標（Quantitative Evaluation）

# ### 1. 前後測問卷分析
# *   針對參與者（如學生、社區居民）在課程或活動前後進行知識、態度與行為意向測驗。
# *   比較其環境知識增長、對空污議題的**敏感度**提升。

# ### 2. 參與人數與活動場次
# *   統計實體或線上課程、講座、工作坊參與人次。
# *   長期追蹤是否有固定參與族群，或是否能觸及新對象。

# ### 3. 行為改變的指標
# *   如居民是否開始使用空氣品質監測器、改變交通工具使用習慣。
# *   學校是否推動校園綠化等。

# ### 4. 社群媒體與平台互動
# *   觀看次數、分享、留言、點讚數等可衡量資訊擴散成效。

# ## 二、質化指標（Qualitative Evaluation）

# ### 1. 深度訪談與焦點團體
# *   透過與學生、教師、居民及在地 NGO 訪談，瞭解空污教育帶來的觀念改變或生活實踐。
# *   瞭解參與者對教育內容的接受度與建議。

# ### 2. 學生與居民的反思紀錄或學習成果
# *   包含學習單、反思日誌、創作（如短片、海報）等。
# *   作為其內化成果的呈現。

# ### 3. 社區合作的實質成果
# *   如與地方政府、學校或社區合作建立空污監測站。
# *   共同提出改善建議等。

# ## 三、長期成效追蹤（Impact Tracking）

# ### 1. 政策影響力
# *   是否促成地方政府或學校在空污議題上的政策修訂。
# *   推動實作方案。

# ### 2. 社區意識抬頭與自發行動
# *   是否出現自主辦理相關活動。
# *   成立自救會或監督平台。

# ### 3. 跨領域與永續擴散
# *   評估是否能與其他 USR 團隊、研究單位或企業形成合作。
# *   將教育模式擴展至其他區域或主題。

# *(Note: The example output deliberately excludes concluding summaries or questions.)*
# """
# )

# 測試v1
# 微調後的版本
HIERARCHICAL_BULLETS_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
你是一位嚴謹的政策分析師，你的任務是根據提供的資料，以清晰、層級化的 Markdown 報告格式，用繁體中文回答使用者的問題。

**--- 核心原則 (MUST FOLLOW) ---**
1.  **絕對忠於事實 (Strictly Grounded in Context):**
    - 你的所有回答、數據和結論 **必須** 且 **只能** 來自下方的 `<CONTEXT>` 區塊。
    - **嚴禁** 虛構、推測或使用任何外部知識。
2.  **坦承資訊不足 (Acknowledge Limits):**
    - 如果 `<CONTEXT>` 中的資訊不足以回答問題的某個部分，**必須** 忽略該部分或明確說明「根據所提供的資料，此部分資訊不足」。**絕不**能為了報告的完整性而編造內容。
3.  **直接切入主題 (Be Direct):**
    - **禁止** 任何開場白（如「好的，這是您要的報告」）或結尾語（如「希望這有幫助」）。
    - **禁止** 提及資料來源（如「根據文本...」）。直接陳述事實。

**--- 輸出格式指南 (OUTPUT FORMATTING) ---**
- 使用標準的 Markdown 標題 (`#`, `##`, `###`) 和列表 (`*`, `-`, `1.`) 來組織內容。
- 確保結構清晰，邏輯連貫。
- 粗體 (`**`) 僅用於段落或列表項目內部的特定關鍵詞強調。

**--- 輸入資料 (INPUT DATA) ---**
<CONTEXT>
{context}
</CONTEXT>

<HISTORY>
{history}
</HISTORY

<QUESTION>
{question}
</QUESTION>

**--- 輸出範例 (OUTPUT EXAMPLE) ---**
# 空氣污染教育成效評估方法
在評估「空氣污染教育成效」時，應結合量化與質化指標，以全面了解教育活動的影響。

## 一、量化指標
*   **前後測問卷分析:** 比較參與者在活動前後的知識、態度與行為意向變化。
*   **參與數據:** 統計活動參與人次、場次，並追蹤參與者的持續性。
*   **行為改變指標:** 觀察居民是否採納如使用監測器、改變交通工具等實際行動。
*   **線上互動數據:** 分析社群媒體的觀看、分享、留言數，以衡量資訊擴散效果。

## 二、質化指標
*   **深度訪談與焦點團體:** 了解參與者在觀念上的實質改變與對內容的建議。
*   **學習成果展現:** 分析學生的反思日誌、創作作品，作為其內化成果的證據。
*   **社區合作成果:** 評估是否產生具體的合作項目，如共建監測站或提出改善建議。

## 三、長期成效追蹤
*   **政策影響力:** 檢視活動是否促成地方或校園政策的修訂。
*   **社區自發行動:** 觀察是否出現由社區自主發起的相關活動或監督組織。
*   **跨域合作擴散:** 評估此教育模式是否成功擴展至其他團隊或區域。

👇 **請嚴格遵守以上所有規則，直接開始撰寫你的層級式分析報告:**
"""
)

# 主要版本
# PARAGRAPH_EMOJI_LEAD_PROMPT = PromptTemplate(
#     input_variables=["question", "context", "history", "format_mode"],
#     template="""
# You are a helpful assistant providing clear, paragraph-based explanations in **Traditional Chinese**. Your task is to answer the user's question based on the retrieved context below, using a specific style where each paragraph starts with a relevant emoji. This style was randomly selected for the default mode.

# 📌 **Format Mode:** {format_mode} (System detected: 'default' - no specific user request in question)

# 🚨 **格式指令 (嚴格遵守此風格):**

# *   你的回答 **必須** 是一系列 **繁體中文** 的段落。
# *   **最關鍵的是：每個段落都必須以 `一個相關的表情符號` + `一個空格` 開頭。** 表情符號應與該段落的主題相關。
# *   **表情符號和空格之後的文字內容，必須是純文字 (plain text)。** 絕對不要自動將這部分文字加粗。
# *   **完全避免** 在回答中使用任何編號列表 (`1.`, `2.`)、項目符號列表 (`*`, `-`)、章節標題 (`#`, `##`, `**標題**`) 或分隔線 (`---`, `...`)。專注於純段落結構。
# *   如果需要在段落 *內部* 強調特定關鍵字，可以 **非常少量地** 使用標準 Markdown 粗體 (`**強調詞**`)。**切勿** 將表情符號後的第一個完整句子加粗。

# 💡 **表情符號建議 (根據段落主題選擇，也可使用其他相關符號):**
#     💡, 🤝, 🏥, 👥, 📚, 🌱, 🔬, 🧭, ✅, ⚠️, 📊, 🏫

# 📘 **Conversation History:** {history}
# 📄 **Retrieved Context:** {context}
# ❓ **User Question:** {question}

# 👇 **請用繁體中文回答。嚴格遵循「段落前置表情符號」格式：每個段落以單一表情符號+空格開頭，後面接續純文字。**

# 📝 **輸出範例 (當 `format_mode` 為 "default" 且選中此風格時):**

# 💡 在USR（大學社會責任）計畫中，與醫療機構的合作是推動社區健康促進的重要一環。該計畫通過多種方式來確保居民能夠獲得持續且有效的健康管理服務。

# 🏥 首先，USR計畫與地方政府及醫療單位合作設立健康檢查站，並定期進行監測。這些健康檢查站不僅提供基本的體檢服務，還包括針對空氣污染等環境因素對健康影響的專門評估。

# 👥 此外，USR計畫也組織聯合衛教活動和社區實踐項目，促使校園與社區形成緊密互動與協同發展。這些活動旨在全方位提升居民的生活品質和健康水平。

# 🤝 為了進一步推動社區健康促進，USR計畫還強調了與醫療機構在資源整合上的重要性。這包括利用學術研究的力量來開發新的健康產品和服務。

# ✅ 總之，USR計畫透過與醫療機構的合作，從多個層面推動社區健康促進工作，確保了居民能夠獲得全面且有效的健康管理服務。
# """
# )

# 測試v1
# 微調後的版本
PARAGRAPH_EMOJI_LEAD_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
你是一位友善且樂於助人的助理，你的任務是使用一種特定的「表情符號引導段落」風格，以繁體中文回答使用者的問題。

**--- 核心輸出規則 (MUST FOLLOW) ---**
1.  **表情符號開頭 (Emoji Prefix):** 你的回答由數個段落組成。**每一個段落都必須以一個與該段內容主題相關的表情符號 (Emoji) 和一個空格開頭。**
2.  **純段落結構 (Paragraphs Only):** 只使用段落來組織你的回答。
3.  **內容基於事實 (Grounded in Context):** 回答的內容必須基於下方 `<CONTEXT>` 提供的資料。

**--- 禁止事項 (ABSOLUTELY PROHIBITED) ---**
*   **禁止** 使用任何數字列表 (`1.`, `2.`) 或項目符號 (`*`, `-`)。
*   **禁止** 使用任何 Markdown 標題 (`#`, `##`) 或分隔線 (`---`)。
*   **禁止** 在段落開頭的表情符號和空格後，對整個句子或長句進行粗體強調。

**--- 輸入資料 (INPUT DATA) ---**
<CONTEXT>
{context}
</CONTEXT>

<HISTORY>
{history}
</HISTORY>

<QUESTION>
{question}
</QUESTION>

**--- 輸出範例 (OUTPUT EXAMPLE) ---**
💡 在USR（大學社會責任）計畫中，與醫療機構的合作是推動社區健康促進的重要一環。該計畫通過多種方式來確保居民能夠獲得持續且有效的健康管理服務。

🏥 首先，USR計畫與地方政府及醫療單位合作設立健康檢查站，並定期進行監測。這些健康檢查站不僅提供基本的體檢服務，還包括針對空氣污染等環境因素對健康影響的專門評估。

👥 此外，USR計畫也組織聯合衛教活動和社區實踐項目，促使校園與社區形成緊密互動與協同發展。這些活動旨在全方位提升居民的生活品質和健康水平。

✅ 總之，USR計畫透過與醫療機構的合作，從多個層面推動社區健康促進工作，確保了居民能夠獲得全面且有效的健康管理服務。

👇 **請嚴格遵循以上規則，以「表情符號引導段落」的風格開始回答:**
"""
)

# 主要版本
# CUSTOM_FORMAT_BASE_PROMPT = PromptTemplate(
#     input_variables=["question", "context", "history", "format_mode"],
#     template="""
# You are a highly obedient and meticulous assistant providing information in **Traditional Chinese**. The user has asked a question and *explicitly included instructions on the desired response format* within their question text. Your SOLE and ABSOLUTE purpose is to follow these format instructions PRECISELY and LITERALLY, without any deviation or addition of your own styling.

# 📌 **Format Mode:** {format_mode} (System detected: 'custom' - user-specified format in question)

# 🚨 **絕對核心指令 - 必須一字不差地、完全排他地遵循 (ABSOLUTE CORE INSTRUCTIONS - MUST BE FOLLOWED LITERALLY AND EXCLUSIVELY):**

# 1.  **完全複製使用者指定的字面格式 (Replicate User's Literal Format EXACTLY)**:
#     Carefully analyze the **User Question** below. Your **ONLY task** is to **EXACTLY replicate the user's specified output format, including all labels, colons, newlines, spacing, and casing as GIVEN BY THE USER.**
#     *   Example: If user specifies `Question: [question_content]\nAnswers: [answer_content]`, your output MUST be in this exact structure.

# 2.  **針對多個項目 (例如多組QA) - 關鍵要求 (CRITICAL for Multiple Items, e.g., multiple QAs)**:
#     If the user requests multiple items (e.g., "給我20組QA", "列出5個優點"):
#     *   **EACH AND EVERY item/QA pair generated MUST independently and completely adhere to the user's specified format.**
#     *   **DO NOT** change the format or omit format labels (like `Question:`) after the first item.
#     *   **DO NOT** insert ANY extra text, numbering (unless specified by user), emojis, or separators BETWEEN the formatted items, unless explicitly part of the user's requested format. Each formatted item should follow the previous one directly, respecting newlines if specified in the format.

# 3.  **禁止任何額外內容或風格 - 絕對排他性 (Prohibit ALL Extra Content or Styles - Absolute Exclusivity)**:
#     *   **ABSOLUTELY NO** conversational preambles, introductions, explanations, or meta-commentary (e.g., "好的，這是您要的格式：" or "以下是N組QA：").
#     *   Your response **MUST START DIRECTLY** with the user's requested formatted content.
#     *   **COMPLETELY IGNORE AND OVERRIDE ALL** default styling or formatting habits from any other prompts or your general training (e.g., NO emojis at the start of paragraphs, NO specific Markdown headers unless user asked for them, NO default list styles if user specified something else).
#     *   In this `custom` mode, **THE USER'S LITERAL SPECIFIED FORMAT IS THE ONLY AND EXCLUSIVE STANDARD. THERE ARE NO OTHER RULES.**

# 4.  **內容生成 (Content Generation)**: While strictly adhering to the format, generate the content (questions, answers) based on the provided `Retrieved Context` and `Conversation History`. If the user asks for in-depth questions or specific topics, strive to meet those content requirements within the rigid format.

# 📘 **Conversation History:** {history}
# 📄 **Retrieved Context:** {context}

# ❓ **User Question:**
# {question}

# 👇 **請用繁體中文回答。請「絕對嚴格地、一字不差地、排他地」遵循使用者在問題中指定的字面格式。特別是當要求多個項目時，確保每個項目都獨立且完整地遵循該格式。不要添加任何您自己的文字、標籤、表情符號或風格。直接開始輸出使用者要求的格式。**

# 📝 **重要範例 - 使用者指定字面格式並要求多組QA (CRITICAL EXAMPLE - User specifies literal format for MULTIPLE QAs):**
# 假設使用者的問題是: "給我2組有關USR計畫的QA，請用以下格式並且每個QA都要有深度：\nQuestion: [此處放問題]\nAnswers: [此處放答案]"

# 你的輸出 **必須** 直接是 (沒有任何其他文字在前面或中間，每個QA獨立成對，並嚴格遵守換行):
# Question: USR計畫在促進大學與社區夥伴關係的長期永續性方面，除了資金考量外，還面臨哪些核心挑戰？應如何從大學治理與社區培力角度著手應對？
# Answers: 核心挑戰包括：一、社區夥伴的期望管理與大學能量的落差；二、計畫結束後社區自主運作能力的建構；三、大學內部跨單位協作的制度性障礙。大學治理層面應建立更彈性的USR專案支持機制與成果認定標準。社區培力則需著重於知識轉移、在地人才培育，以及協助建立社區自主組織。
# Question: 在評估USR計畫的社會影響力時，除了量化指標（如參與人數、滿意度），有哪些更具深度的質化評估方法能夠捕捉計畫對社區結構與關係網絡帶來的細微但長遠的改變？
# Answers: 深度質化評估方法可包括：一、參與式行動研究(PAR)，讓社區成員共同參與評估過程；二、社會網絡分析(SNA)，分析計畫前後社區內部及大學與社區間的互動網絡變化；三、敘事探究，收集並分析社區成員關於計畫影響的個人生命故事與集體記憶；四、運用「最顯著改變」技術(Most Significant Change)，從眾多故事中篩選並討論最具代表性的影響案例。

# (如果使用者要求的格式是 `Q: [問題]\nA: [答案]`，或者其他任何格式，你就必須一字不差地使用該格式。如果要求10組QA，就生成10組，每組都嚴格遵循。)
# """
# )

# 測試v1
# 微調後的版本
CUSTOM_FORMAT_BASE_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
你是一個精確且嚴格遵循指令的助理，你的唯一任務是根據使用者在問題中指定的格式生成內容。

**--- 核心指令 (CORE INSTRUCTIONS) ---**

**1. 最高原則：完全複製使用者指定的字面格式 (Prime Directive: Replicate User's Literal Format)**
   - 你的輸出 **必須** 完全複製使用者在 `<QUESTION>` 中指定的格式。
   - 這包括所有的標籤、標點符號、換行、空格和大小寫。
   - **禁止** 任何形式的自我發揮或添加。你的輸出 **必須** 直接以使用者指定的格式開始。

**2. 處理多個項目時的規則 (Rule for Multiple Items)**
   - 如果使用者要求生成多個項目（例如「給我 N 組 QA」），則 **每一個** 項目都 **必須** 獨立且完整地遵循使用者指定的格式。
   - **禁止** 在項目之間添加任何額外的文字、編號或分隔符（除非使用者格式中已包含）。

**3. 內容生成 (Content Generation)**
   - 在嚴格遵守格式的前提下，使用 `<CONTEXT>` 和 `<HISTORY>` 的資訊來生成回答的內容。

**--- 輸入資料 (INPUT DATA) ---**
<CONTEXT>
{context}
</CONTEXT>

<HISTORY>
{history}
</HISTORY>

<QUESTION>
{question}
</QUESTION>

**--- 範例 (EXAMPLE) ---**
# 假設使用者的問題是: "請給我2組QA，格式如下：
# Question: [問題]
# Answer: [答案]"

# 你的輸出必須是，且只能是：
Question: [這裡根據CONTEXT生成第一個問題]
Answer: [這裡根據CONTEXT生成第一個答案]
Question: [這裡根據CONTEXT生成第二個問題]
Answer: [這裡根據CONTEXT生成第二個答案]

# 如果使用者指定的格式是 `Q: [問題] A: [答案]`，你就必須使用該格式。

👇 **請嚴格遵循以上所有規則，直接輸出使用者要求的格式化內容:**
"""
)

#主要版本
# RESEARCH_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["question", "context", "history", "format_mode"],
#     template="""
# You are a policy analyst and academic writer providing an evaluation in **Traditional Chinese**. Your output should be suitable for a formal academic or governmental report, using **Markdown headings** for structure.

# 📌 **Format Mode:** {format_mode}

# 🚨 **Formatting Guidelines (嚴格遵守):**

# *   **If `format_mode` is `custom`:** This means the user specified a custom output structure in their question. Follow the user's requested formatting **exactly**, maintaining a formal and academic tone. Use standard Markdown (`#`, `##`, `###`, `* list item`, `**bold**`) as appropriate only if explicitly part of the user's requested format or if the user's request clearly implies standard Markdown (e.g., "give me a list"). **Prioritize the user's literal format over adding Markdown not explicitly requested.**
# *   **If `format_mode` is `default`:** Follow the standard structured report format using **Markdown headings** as described below. Strictly adhere to the formatting rules.

#     1.  **主標題 (Main Title):** 使用 `#` (一個井號) 作為整個報告的主標題 (若適用)。
#     2.  **主要章節標題 (Section Headers):** 使用 `##` (兩個井號) 標示主要章節 (例如: `## 一、成效亮點`)。
#     3.  **次要章節/小節標題 (Sub-section Headers):** 使用 `###` (三個井號) 標示次要章節或小節 (例如: `### 1. PM2.5 濃度變化`)，視需要使用。
#     4.  **內容 (Content):** 在標題下方，使用清晰的段落文字。
#     5.  **列表 (Lists):** 若需條列，請使用標準 Markdown 列表，以 `* ` 或 `- ` 開頭 (例如: `* 項目一`)。列表項目本身應為**純文字**，除非特定關鍵字需要**粗體** (`**強調詞**`)。**不要將整個列表項目加粗**。
#     6.  **間距 (Spacing):** 在主標題和第一個章節標題之間、以及各個章節標題 (`##`) 之間保留一個空行。
#     7.  **語言與風格 (Language & Tone):** 全文使用**繁體中文**。語氣必須正式、客觀、學術。**禁止**使用表情符號 (emojis)、非標準的 Markdown 樣式 (除了指示的 `#`, `##`, `###`, `*`, `**`) 或口語化/非正式語言。

# 📘 **Conversation History:**
# {history}

# 📘 **Document Context:**
# {context}

# 🧾 **User's Question:**
# {question}

# 👇 **請用繁體中文回答:**
# *   若 `format_mode` 是 `custom`，請完全遵循使用者在問題中定義的字面格式。**不要自行添加任何 Markdown 結構，除非使用者明確要求或其要求明顯暗示了標準 Markdown 結構。**
# *   若 `format_mode` 是 `default`，請嚴格遵守上述使用 **Markdown 標題 (`#`, `##`, `###`)** 的結構化報告格式。**粗體 (`**`) 僅用於內文特定詞語強調，不可用於標題或整個列表項目。**

# 📝 **EXAMPLE OUTPUT FORMAT (when `format_mode` is "default"):**
# # 高雄小港空氣污染議題分析報告
# ## 一、成效亮點
# 高雄小港區為台灣重要的工業重鎮，過去長期遭受石化業與重工業排放所帶來的空氣污染。自政府實施空污防制強化方案以來，已逐步見到成果：
# *   PM2.5 年均濃度下降：2023年，小港區PM2.5濃度首次降至15μg/m³以下，符應國家標準。
# *   高污染工廠改善：多家高污染事業完成鍋爐設備更新或污染防制設施強化。
# *   在地參與提升：透過社區論壇、校園教育及USR協作，小港居民參與空品改善活動人數顯著提升，展現地方共治的潛能。
# 這些成果說明政策具備初步效益，也顯示社區力量在環境治理中日益關鍵。
# ## 二、主要挑戰與限制
# 儘管已有顯著進展，小港區的空氣污染問題仍存在下列挑戰：
# *   結構性污染源持續存在：小港擁有多個大型石化工業區，污染物總量基數高，使得單一改善措施難以對空品產生劇烈改變。
# *   地形與氣候劣勢：背風側效應與氣候條件使污染物易滯留，加重局部污染濃度。
# *   政策協作落差：中央與地方在污染熱區資料整合與應變作為上，仍顯不足，導致反應時間延遲、缺乏即時調控力。
# *   公民動能不足：部分居民對空污議題已產生「習慣性麻痺」，缺乏主動監督與行動參與。
# ## 三、政策建議與改進方向
# 為有效深化治理成效，建議可從以下三個策略推動：
# ### 1. 擴大感測與資訊公開
# *   建立高密度感測網，強化移動式監測。
# *   發展即時污染視覺化平台，提升公眾風險意識。
# ### 2. 產業轉型與排放總量控管
# *   推行「總量管制 + 差別費率」，引導污染業者升級。
# *   鼓勵潔淨能源使用與碳排透明揭露。
# ### 3. 強化社區共治與環境教育
# *   整合USR計畫與地方課程，發展空污資料解讀與倡議能力。
# *   建立社區參與制度，如居民空品會議、政策參與管道等。
# ## 四、總體觀察與評論
# 小港空污問題的治理難度源自結構性深層因素，短期內難以徹底逆轉。然而，已有跡象顯示，只要政策持續推進並強化地方共治，將有機會轉危為機。特別是結合**科技監測**與**民眾行動力**，能構築出一套適合台灣重工業都市的永續治理模式。
# ## 五、結論
# 高雄小港的空污改善歷程是台灣工業區環境治理的縮影。持續落實「科技導向 + 公民參與 + 法制改革」三位一體的策略，將是未來提升空品與健康福祉的關鍵方向。
# """
# )

#測試v1
# 重構後的版本
RESEARCH_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
你是一位專精於台灣空污議題的政策分析師與學術寫手。你的任務是根據提供的資料，以客觀、專業的繁體中文回答使用者的問題。

**--- 核心原則 (MUST FOLLOW) ---**
1.  **事實至上:** 所有回答、數據和結論 **必須** 完全基於下方 `<CONTEXT>` 區塊的內容。**嚴禁** 虛構、推測或使用任何外部知識。
2.  **知之為知之:** 如果 `<CONTEXT>` 資訊不足，**必須** 明確回答「根據所提供的資料，無法回答此問題」。
3.  **專業語氣:** 保持客觀、學術的風格，不使用口語或表情符號。

**--- 輸入資料 (INPUT DATA) ---**
<CONTEXT>
{context}
</CONTEXT>

<HISTORY>
{history}
</HISTORY>

<QUESTION>
{question}
</QUESTION>

**--- 輸出格式指南 (OUTPUT FORMATTING) ---**
Format Mode: {format_mode}

*   **如果 `format_mode` 是 `custom`:**
    嚴格遵循使用者在 `<QUESTION>` 中定義的格式。除非使用者明確要求，否則不要自行添加 Markdown 標題（`#`, `##`）或列表（`*`）。

*   **如果 `format_mode` 是 `default`:**
    請使用標準的 Markdown 報告結構。粗體(`**`)僅用於內文關鍵詞強調。

    **範例格式 (EXAMPLE FOR "default" MODE):**
    # [報告主標題]
    ## 一、[章節標題一]
    段落內容...
    * 列表項目一
    * 列表項目二

    ## 二、[章節標題二]
    ### 1. [次章節標題]
    段落內容...

    ## 三、結論
    段落內容...

👇 **請嚴格遵守以上所有規則，開始撰寫你的分析報告:**
"""
)
DEFAULT_PROMPT_OPTIONS = [
    STRUCTURED_LIST_PROMPT,
    HIERARCHICAL_BULLETS_PROMPT,
    PARAGRAPH_EMOJI_LEAD_PROMPT
]

common_llm_config = { "temperature": 0.1, "top_p": 0.8 }
SUPPORTED_MODELS = [
    "qwen3:14b", "gemma3:12b", "gemma3:12b-it-q4_K_M", "qwen2.5:14b-instruct-q5_K_M",
    # 為了測試，可以加入一個較小的模型
    "llama3:8b", "qwen:4b"
]
DEFAULT_MODEL = "gemma3:12b" # Or your preferred default

@lru_cache(maxsize=5)
def get_model(model_name: str) -> Optional[OllamaLLM]:
    if model_name not in SUPPORTED_MODELS:
        logger.warning(f"⚠️ 請求的模型 '{model_name}' 不在支援列表，將使用預設模型 '{DEFAULT_MODEL}'。")
        model_name = DEFAULT_MODEL
    try:
        logger.info(f"⏳ 正在載入或獲取緩存的模型: {model_name}...")
        # 設定一個合理的超時時間，例如10分鐘
        model = OllamaLLM(model=model_name, **common_llm_config, request_timeout=600.0)
        # 預熱測試，如果這裡就失敗，表示模型載入有問題
        _ = model.invoke("請用繁體中文做個簡短的自我介紹")
        logger.info(f"✅ 模型 {model_name} 載入並測試成功")
        return model
    except Exception as e:
        logger.error(f"❌ 載入或測試模型 {model_name} 失敗: {str(e)}", exc_info=True)
        logger.error(f"💡 提示：模型載入失敗可能是因為 VRAM/RAM 不足，或模型檔案損毀。請嘗試：1. 重新啟動 Ollama 服務。 2. 執行 'ollama pull {model_name}' 重新下載模型。 3. 嘗試更小的模型（如 llama3:8b）。")
        return None

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if "/chat" in str(request.url) or "/api/v1/public/rag/ask" in str(request.url):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        request_timestamps = request_counters.get(client_ip, [])
        valid_timestamps = [t for t in request_timestamps if current_time - t < 60]
        rate_limit_count = 30
        if len(valid_timestamps) >= rate_limit_count:
            logger.warning(f"🚦 速率限制觸發: IP {client_ip} for URL {request.url}")
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429, content={"error": "請求過於頻繁"})
        valid_timestamps.append(current_time)
        request_counters[client_ip] = valid_timestamps
    return await call_next(request)

class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: Optional[str] = DEFAULT_MODEL
    prompt_mode: str = "default"

class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    model: str
    original_answer: str
    user_expected_question: Optional[str] = None
    user_expected_answer: str

class ChatListItem(BaseModel): id: str; title: str; updated_at: str
class NewChatRequest(BaseModel): id: str; title: str
class RenameChatRequest(BaseModel): title: str
class Message(BaseModel): role: str; content: str

def _load_user_chats_metadata(username: str) -> Dict[str, Dict[str, str]]:
    metadata_file = get_user_chats_metadata_file(username)
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f: return json.load(f)
        except Exception as e: logger.error(f"❌ 無法為用戶 {username} 加載聊天元數據: {e}", exc_info=True)
    return {}

def _save_user_chats_metadata(username: str, metadata: Dict[str, Dict[str, str]]):
    metadata_file = get_user_chats_metadata_file(username)
    try:
        sorted_metadata = dict(sorted(metadata.items(), key=lambda item: item[1].get('updated_at', '1970-01-01T00:00:00Z'), reverse=True))
        with open(metadata_file, "w", encoding="utf-8") as f: json.dump(sorted_metadata, f, ensure_ascii=False, indent=2)
        logger.debug(f"用戶 {username} 的聊天元數據已保存到 {metadata_file}")
    except Exception as e: logger.error(f"❌ 保存用戶 {username} 的聊天元數據失敗: {e}", exc_info=True)

def _get_user_chat_messages(username: str, chat_id: str) -> List[Dict[str, str]]:
    message_file = get_user_chat_messages_dir(username) / f"{chat_id}.json"
    if message_file.exists():
        try:
            with open(message_file, "r", encoding="utf-8") as f: return json.load(f)
        except Exception as e: logger.error(f"❌ 無法讀取用戶 {username} 聊天 {chat_id} 的訊息: {e}", exc_info=True)
    return []

def _save_user_chat_messages(username: str, chat_id: str, messages: List[Dict[str, str]]):
    message_file = get_user_chat_messages_dir(username) / f"{chat_id}.json"
    try:
        with open(message_file, "w", encoding="utf-8") as f: json.dump(messages, f, ensure_ascii=False, indent=2)
        logger.debug(f"用戶 {username} 聊天 {chat_id} 的訊息已保存到 {message_file}")
    except Exception as e: logger.error(f"❌ 保存用戶 {username} 聊天 {chat_id} 的訊息失敗: {e}", exc_info=True)

# MODIFIED: post_process_answer with refined Markdown handling for default mode
# MODIFIED: post_process_answer with EXTREMELY conservative Markdown handling for default mode
def post_process_answer(answer: str, format_mode: str = "default") -> str:
    original_answer = answer
    processed_answer = answer  # Start with the original answer

    try:
        # 1. Strip leading/trailing whitespace from the whole answer FIRST
        processed_answer = processed_answer.strip()

        # --- ADDED: Remove LLM preamble/explanatory phrases ---
        # These phrases are typically at the beginning of a response or a new paragraph.
        # We use re.IGNORECASE for case-insensitivity and re.UNICODE.
        llm_explanatory_phrases = [
            r"^\s*根據提供的資訊(?:內容)?(?:，|：|,|:)?\s*",
            r"^\s*根據你提供的資訊(?:，|：|,|:)?\s*",
            r"^\s*根據提供的文本(?:內容)?(?:，|：|,|:)?\s*",
            r"^\s*根據提供的上下文(?:，|：|,|:)?\s*",
            r"^\s*根據文檔(?:內容)?(?:，|：|,|:)?\s*",
            r"^\s*根據以上資訊(?:，|：|,|:)?\s*",
            r"^\s*從提供的資料來看(?:，|：|,|:)?\s*",
            r"^\s*資料顯示(?:，|：|,|:)?\s*",
            r"^\s*文本中提到(?:，|：|,|:)?\s*",
            r"^\s*文件中說明(?:，|：|,|:)?\s*",
            r"^\s*雖然資訊(?:中)?(?:並未|沒有)(?:明確)?(?:列出|指出|提及)(?:，|：|,|:)?\s*",
            r"^\s*雖然提供的資料顯示(?:，|：|,|:)?\s*",
            r"^\s*是的，根據資料(?:，|：|,|:)?\s*",
            r"^\s*好的，根據您的問題和提供的資料(?:，|：|,|:)?\s*",
            r"^\s*在提供的資料中(?:，|：|,|:)?\s*",
            r"^\s*從上下文中我們可以得知(?:，|：|,|:)?\s*",
            r"^\s*根據上下文(?:，|：|,|:)?\s*",
            r"^\s*文中(?:並未|沒有)明確提及(?:，|：|,|:)?\s*",
            r"^\s*以下是.*?的回答：\s*",
            r"^\s*針對您的問題(?:，|：|,|:)?\s*",
            r"^\s*回答如下(?:，|：|,|:)?\s*",
            # Add more patterns as observed
        ]

        # Apply removal of explanatory phrases
        # We loop a few times in case one removal uncovers another preamble.
        for _ in range(3): # Loop a few times to catch nested/sequential preambles
            cleaned_this_iteration = False
            for phrase_pattern in llm_explanatory_phrases:
                # Use re.match to ensure it's at the beginning of the current processed_answer
                match = re.match(phrase_pattern, processed_answer, flags=re.IGNORECASE | re.UNICODE)
                if match:
                    new_answer = processed_answer[match.end():].lstrip(" ，。、；：:,.")
                    if new_answer != processed_answer:
                        processed_answer = new_answer
                        cleaned_this_iteration = True
            if not cleaned_this_iteration:
                break # No changes in this iteration, so stop
        # --- ADDED END ---

        if format_mode == "custom":
            # --- Custom Mode: Minimal cleaning after preamble removal ---
            # Remove common LLM preambles if they slip through (might be redundant now but safe)
            processed_answer = re.sub(
                r"^\s*(好的，這是您要求的格式：|好的，這就為您提供：|根據您的要求，格式如下：|好的，這就為您呈現：|以下是符合您要求的格式：)\s*",
                "", processed_answer, flags=re.IGNORECASE | re.MULTILINE
            ).lstrip()

            if "Question:" in original_answer and "Answers:" in original_answer:
                match_q = re.search(r"Question\s*:", processed_answer, re.IGNORECASE)
                if match_q and match_q.start() > 0:
                    preamble_candidate = processed_answer[:match_q.start()]
                    if len(preamble_candidate) < 100 and "以下" not in preamble_candidate and "：" not in preamble_candidate[-5:]:
                        logger.debug(f"Custom format: Stripping potential preamble: '{preamble_candidate}'")
                        processed_answer = processed_answer[match_q.start():]
                elif match_q and match_q.start() == 0:
                    pass
                else:
                    # If preambles were removed above, this warning might be less frequent
                    logger.warning(f"Custom format: 'Question:' expected but not found at start. Processed: '{processed_answer[:100]}...'")

            processed_answer = re.sub(r'\n{3,}', '\n\n', processed_answer).strip()
            # Log for custom mode if changes were made (excluding initial strip)
            if original_answer.strip() != processed_answer: # Compare after initial strip of original
                 log_message_custom = f"Custom format post-processing. Orig len: {len(original_answer.strip())}, Proc len: {len(processed_answer)}. Starts with: '{processed_answer[:100]}...'"
                 logger.info(log_message_custom)
            else:
                 logger.debug(f"Custom format post-processing made no significant changes beyond initial strip.")
            return processed_answer or original_answer.strip() # Ensure original is also stripped for comparison

        # --- Default Mode: Extremely conservative cleaning after preamble removal ---

        # 2. Remove known LLM artifacts/template leakage (usually safe)
        #    This should run AFTER preamble removal.
        markers_to_remove = [
            "📘 Conversation History:", "📄 Retrieved Context:", "❓ User Question:",
            "👇 Please write your answer", "📝 EXAMPLE OUTPUT FORMAT",
            "You are a helpful assistant", "You are a policy analyst",
            "📌 **Format Mode:**"
        ]
        for m in markers_to_remove:
            if m in processed_answer:
                processed_answer = processed_answer.split(m, 1)[0].strip()

        # 3. Normalize multiple newlines to a maximum of two (usually safe for Markdown)
        processed_answer = re.sub(r'\n{3,}', '\n\n', processed_answer)

        # 4. Normalize bold syntax (keep this conservative)
        processed_answer = re.sub(r'\*{3,}(?P<content>.+?)\*{3,}', r'**\g<content>**', processed_answer)
        processed_answer = re.sub(r'\*{3,}', '**', processed_answer)
        processed_answer = re.sub(r'\*\*\s*(?P<content>.*?)\s*\*\*', r'**\g<content>**', processed_answer)
        # The following emoji/specific term bolding rules were part of the previous attempt to fix * display issues.
        # Since the primary issue is now about removing preambles, and frontend handles Markdown rendering,
        # these backend regexes for altering * to ** might be too aggressive or conflict.
        # Consider removing or heavily testing them if they cause unintended side effects.
        # For now, I'm commenting them out to ensure max conservatism from backend on MD structure.
        # processed_answer = re.sub(r'([🏭🚗💨🌍💡🚨📊📈🔍📝])\*(?P<content>[^\*]+?)\*', r'\1**\g<content>**', processed_answer)
        # processed_answer = re.sub(r'(^|\n)\*(?P<content>[^\*:]+?(?:源|類|法|策略|措施|評估)[^\*]*?)\*(?=:|：|\s|$)', r'\1**\g<content>**', processed_answer, flags=re.MULTILINE)


        # 5. Remove trailing whitespace from each line (generally safe)
        processed_answer = "\n".join([line.rstrip() for line in processed_answer.splitlines()])

        # 6. Final strip of leading/trailing whitespace from the whole answer
        processed_answer = processed_answer.strip()


        if original_answer.strip() != processed_answer:
            log_message_default = f"Default format (conservative) post-processing. Orig len: {len(original_answer.strip())}, Proc len: {len(processed_answer)}. Starts with: '{processed_answer[:100]}...'"
            logger.info(log_message_default)
        else:
            logger.debug(f"Default format post-processing made no significant changes beyond initial strip.")

        return processed_answer or original_answer.strip()

    except Exception as e:
        logger.error(f"❌ Post-processing failed: {e}. Original Answer: '{original_answer[:200]}...'", exc_info=True)
        return original_answer # Fallback to original answer on error


@api_router.get("/chats", response_model=List[ChatListItem])
async def get_chat_list_for_user(username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    chat_list = [ChatListItem(id=chat_id, title=meta.get("title", "無標題"), updated_at=meta.get("updated_at", "")) for chat_id, meta in user_chats_metadata.items()]
    chat_list.sort(key=lambda x: x.updated_at, reverse=True)
    return chat_list

@api_router.post("/chats", response_model=ChatListItem)
async def create_new_chat_for_user(req: NewChatRequest, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if req.id in user_chats_metadata: raise HTTPException(status_code=409, detail=f"聊天 ID {req.id} 已存在")
    now_iso = datetime.now().isoformat()
    user_chats_metadata[req.id] = {"title": req.title, "created_at": now_iso, "updated_at": now_iso}
    _save_user_chats_metadata(username, user_chats_metadata)
    _save_user_chat_messages(username, req.id, [])
    logger.info(f"✅ 用戶 {username} 的新聊天已創建: ID={req.id}, Title='{req.title}'")
    return ChatListItem(id=req.id, title=req.title, updated_at=now_iso)

@api_router.get("/chats/{chat_id}/messages", response_model=List[Message])
async def get_chat_messages_for_user(chat_id: str, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if chat_id not in user_chats_metadata:
        logger.warning(f"用戶 {username} 請求不存在或不屬於他的聊天 {chat_id} 的訊息。")
        raise HTTPException(status_code=404, detail=f"聊天 ID {chat_id} 未找到")
    messages = _get_user_chat_messages(username, chat_id)
    return [Message(role=msg["role"], content=msg["content"]) for msg in messages]

@api_router.put("/chats/{chat_id}", response_model=ChatListItem)
async def rename_chat_for_user(chat_id: str, req: RenameChatRequest, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if chat_id not in user_chats_metadata: raise HTTPException(status_code=404, detail=f"聊天 ID {chat_id} 未找到")
    new_title = req.title.strip()
    if not new_title: raise HTTPException(status_code=400, detail="標題不能為空")
    now_iso = datetime.now().isoformat()
    user_chats_metadata[chat_id]["title"] = new_title
    user_chats_metadata[chat_id]["updated_at"] = now_iso
    _save_user_chats_metadata(username, user_chats_metadata)
    logger.info(f"✅ 用戶 {username} 的聊天已重命名: ID={chat_id}, New Title='{new_title}'")
    return ChatListItem(id=chat_id, title=new_title, updated_at=now_iso)

@api_router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat_for_user(chat_id: str, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if chat_id not in user_chats_metadata: raise HTTPException(status_code=404, detail=f"聊天 ID {chat_id} 未找到")
    del user_chats_metadata[chat_id]
    _save_user_chats_metadata(username, user_chats_metadata)
    message_file = get_user_chat_messages_dir(username) / f"{chat_id}.json"
    if message_file.exists():
        try:
            message_file.unlink()
            logger.info(f"用戶 {username} 的聊天訊息文件 {message_file} 已刪除。")
        except Exception as e: logger.error(f"❌ 刪除用戶 {username} 的聊天訊息文件 {message_file} 失敗: {e}", exc_info=True)
    logger.info(f"🗑️ 用戶 {username} 的聊天已刪除: ID={chat_id}")
    if not user_chats_metadata:
        user_chat_data_d = get_user_chat_data_dir(username)
        user_chat_messages_d = get_user_chat_messages_dir(username)
        try:
            if user_chat_messages_d.exists() and not any(user_chat_messages_d.iterdir()): # Check if messages dir is empty
                if user_chat_data_d.exists(): # Check if base user data dir exists before trying to delete
                    shutil.rmtree(user_chat_data_d)
                    logger.info(f"用戶 {username} 的數據目錄 {user_chat_data_d} 因無聊天記錄而被刪除。")
        except Exception as e: logger.error(f"❌ 刪除空的用戶 {username} 數據目錄 {user_chat_data_d} 失敗: {e}", exc_info=True)
    return None

class PublicRAGDocumentSource(BaseModel):
    content: str
    metadata: Dict

async def process_rag_request(
    username: str, session_id: str, question: str,
    selected_model: str, prompt_mode: str,
) -> Dict:
    start_all_processing = time.time()
    if not question:
        logger.warning(f"⚠️ 用戶 {username} 收到空問題。")
        raise HTTPException(status_code=400, detail="問題不能為空.")
    llm = get_model(selected_model)
    if llm is None:
        logger.error(f"❌ RAG process error: LLM '{selected_model}' not loaded.")
        raise HTTPException(status_code=500, detail=f"無法載入語言模型 '{selected_model}'. 請檢查伺服器日誌以了解詳情。")
    if vectordb is None:
        logger.error(f"❌ RAG process error: Vector database not available.")
        raise HTTPException(status_code=500, detail="向量資料庫不可用.")

    user_chats_metadata = _load_user_chats_metadata(username)
    if session_id not in user_chats_metadata:
        now_iso = datetime.now().isoformat()
        default_title = question[:30].strip() + "..." if len(question) > 30 else question.strip() or f"對話 {session_id[:8]}"
        user_chats_metadata[session_id] = {"title": default_title, "created_at": now_iso, "updated_at": now_iso}
        _save_user_chats_metadata(username, user_chats_metadata)
        _save_user_chat_messages(username, session_id, [])
        logger.info(f"ℹ️ 自動為用戶 {username} session '{session_id}' 創建聊天元數據. 標題: '{default_title}'")

    format_mode = detect_format_mode(question)
    logger.info(f"🚀 RAG - User: {username}, Session: {session_id}, Model: {selected_model}, PromptMode(TemplateGroup): {prompt_mode}, FormatMode(LLMInstruction): {format_mode}")
    logger.info(f"❓ Question for {username}: {question[:200]}...") # Log truncated question

    messages_for_prompt_history = _get_user_chat_messages(username, session_id)
    history_pairs = []
    temp_user_q = None
    for msg_dict in messages_for_prompt_history:
        if msg_dict["role"] == "user": temp_user_q = msg_dict["content"]
        elif msg_dict["role"] == "assistant" and temp_user_q:
            history_pairs.append((temp_user_q, msg_dict["content"]))
            temp_user_q = None
    history_text = "\n".join([f"使用者: {q}\n助理: {a}" for q, a in history_pairs[-MAX_HISTORY_PER_SESSION:]]) or "無歷史對話紀錄。"

    start_retrieve = time.time()

    # --- 檢索策略 ---
    # 你的目標是使用 MMR，你已經正確地實現了。
    # similarity_score_threshold 策略已被註解掉。
    # retriever_k: 返回的文檔數量。
    # retriever_fetch_k: 初始獲取以進行 MMR 計算的文檔數量，應大於 k。
    # retriever_lambda_mult: MMR 算法中的多樣性參數 (0=最大多樣性, 1=最大相似度)。
    
    # 最大相似度 (已註解)
    #retriever_k = 10
    #retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": retriever_k, "score_threshold": 0.5})

    # # MMR 策略 (當前生效) # 試試看調整 5, 30, 0.4
    retriever_k = 10   # 10
    retriever_fetch_k = 30  # 40
    retriever_lambda_mult = 0.4   #0.6
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retriever_k, 
            "fetch_k": retriever_fetch_k, 
            "lambda_mult": retriever_lambda_mult
        }
    )

    retrieved_docs_list: List[Dict] = []
    context_str = "沒有找到相關的背景資料。"

    docs_langchain = []
    try:
        docs_langchain = retriever.invoke(question)
        if docs_langchain:
            context_str = "\n\n".join([doc.page_content for doc in docs_langchain])
            retrieved_docs_list = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs_langchain]
        logger.info(f"⏱️ Retrieval: {time.time() - start_retrieve:.2f}s, Found {len(docs_langchain)} docs using MMR for {username}.")
    except Exception as e:
        logger.error(f"❌ Retrieval error for {username}, q='{question[:100]}...': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"向量資料庫檢索錯誤: {str(e)}")

    selected_template: Optional[PromptTemplate] = None
    template_name_for_log = "N/A"

    if prompt_mode == "research":
        selected_template = RESEARCH_PROMPT_TEMPLATE
        template_name_for_log = f"Research (Format: {format_mode})"
    elif format_mode == "custom":
        selected_template = CUSTOM_FORMAT_BASE_PROMPT
        template_name_for_log = "Custom Format Request"
    else:
        selected_template = random.choice(DEFAULT_PROMPT_OPTIONS)
        if selected_template == STRUCTURED_LIST_PROMPT: template_name_for_log = "Default Style (Random: Structured List)"
        elif selected_template == HIERARCHICAL_BULLETS_PROMPT: template_name_for_log = "Default Style (Random: Hierarchical Bullets)"
        elif selected_template == PARAGRAPH_EMOJI_LEAD_PROMPT: template_name_for_log = "Default Style (Random: Paragraph Emoji Lead)"
        else: template_name_for_log = "Default Style (Random: Unknown)"
    
    if selected_template is None:
        logger.error(f"❌ Critical error: No template selected for prompt_mode '{prompt_mode}' and format_mode '{format_mode}'")
        raise HTTPException(status_code=500, detail="內部錯誤：無法選擇提示模板。")
        
    logger.info(f"Using template for {username}: {template_name_for_log} (PromptMode: {prompt_mode}, Detected FormatMode: {format_mode})")

    try:
        prompt_input = {"context": context_str, "question": question, "history": history_text, "format_mode": format_mode}
        prompt = selected_template.format(**prompt_input)
    except Exception as e:
        logger.error(f"❌ Prompt formatting error for {username}: {e} with input keys {list(prompt_input.keys())}", exc_info=True)
        raise HTTPException(status_code=500, detail="內部錯誤：提示詞格式化失敗.")

    final_answer = None
    llm_actual_attempts = 0
    start_llm_total_processing = time.time()
    
    for attempt in range(MAX_LLM_RETRIES + 1):
        llm_actual_attempts = attempt + 1
        try:
            start_llm_one_attempt = time.time()
            raw_answer = llm.invoke(prompt)
            logger.info(f"⏱️ LLM raw response (Attempt {llm_actual_attempts}) for {username}: {time.time() - start_llm_one_attempt:.2f}s. Length: {len(raw_answer)}")
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"LLM raw output (Attempt {llm_actual_attempts}) for {username} before post-processing: '{raw_answer[:500]}...'")

            processed_answer = post_process_answer(raw_answer, format_mode=format_mode)
            if logger.isEnabledFor(logging.DEBUG):
                 logger.debug(f"LLM output (Attempt {llm_actual_attempts}) for {username} AFTER post-processing: '{processed_answer[:500]}...'")

            if not processed_answer or processed_answer.isspace():
                logger.warning(f"LLM returned empty/whitespace answer (Attempt {llm_actual_attempts}) for {username} after processing. Raw: '{raw_answer[:200]}...'")
                if attempt < MAX_LLM_RETRIES: 
                    logger.info(f"Retrying LLM call for {username} (attempt {attempt + 2})")
                    time.sleep(1)
                    continue
                else:
                    raise ValueError("LLM returned empty or whitespace answer after all retries")
            
            final_answer = processed_answer
            break 
        except Exception as e:
            # BUG FIX: 處理 Ollama 連線錯誤
            # 你遇到的 `wsarecv: An existing connection was forcibly closed` 錯誤會在這裡被捕捉到。
            # 這通常意味著 Ollama 伺服器因為資源不足 (VRAM/RAM) 而崩潰。
            logger.error(f"❌ LLM error (Attempt {llm_actual_attempts}) for {username}: {e}", exc_info=True)
            logger.error("💡--- Ollama 執行錯誤分析 ---💡")
            logger.error("這個錯誤通常不是 Python 程式碼邏輯問題，而是 Ollama 服務本身出現了問題。")
            logger.error("最可能的原因是：")
            logger.error("  1. **資源不足**: 請求的模型 (e.g., gemma3:12b) 對於你的 VRAM/RAM 來說太大了。")
            logger.error("  2. **Ollama 服務崩潰**: 服務可能因不明原因停止運作。")
            logger.error("  3. **模型檔案損毀**: 模型檔案可能不完整或已損壞。")
            logger.error("--- 建議的除錯步驟 ---")
            logger.error("  1. **監控資源**: 在提出請求時，打開工作管理員監控 VRAM 和 RAM 的使用情況。")
            logger.error("  2. **嘗試小模型**: 修改請求，使用一個較小的模型，例如 `llama3:8b` 或 `qwen:4b`。如果小模型可以正常運作，幾乎可以肯定是資源問題。")
            logger.error("  3. **重啟 Ollama**: 完全關閉並重新啟動 Ollama 應用程式。")
            logger.error("  4. **重新下載模型**: 在終端執行 `ollama rm <model_name>`，然後執行 `ollama pull <model_name>` 來重新下載。")
            logger.error("💡--------------------------💡")
            
            if attempt < MAX_LLM_RETRIES:
                logger.info(f"Retrying LLM call for {username} (attempt {attempt + 2}) due to error.")
                time.sleep(random.uniform(1,3))
            else:
                logger.error(f"❌ Failed to get LLM response for {username} after {llm_actual_attempts} attempts due to error: {e}")
                # 向前端返回一個更友好的錯誤訊息
                raise HTTPException(status_code=500, detail="LLM 處理錯誤：與 Ollama 服務的連線中斷。請檢查 Ollama 服務狀態和系統資源。")
    
    if final_answer is None:
        logger.error(f"❌ Failed to get valid LLM response for {username} after {MAX_LLM_RETRIES + 1} attempts.")
        raise HTTPException(status_code=500, detail="LLM 回應或處理失敗.")

    current_chat_session_messages = _get_user_chat_messages(username, session_id)
    current_chat_session_messages.append({"role": "user", "content": question})
    current_chat_session_messages.append({"role": "assistant", "content": final_answer})
    _save_user_chat_messages(username, session_id, current_chat_session_messages)

    user_chats_metadata_to_update = _load_user_chats_metadata(username)
    if session_id in user_chats_metadata_to_update:
        user_chats_metadata_to_update[session_id]["updated_at"] = datetime.now().isoformat()
        _save_user_chats_metadata(username, user_chats_metadata_to_update)
    logger.info(f"用戶 {username} 的聊天 {session_id} 記錄已更新並保存。")

    if SAVE_QA:
        try:
            user_qa_log_dir = get_user_qa_log_path(username)
            today_str = datetime.now().strftime("%Y-%m-%d")
            qa_filename = user_qa_log_dir / f"qa_log_{today_str}.jsonl"
            qa_record = {
                "username_source_type": "api_call" if username.startswith("api_consumer_") else "frontend_user",
                "user_identifier": username, "session_id": session_id, "timestamp": datetime.now().isoformat(),
                "model": selected_model, "prompt_mode_requested": prompt_mode,
                "format_mode_detected": format_mode, "question": question, "answer": final_answer,
                "llm_attempts": llm_actual_attempts, "template_used": template_name_for_log,
                "retrieved_docs_count": len(docs_langchain),
                "total_processing_time_seconds": round(time.time() - start_all_processing, 2)
            }
            with open(qa_filename, "a", encoding="utf-8") as f: f.write(json.dumps(qa_record, ensure_ascii=False) + "\n")
        except Exception as e: logger.error(f"❌ Failed to save QA record for {username}: {e}", exc_info=True)
    
    llm_total_time = time.time() - start_llm_total_processing
    retrieval_total_time = time.time() - start_retrieve

    return {
        "answer": final_answer, "model_used": selected_model,
        "prompt_mode_used": prompt_mode, "format_mode_used": format_mode,
        "template_style_used": template_name_for_log, "sources": retrieved_docs_list,
        "session_id": session_id,
        "llm_processing_time_seconds": round(llm_total_time, 2),
        "retrieval_time_seconds": round(retrieval_total_time, 2),
    }

@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(get_current_username)):
    start_overall_request = time.time()
    session_id = req.session_id
    question = req.question.strip()
    selected_model = req.model if req.model and req.model in SUPPORTED_MODELS else DEFAULT_MODEL
    prompt_mode_from_req = req.prompt_mode if req.prompt_mode in ["default", "research"] else "default"
    try:
        result_dict = await process_rag_request(
            username=username, session_id=session_id, question=question,
            selected_model=selected_model, prompt_mode=prompt_mode_from_req
        )
        total_time = time.time() - start_overall_request
        logger.info(f"⏱️ Total /chat request for {username}: {total_time:.2f}s. LLM: {result_dict.get('llm_processing_time_seconds', 'N/A')}s. Retrieval: {result_dict.get('retrieval_time_seconds', 'N/A')}s")
        return {
            "answer": result_dict["answer"], "model_used": result_dict["model_used"],
            "prompt_mode_used": result_dict["prompt_mode_used"],
            "format_mode_used": result_dict["format_mode_used"],
            "template_style_used": result_dict["template_style_used"]
        }
    except HTTPException as e_http:
        logger.error(f"HTTP Exception during /chat for {username}: {e_http.detail}")
        raise e_http
    except Exception as e_general:
        logger.error(f"❌ Unexpected error in /chat for {username}: {str(e_general)}", exc_info=True)
        raise HTTPException(status_code=500, detail="LLM 回應或處理失敗或內部錯誤")

@app.post("/feedback")
async def submit_feedback_for_user(feedback: FeedbackRequest, username: str = Depends(get_current_username)):
    if not feedback.user_expected_answer: raise HTTPException(status_code=400, detail="預期正確回答不能為空")
    record = {
        "question": feedback.user_expected_question or feedback.question,
        "answer": feedback.user_expected_answer,
        "metadata": {"source": "manual_feedback", "original_question": feedback.question,
                     "original_answer": feedback.original_answer, "session_id": feedback.session_id,
                     "model_used": feedback.model, "timestamp": datetime.now().isoformat(),
                     "user_identifier": username}
    }
    try:
        user_feedback_dir = get_user_feedback_save_path(username)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = user_feedback_dir / f"feedback_{feedback.session_id}_{ts_str}.json"
        with open(filename, "w", encoding="utf-8") as f: json.dump(record, f, ensure_ascii=False, indent=2)
        logger.info(f"📝 用戶 {username} 的回饋已保存到 {filename}")
        return {"message": "✅ 使用者回饋已儲存", "filename": str(filename)}
    except Exception as e:
        logger.error(f"❌ 保存用戶 {username} 的回饋失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="儲存回饋失敗.")

class PublicRAGRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question for the RAG system.")
    session_id: Optional[str] = Field(None, description="Optional session ID.")
    model: Optional[str] = Field(DEFAULT_MODEL, description=f"Optional LLM. Supported: {', '.join(SUPPORTED_MODELS)}. Defaults to {DEFAULT_MODEL}.")
    prompt_mode: Optional[str] = Field("default", description="Optional prompt mode. Supported: 'default', 'research'. Defaults to 'default'.")

class PublicRAGResponse(BaseModel):
    answer: str; model_used: str; prompt_mode_used: str; format_mode_used: str
    template_style_used: Optional[str] = None
    sources: List[PublicRAGDocumentSource]
    session_id: str
    llm_processing_time_seconds: Optional[float] = None
    retrieval_time_seconds: Optional[float] = None
    total_request_time_seconds: Optional[float] = None

public_api_v1_router = APIRouter(prefix="/api/v1/public", tags=["Public RAG API v1 (X-API-Key Auth)"])

@public_api_v1_router.post("/rag/ask", response_model=PublicRAGResponse, summary="Ask the RAG system (API Key)")
async def public_rag_ask(req: PublicRAGRequest, api_user_identifier: str = Depends(get_api_key_user)):
    start_overall_request = time.time()
    session_id_to_use = req.session_id
    if not session_id_to_use:
        timestamp_ms = int(time.time() * 1000)
        random_suffix = random.randint(10000, 99999)
        session_id_to_use = f"api_{api_user_identifier.replace('_','-')[:10]}_{timestamp_ms}_{random_suffix}"
        logger.info(f"No session_id by API user '{api_user_identifier}', generated: '{session_id_to_use}'")
    
    selected_model_for_api = req.model if req.model and req.model in SUPPORTED_MODELS else DEFAULT_MODEL
    prompt_mode_for_api = req.prompt_mode if req.prompt_mode in ["default", "research"] else "default"
    try:
        result_dict = await process_rag_request(
            username=api_user_identifier, session_id=session_id_to_use, question=req.question,
            selected_model=selected_model_for_api, prompt_mode=prompt_mode_for_api,
        )
        total_time = time.time() - start_overall_request
        result_dict["total_request_time_seconds"] = round(total_time, 2)
        logger.info(f"⏱️ Total Public API request for API user '{api_user_identifier}': {total_time:.2f}s. LLM: {result_dict.get('llm_processing_time_seconds','N/A')}s. Retrieval: {result_dict.get('retrieval_time_seconds','N/A')}s")
        return PublicRAGResponse(**result_dict)
    except HTTPException as e_http:
        logger.error(f"HTTP Exception Public API for user '{api_user_identifier}': {e_http.detail}")
        raise e_http
    except Exception as e_general:
        logger.error(f"❌ Unexpected error public_rag_ask for API user '{api_user_identifier}': {str(e_general)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing RAG request.")

app.include_router(api_router)
app.include_router(public_api_v1_router)

# To run (replace `your_filename` with the actual name of your Python file):
# uvicorn 6_10test:app --host 0.0.0.0 --port 8000 --reload   6/11 有微調prompt 如果效果不好就用這個 目前較為準確的版本  6/18為目前最優先後端

# 使用分離的IP  uvicorn 6_10test:app --host 0.0.0.0 --port 8000  目前先使用這個 Duck DNS和Nginx
