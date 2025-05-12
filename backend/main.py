# -*- coding: utf-8 -*-
from fastapi import FastAPI, Body, Depends, Request, HTTPException, APIRouter, Header
from pydantic import BaseModel
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
from typing import Optional, List, Dict, Annotated # Annotated for Header
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from functools import lru_cache
import random
import re
import shutil # For deleting user directories

# ✅ FastAPI 初始化
app = FastAPI()
api_router = APIRouter(prefix="/api")


# ✅ 判斷使用者是否有指定格式需求
def detect_format_mode(question: str) -> str:
    format_triggers = [
        "請用一段話", "摘要", "表格", "表列", "條列式", "清單形式", "一句話", "說明就好",
        "summarize", "as a table", "one paragraph", "bullet points", "list format"
    ]
    if any(re.search(r'\b' + re.escape(kw) + r'\b', question, re.IGNORECASE) for kw in format_triggers) or "簡單說明" in question:
         return "custom"
    return "default"

# --- Middleware Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-Username"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# --- Logging Configuration ---
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"日誌級別設定為: {log_level}")


# --- Global Variables & Constants ---
MAX_HISTORY_PER_SESSION = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"使用設備: {device}")

embedding = None
vectordb = None
request_counters = {}

FEEDBACK_SAVE_PATH_BASE = Path("user_specific_feedback")
QA_LOG_PATH_BASE = Path("user_specific_qa_logs")
SAVE_QA = True
MAX_LLM_RETRIES = 1

USER_DATA_BASE_DIR = Path("user_specific_chat_data")

def sanitize_username(username: str) -> str:
    if not username:
        return "default_user"
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', username).lower()
    return sanitized if sanitized else "invalid_user"

def get_user_chat_data_dir(username: str) -> Path:
    return USER_DATA_BASE_DIR / sanitize_username(username)

def get_user_chats_metadata_file(username: str) -> Path:
    return get_user_chat_data_dir(username) / "chats_metadata.json"

def get_user_chat_messages_dir(username: str) -> Path:
    return get_user_chat_data_dir(username) / "chat_messages"

def get_user_feedback_save_path(username: str) -> Path:
    return FEEDBACK_SAVE_PATH_BASE / sanitize_username(username)

def get_user_qa_log_path(username: str) -> Path:
    return QA_LOG_PATH_BASE / sanitize_username(username)

async def get_current_username(x_username: Optional[str] = Header(None)) -> str:
    # 如果 x_username 沒有在 header 中提供，FastAPI 會因為 Header(None) 將其設為 None
    if x_username is None:
        logger.warning("請求中未找到 X-Username Header")
        raise HTTPException(status_code=400, detail="請求標頭中缺少 X-Username")
    
    sanitized = sanitize_username(x_username)
    if not sanitized or sanitized == "invalid_user":
        logger.warning(f"提供的用戶名 '{x_username}' 清理後無效")
        raise HTTPException(status_code=400, detail="提供的用戶名無效")
    
    # ... (後續的目錄創建和返回邏輯不變) ...
    user_data_dir = get_user_chat_data_dir(sanitized)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    (user_data_dir / "chat_messages").mkdir(parents=True, exist_ok=True)
    
    user_feedback_dir = get_user_feedback_save_path(sanitized)
    user_feedback_dir.mkdir(parents=True, exist_ok=True)
    
    user_qa_dir = get_user_qa_log_path(sanitized)
    user_qa_dir.mkdir(parents=True, exist_ok=True)

    return sanitized

try:
    embedding_model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    logger.info(f"正在載入嵌入模型: {embedding_model_name}")
    embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    _ = embedding.embed_query("測試嵌入模型")
    logger.info("✅ 嵌入模型載入並測試成功")
except Exception as e:
    logger.error(f"❌ 嵌入模型載入失敗: {str(e)}", exc_info=True)
    embedding = None

@app.on_event("startup")
def startup_event():
    global vectordb
    if embedding is None:
        logger.error("❌ 嵌入模型未載入，無法初始化向量資料庫。")
    else:
        try:
            persist_dir = os.environ.get("VECTORDB_PATH", "5_10test")
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
    logger.info(f"用戶特定數據的基礎目錄已準備就緒: {USER_DATA_BASE_DIR}")

# --- Prompt Templates ---
STRUCTURED_LIST_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
You are a helpful assistant providing clear and structured information in **Traditional Chinese**. Your task is to answer the user's question based on the retrieved context below, adhering strictly to the specified format based on the `format_mode`.

📌 **Format Mode:** {format_mode}

🚨 **格式指令 (Format Instructions - 嚴格遵守):**

*   **If `format_mode` is `custom`:** 代表使用者在問題中指定了格式。請 **完全遵循使用者要求的格式** 進行回答。如果使用者要求隱含了 Markdown 結構 (如列表)，請使用標準 Markdown (`* `, `1. `)。確保 `**粗體**` 使用兩個星號。
*   **If `format_mode` is `default`:** 代表系統隨機選中了下述的「結構化列表風格」。你 **必須** 嚴格使用此風格，任何偏差都視為錯誤。
    *   **最關鍵要求 (CRITICAL for default mode):** 章節標題 **必須** 使用 **兩個星號** 包裹 (格式為 `**一、中文標題**`)。**絕對禁止** 使用多餘的星號 (如 `***標題***` 或 `* **標題** *`) 或單個星號。**標題後的內容文字絕不能加粗。**

🎯 **預設回應格式 (Default Response Format - 結構化列表風格 - 僅在 `format_mode` 為 'default' 時使用):**

1.  **引言 (Introduction):** 在開頭提供一個簡短的引言 (1-2 句話)。
2.  **編號章節 (Numbered Sections):** 包含 2 至 5 個章節，使用 `**一、**`, `**二、**` 等作為標題前綴，標題本身需加粗，完整格式為 `**一、中文標題**`。
3.  **章節內容 (Content):** 每個章節標題下方撰寫 1-3 句話的內容。**內容必須是純文字 (plain text)**，不得加粗或使用斜體。語言需清晰易懂。可以在內容文字中 **少量** 使用相關的表情符號 (見下方建議)，但 **標題中禁止使用任何表情符號**。
4.  **分隔線 (Separator Line):** 在 **每個** 章節的內容文字結束後，**必須** 插入一行由 **100個** 半形句點 (`.`) 組成的分隔線，如下：
    ....................................................................................................
5.  **間距 (Spacing):** 引言和第一個章節標題之間保留一個空行。每個分隔線和下一個章節標題之間也保留一個空行。
6.  **語言 (Language):** 全文使用 **繁體中文**。
7.  **內容來源 (Context Usage):** 回答內容 **僅能** 根據下方提供的 `Retrieved Context` 生成，不要提及「根據上下文」或直接複製上下文原文。
8.  **表情符號建議 (Emojis - 僅用於內容文字):** 💡, ✅, ⚠️, 📊, 👥, 🏫, 🌱, 🤝, ❤️ (或其他與內容相關的符號)

📘 **Conversation History:** {history}
📄 **Retrieved Context:** {context}
❓ **User Question:** {question}

👇 **請用繁體中文回答。根據偵測到的 `format_mode` 遵循對應的格式指令。若為 `default` 模式，請極度嚴格地遵守「結構化列表風格」的所有細節，特別是 `**標題**` 格式和純文字內容要求。**

📝 **輸出範例 (EXAMPLE OUTPUT FORMAT - when `format_mode` is "default"):**

小港空污USR計畫已展現出一定的成效 ✅，主要體現在提升社區居民對空污議題的認知、促進健康行為的改變以及建立社區參與的機制。

**一、社區參與與教育推廣**
計畫團隊深入小港區的學校 🏫 與社區 👥，舉辦各類環境與健康教育活動，例如針對空污敏感族群的兒童氣喘衛教營隊，以及在鳳林國小舉辦的「空污健康站」環境教育嘉年華。這些活動吸引了數百名小港地區居民參與，成功將空污知識轉化為社區行動。
.................................................................................................................................................................

**二、針對特定族群的健康促進**
計畫針對空污敏感族群的兒童氣喘進行衛教，並擴及高齡族群，例如舉辦高齡者社區健康促進講座和長照據點合作的呼吸保健課程。這些活動提升了不同年齡層對空氣品質與健康風險的認知，並促進了健康行為的改變 ❤️。
.................................................................................................................................................................

**三、與企業合作的亮點**
計畫與小港醫院及地方企業合作 🤝 推動空污監測系統和ESG健康促進方案，顯示計畫在整合資源、促進社區發展方面的努力。
.................................................................................................................................................................

**四、計畫目標的落實與成果分享**
計畫申請書完整闡述了小港空污議題的背景、目標、執行方案與預期效益 📊，並定期提交進度與成果報告，確保計畫目標的落實，並通過多元方式進行成效評估與分享。
.................................................................................................................................................................
"""
)

HIERARCHICAL_BULLETS_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
You are a helpful assistant tasked with providing detailed, hierarchically structured information **in Traditional Chinese**. Your task is to answer the user's question based on the retrieved context below using **standard Markdown headings and lists** for structure. This style was randomly selected for the default mode.

📌 **Format Mode:** {format_mode} (System detected: 'default' - no specific user request in question)

🚨 **Format Instructions (Strictly Adhere to this Style):**

*   Your response **MUST** be entirely in **Traditional Chinese**, structured using standard Markdown.
*   **Headings:**
    *   Use `# ` (one hash + space) for the optional **Main Title**.
    *   Use `## ` (two hashes + space) for **Main Section Headers** (e.g., `## 一、Main Section Title`).
    *   Use `### ` (three hashes + space) for **Sub-section Headers** (e.g., `### 1. Sub-section Title`).
*   **Lists:**
    *   Below headings, when itemization is needed, use standard Markdown **unordered lists** (`* ` or `- ` followed by a space) or **ordered lists** (`1. `, `2. ` followed by a space).
    *   List item text must be **plain text starting immediately after the marker and space**. **CRITICAL: Absolutely DO NOT apply bold markdown (`**`) to the text immediately following the list marker (`* / - / 1. `) or any introductory label ending with a colon (like `Label:`).** Use `**bold emphasis**` very sparingly *only* for specific keywords deep *within* the list item text itself.
    *   Maintain consistent list indentation (typically 2 or 4 spaces).
*   **Spacing:** Ensure one blank line between the Main Title (if used) and the first Section Header, and one blank line between subsequent `##` Section Headers.
*   **Critical Requirements:**
    *   **Strictly adhere** to the Markdown syntax specified above, especially the **required spaces** after heading markers (`#`, `##`, `###`) and list markers (`*`, `-`, `1.`).
    *   **Absolutely prohibit** non-standard formats or extra symbols.
    *   **Bold (`**`)** is used *only* for emphasizing specific words within paragraphs or list items. **Do not bold entire headings or list items, and especially not the text immediately following a list marker or label.**
*   **Context Usage:** Base your answer *only* on the provided `Retrieved Context`. **Absolutely prohibit mentioning** "According to the text," "The text indicates," "The text doesn't mention," or any phrases referencing the context source. State the information derived from the context directly.
*   **No Preamble / Meta-commentary:** Start the response **directly** with the required formatted content (e.g., the `#` title or first `##` heading). **Do not** add introductory phrases like "Okay, here is the information..." or comments about the format itself like "Here is the response in hierarchical format:".
*   **No Conversational Closing:** Do not add concluding remarks like "Hope this helps!" or follow-up questions.

📘 **Conversation History:**
{history}

📄 **Retrieved Context:**
{context}

❓ **User Question:**
{question}

👇 **Respond entirely in Traditional Chinese.** Strictly follow the 'Hierarchical Markdown' format: use headings with spaces (`# /## /### `) and standard lists (`* / - / 1. `). **Ensure list item text immediately following the marker or a label ending in a colon is plain text, NOT bold.** Start directly with the formatted content.

📝 **Example Output Format (when `format_mode` is "default" and this style is chosen):**

# 空氣污染教育成效評估方法

在台灣的大學社會責任（USR）計畫中，評估「空氣污染教育成效」是一項多面向的任務，應結合量化與質化指標，以全面了解教育活動對社區、學生與政策層面的影響。以下是幾種常見且建議使用的評估方式：

## 一、量化指標（Quantitative Evaluation）

### 1. 前後測問卷分析
*   針對參與者（如學生、社區居民）在課程或活動前後進行知識、態度與行為意向測驗。
*   比較其環境知識增長、對空污議題的**敏感度**提升。

### 2. 參與人數與活動場次
*   統計實體或線上課程、講座、工作坊參與人次。
*   長期追蹤是否有固定參與族群，或是否能觸及新對象。

### 3. 行為改變的指標
*   如居民是否開始使用空氣品質監測器、改變交通工具使用習慣。
*   學校是否推動校園綠化等。

### 4. 社群媒體與平台互動
*   觀看次數、分享、留言、點讚數等可衡量資訊擴散成效。

## 二、質化指標（Qualitative Evaluation）

### 1. 深度訪談與焦點團體
*   透過與學生、教師、居民及在地 NGO 訪談，瞭解空污教育帶來的觀念改變或生活實踐。
*   瞭解參與者對教育內容的接受度與建議。

### 2. 學生與居民的反思紀錄或學習成果
*   包含學習單、反思日誌、創作（如短片、海報）等。
*   作為其內化成果的呈現。

### 3. 社區合作的實質成果
*   如與地方政府、學校或社區合作建立空污監測站。
*   共同提出改善建議等。

## 三、長期成效追蹤（Impact Tracking）

### 1. 政策影響力
*   是否促成地方政府或學校在空污議題上的政策修訂。
*   推動實作方案。

### 2. 社區意識抬頭與自發行動
*   是否出現自主辦理相關活動。
*   成立自救會或監督平台。

### 3. 跨領域與永續擴散
*   評估是否能與其他 USR 團隊、研究單位或企業形成合作。
*   將教育模式擴展至其他區域或主題。

*(Note: The example output deliberately excludes concluding summaries or questions.)*
"""
)

PARAGRAPH_EMOJI_LEAD_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
You are a helpful assistant providing clear, paragraph-based explanations in **Traditional Chinese**. Your task is to answer the user's question based on the retrieved context below, using a specific style where each paragraph starts with a relevant emoji. This style was randomly selected for the default mode.

📌 **Format Mode:** {format_mode} (System detected: 'default' - no specific user request in question)

🚨 **格式指令 (嚴格遵守此風格):**

*   你的回答 **必須** 是一系列 **繁體中文** 的段落。
*   **最關鍵的是：每個段落都必須以 `一個相關的表情符號` + `一個空格` 開頭。** 表情符號應與該段落的主題相關。
*   **表情符號和空格之後的文字內容，必須是純文字 (plain text)。** 絕對不要自動將這部分文字加粗。
*   **完全避免** 在回答中使用任何編號列表 (`1.`, `2.`)、項目符號列表 (`*`, `-`)、章節標題 (`#`, `##`, `**標題**`) 或分隔線 (`---`, `...`)。專注於純段落結構。
*   如果需要在段落 *內部* 強調特定關鍵字，可以 **非常少量地** 使用標準 Markdown 粗體 (`**強調詞**`)。**切勿** 將表情符號後的第一個完整句子加粗。

💡 **表情符號建議 (根據段落主題選擇，也可使用其他相關符號):**
    💡, 🤝, 🏥, 👥, 📚, 🌱, 🔬, 🧭, ✅, ⚠️, 📊, 🏫

📘 **Conversation History:** {history}
📄 **Retrieved Context:** {context}
❓ **User Question:** {question}

👇 **請用繁體中文回答。嚴格遵循「段落前置表情符號」格式：每個段落以單一表情符號+空格開頭，後面接續純文字。**

📝 **輸出範例 (當 `format_mode` 為 "default" 且選中此風格時):**

💡 在USR（大學社會責任）計畫中，與醫療機構的合作是推動社區健康促進的重要一環。該計畫通過多種方式來確保居民能夠獲得持續且有效的健康管理服務。

🏥 首先，USR計畫與地方政府及醫療單位合作設立健康檢查站，並定期進行監測。這些健康檢查站不僅提供基本的體檢服務，還包括針對空氣污染等環境因素對健康影響的專門評估。

👥 此外，USR計畫也組織聯合衛教活動和社區實踐項目，促使校園與社區形成緊密互動與協同發展。這些活動旨在全方位提升居民的生活品質和健康水平。

🤝 為了進一步推動社區健康促進，USR計畫還強調了與醫療機構在資源整合上的重要性。這包括利用學術研究的力量來開發新的健康產品和服務。

✅ 總之，USR計畫透過與醫療機構的合作，從多個層面推動社區健康促進工作，確保了居民能夠獲得全面且有效的健康管理服務。
"""
)

CUSTOM_FORMAT_BASE_PROMPT = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
You are a helpful assistant providing information in **Traditional Chinese**. The user has asked a question and appears to have included specific instructions on the desired response format within their question text.

📌 **Format Mode:** {format_mode} (System detected: 'custom' - likely user-specified format in question)

🚨 **最優先指令 (ABSOLUTE TOP PRIORITY):**
仔細分析下方的 **User Question**。你的 **首要任務** 是 **精確地理解並嚴格遵守** 使用者在問題文字中包含的 **任何關於輸出格式的明確指示**。
*   例如："請用一段話總結", "條列式說明優缺點", "製作一個比較表格", "summarize in bullet points", "給我點列式清單" 等。
*   **使用者在問題中提出的格式要求，擁有絕對的最高優先權，必須覆蓋所有其他預設的格式或風格。**
*   如果使用者的要求隱含了某種結構（如要求列表），請使用 **標準且語義正確的 Markdown** 來實現 (例如，列表使用 `* ` 或 `1. `，強調使用 `**粗體文字**`)。
*   **嚴格確保 Markdown 語法的正確性：** `**粗體**` 必須使用兩個星號，**絕對避免** 使用多餘的星號 (如 `***標題***`) 或單個星號 (`*強調*`) 來表示粗體或標題。

💡 **例外情況處理:**
如果經過仔細分析，你 **確認** 使用者的問題文字中 **確實沒有包含任何明確的格式指令** (即使 `format_mode` 被設為 'custom')，那麼請 **忽略 `custom` 模式**，並根據提供的上下文，以 **清晰、有條理的標準段落** 形式，完整地回答問題即可。

📘 **Conversation History:** {history}
📄 **Retrieved Context:** {context}

❓ **User Question:**
{question}

👇 **請用繁體中文回答。絕對優先遵循使用者問題中的格式要求。若無明確要求，則以標準段落回答。請確保 Markdown 語法使用正確。**
"""
)

RESEARCH_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["question", "context", "history", "format_mode"],
    template="""
You are a policy analyst and academic writer providing an evaluation in **Traditional Chinese**. Your output should be suitable for a formal academic or governmental report, using **Markdown headings** for structure.

📌 **Format Mode:** {format_mode}

🚨 **Formatting Guidelines (嚴格遵守):**

*   **If `format_mode` is `custom`:** This means the user specified a custom output structure in their question. Follow the user's requested formatting **exactly**, maintaining a formal and academic tone. Use standard Markdown (`#`, `##`, `###`, `* list item`, `**bold**`) as appropriate unless the user specifies otherwise.
*   **If `format_mode` is `default`:** Follow the standard structured report format using **Markdown headings** as described below. Strictly adhere to the formatting rules.

    1.  **主標題 (Main Title):** 使用 `#` (一個井號) 作為整個報告的主標題 (若適用)。
    2.  **主要章節標題 (Section Headers):** 使用 `##` (兩個井號) 標示主要章節 (例如: `## 一、成效亮點`)。
    3.  **次要章節/小節標題 (Sub-section Headers):** 使用 `###` (三個井號) 標示次要章節或小節 (例如: `### 1. PM2.5 濃度變化`)，視需要使用。
    4.  **內容 (Content):** 在標題下方，使用清晰的段落文字。
    5.  **列表 (Lists):** 若需條列，請使用標準 Markdown 列表，以 `* ` 或 `- ` 開頭 (例如: `* 項目一`)。列表項目本身應為**純文字**，除非特定關鍵字需要**粗體** (`**強調詞**`)。**不要將整個列表項目加粗**。
    6.  **間距 (Spacing):** 在主標題和第一個章節標題之間、以及各個章節標題 (`##`) 之間保留一個空行。
    7.  **語言與風格 (Language & Tone):** 全文使用**繁體中文**。語氣必須正式、客觀、學術。**禁止**使用表情符號 (emojis)、非標準的 Markdown 樣式 (除了指示的 `#`, `##`, `###`, `*`, `**`) 或口語化/非正式語言。

📘 **Conversation History:**
{history}

📘 **Document Context:**
{context}

🧾 **User's Question:**
{question}

👇 **請用繁體中文回答:**
*   若 `format_mode` 是 `custom`，請完全遵循使用者在問題中定義的格式。
*   若 `format_mode` 是 `default`，請嚴格遵守上述使用 **Markdown 標題 (`#`, `##`, `###`)** 的結構化報告格式。**粗體 (`**`) 僅用於內文特定詞語強調，不可用於標題或整個列表項目。**

📝 **EXAMPLE OUTPUT FORMAT (when `format_mode` is "default"):**

# 高雄小港空氣污染議題分析報告

## 一、成效亮點
高雄小港區為台灣重要的工業重鎮，過去長期遭受石化業與重工業排放所帶來的空氣污染。自政府實施空污防制強化方案以來，已逐步見到成果：

*   PM2.5 年均濃度下降：2023年，小港區PM2.5濃度首次降至15μg/m³以下，符應國家標準。
*   高污染工廠改善：多家高污染事業完成鍋爐設備更新或污染防制設施強化。
*   在地參與提升：透過社區論壇、校園教育及USR協作，小港居民參與空品改善活動人數顯著提升，展現地方共治的潛能。

這些成果說明政策具備初步效益，也顯示社區力量在環境治理中日益關鍵。

## 二、主要挑戰與限制
儘管已有顯著進展，小港區的空氣污染問題仍存在下列挑戰：

*   結構性污染源持續存在：小港擁有多個大型石化工業區，污染物總量基數高，使得單一改善措施難以對空品產生劇烈改變。
*   地形與氣候劣勢：背風側效應與氣候條件使污染物易滯留，加重局部污染濃度。
*   政策協作落差：中央與地方在污染熱區資料整合與應變作為上，仍顯不足，導致反應時間延遲、缺乏即時調控力。
*   公民動能不足：部分居民對空污議題已產生「習慣性麻痺」，缺乏主動監督與行動參與。

## 三、政策建議與改進方向
為有效深化治理成效，建議可從以下三個策略推動：

### 1. 擴大感測與資訊公開
*   建立高密度感測網，強化移動式監測。
*   發展即時污染視覺化平台，提升公眾風險意識。

### 2. 產業轉型與排放總量控管
*   推行「總量管制 + 差別費率」，引導污染業者升級。
*   鼓勵潔淨能源使用與碳排透明揭露。

### 3. 強化社區共治與環境教育
*   整合USR計畫與地方課程，發展空污資料解讀與倡議能力。
*   建立社區參與制度，如居民空品會議、政策參與管道等。

## 四、總體觀察與評論
小港空污問題的治理難度源自結構性深層因素，短期內難以徹底逆轉。然而，已有跡象顯示，只要政策持續推進並強化地方共治，將有機會轉危為機。特別是結合**科技監測**與**民眾行動力**，能構築出一套適合台灣重工業都市的永續治理模式。

## 五、結論
高雄小港的空污改善歷程是台灣工業區環境治理的縮影。持續落實「科技導向 + 公民參與 + 法制改革」三位一體的策略，將是未來提升空品與健康福祉的關鍵方向。
"""
)

DEFAULT_PROMPT_OPTIONS = [
    STRUCTURED_LIST_PROMPT,
    HIERARCHICAL_BULLETS_PROMPT,
    PARAGRAPH_EMOJI_LEAD_PROMPT
]

# --- LLM Configuration and Loading ---
common_llm_config = { "temperature": 0.1, "top_p": 0.8 }
SUPPORTED_MODELS = [
    "qwen2.5:14b", "gemma3:12b", "gemma3:12b-it-q4_K_M", "qwen2.5:14b-instruct-q5_K_M",
]
DEFAULT_MODEL = "gemma3:12b"

@lru_cache(maxsize=5)
def get_model(model_name: str) -> Optional[OllamaLLM]:
    if model_name not in SUPPORTED_MODELS:
        logger.warning(f"⚠️ 請求的模型 '{model_name}' 不在支援列表，將使用預設模型 '{DEFAULT_MODEL}'。")
        model_name = DEFAULT_MODEL
    try:
        logger.info(f"⏳ 正在載入或獲取緩存的模型: {model_name}...")
        model = OllamaLLM(model=model_name, **common_llm_config, request_timeout=300.0)
        _ = model.invoke("請做個自我介紹")
        logger.info(f"✅ 模型 {model_name} 載入並測試成功")
        return model
    except Exception as e:
        logger.error(f"❌ 載入或測試模型 {model_name} 失敗: {str(e)}", exc_info=True)
        return None

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if "/chat" not in str(request.url):
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    request_timestamps = request_counters.get(client_ip, [])
    valid_timestamps = [t for t in request_timestamps if current_time - t < 60]
    rate_limit_count = 30
    if len(valid_timestamps) >= rate_limit_count:
        logger.warning(f"🚦 速率限制觸發: IP {client_ip}")
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

class ChatListItem(BaseModel):
    id: str
    title: str
    updated_at: str

class NewChatRequest(BaseModel):
    id: str
    title: str

class RenameChatRequest(BaseModel):
    title: str

class Message(BaseModel):
    role: str
    content: str

def _load_user_chats_metadata(username: str) -> Dict[str, Dict[str, str]]:
    metadata_file = get_user_chats_metadata_file(username)
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 無法為用戶 {username} 加載聊天元數據: {e}", exc_info=True)
    return {}

def _save_user_chats_metadata(username: str, metadata: Dict[str, Dict[str, str]]):
    metadata_file = get_user_chats_metadata_file(username)
    try:
        sorted_metadata = dict(sorted(metadata.items(), key=lambda item: item[1].get('updated_at', ''), reverse=True))
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(sorted_metadata, f, ensure_ascii=False, indent=2)
        logger.debug(f"用戶 {username} 的聊天元數據已保存到 {metadata_file}")
    except Exception as e:
        logger.error(f"❌ 保存用戶 {username} 的聊天元數據失敗: {e}", exc_info=True)

def _get_user_chat_messages(username: str, chat_id: str) -> List[Dict[str, str]]:
    message_file = get_user_chat_messages_dir(username) / f"{chat_id}.json"
    if message_file.exists():
        try:
            with open(message_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 無法讀取用戶 {username} 聊天 {chat_id} 的訊息: {e}", exc_info=True)
    return []

def _save_user_chat_messages(username: str, chat_id: str, messages: List[Dict[str, str]]):
    message_file = get_user_chat_messages_dir(username) / f"{chat_id}.json"
    try:
        with open(message_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        logger.debug(f"用戶 {username} 聊天 {chat_id} 的訊息已保存到 {message_file}")
    except Exception as e:
        logger.error(f"❌ 保存用戶 {username} 聊天 {chat_id} 的訊息失敗: {e}", exc_info=True)


def post_process_answer(answer: str) -> str:
    original_answer = answer
    try:
        # 1. 移除開頭多餘片語
        patterns = [
            r"^\s*根據提供的文本[,，]?\s*",
            r"^\s*文本指出[,，]?\s*",
            r"^\s*文本未(?:直接)?提及(?:，|,)但可以推斷[,，]?\s*",
            r"^\s*以下將.*?格式呈現：?\s*",
            r"^\s*這份文件提到(?:了)?\s*",
            r"^\s*好的[,，]?\s*",
            r"^\s*是的[,，]?\s*",
            r"^\s*Here is the.*?you requested[:.]?\s*",
        ]
        for p in patterns:
            answer = re.sub(p, "", answer, count=1, flags=re.IGNORECASE|re.MULTILINE).lstrip()

        # 2. 自訂替換範例（保留）
        def replace_lack(m): return "執法機制的有效性是影響改善速度的關鍵因素。"
        answer = re.sub(
            r"(\n\s*|\A\s*)(缺乏嚴格的執法機制.*?)(?:\s*。|\s*\n|\s*$)",
            lambda m: m.group(1) + replace_lack(m),
            answer, flags=re.MULTILINE
        )
        answer = re.sub(
            r"(\n\s*|\A\s*)(這可能涉及.*?(?:。|\n|$))",
            r"\1", answer, flags=re.MULTILINE
        )

        # 3. 移除不想要的短語
        to_remove = [
            "(文本未直接提及，但可推測)",
            "文章多次提及", "文章提及",
            "文本提及", "根據文本",
            "文件提及", "根據文件",
            "可推論", "但可推論"
        ]
        for ph in to_remove:
            answer = answer.replace(ph, "")

        # 4. 簡化空白與標點間距
        answer = re.sub(r'[ \t]{2,}', ' ', answer)
        answer = re.sub(r'\s+([,.:;!?，。：；！？])', r'\1', answer)
        answer = re.sub(r'([,.:;!?，。：；！？])([^\s,.:;!?，。：；！？\n])', r'\1 \2', answer)
        answer = re.sub(r'^\s*[,，：:]\s*', '', answer).strip()

        # 5. 移除系統標記
        markers = [
            "📘 Conversation History:", "📄 Retrieved Context:",
            "❓ User Question:", "👇 Please write your answer",
            "📝 EXAMPLE OUTPUT FORMAT", "You are a helpful assistant",
            "You are a policy analyst"
        ]
        for m in markers:
            if m in answer:
                answer = answer.split(m, 1)[0].strip()

        # 6. 統一列表符號：將行首 * 或 -（有無空格）都改為 • 
        answer = re.sub(r'(?m)^\s*[\*\-]\s*', '• ', answer)

        # 7a. 只針對「• *一、…」這種情況，移除「• 」後面單一星號
        answer = re.sub(
            r'(?m)^(•\s*)\*(?=[一二三四五六七八九十]+、)',
            r'\1',
            answer
        )
        # 7b. 同時移除行尾多餘的單一星號（如果有），例如「• 一、標題*」
        answer = re.sub(
            r'(?m)^(•.*?)[*]+\s*$',
            r'\1',
            answer
        )

        # 8. 處理多重星號與標題標記（保留既有邏輯）
        answer = re.sub(r'\*{3,}', '**', answer)
        answer = answer.replace('** ', '**').replace(' **', '**')
        answer = re.sub(r'(?m)^###\s+(.*?)\s*$', r'<h3>\1</h3>', answer)
        answer = re.sub(r'(?m)^##\s+(.*?)\s*$', r'<h2>\1</h2>', answer)
        answer = re.sub(r'(?m)^#\s+(.*?)\s*$', r'<h1>\1</h1>', answer)
        answer = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', answer)

        # 9. 清理多餘換行
        answer = re.sub(r'\n{3,}', '\n\n', answer).strip()

        # 10. 若結果為空，回傳原始
        return answer or original_answer

    except Exception as e:
        logger.error(f"❌ Post-processing failed: {e}", exc_info=True)
        return original_answer





@api_router.get("/chats", response_model=List[ChatListItem])
async def get_chat_list_for_user(username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    chat_list = [
        ChatListItem(id=chat_id, title=meta.get("title", "無標題"), updated_at=meta.get("updated_at", ""))
        for chat_id, meta in user_chats_metadata.items()
    ]
    chat_list.sort(key=lambda x: x.updated_at, reverse=True)
    return chat_list

@api_router.post("/chats", response_model=ChatListItem)
async def create_new_chat_for_user(req: NewChatRequest, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    chat_id = req.id
    title = req.title
    if chat_id in user_chats_metadata:
        raise HTTPException(status_code=409, detail=f"聊天 ID {chat_id} 已存在")
    
    now_iso = datetime.now().isoformat()
    user_chats_metadata[chat_id] = {"title": title, "created_at": now_iso, "updated_at": now_iso}
    _save_user_chats_metadata(username, user_chats_metadata)
    _save_user_chat_messages(username, chat_id, [])
    logger.info(f"✅ 用戶 {username} 的新聊天已創建: ID={chat_id}, Title='{title}'")
    return ChatListItem(id=chat_id, title=title, updated_at=now_iso)

@api_router.get("/chats/{chat_id}/messages", response_model=List[Message])
async def get_chat_messages_for_user(chat_id: str, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if chat_id not in user_chats_metadata:
        logger.warning(f"用戶 {username} 請求不存在的聊天 {chat_id} 的訊息。")
        raise HTTPException(status_code=404, detail=f"聊天 ID {chat_id} 未找到")

    messages = _get_user_chat_messages(username, chat_id)
    return [Message(role=msg["role"], content=msg["content"]) for msg in messages]

@api_router.put("/chats/{chat_id}", response_model=ChatListItem)
async def rename_chat_for_user(chat_id: str, req: RenameChatRequest, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if chat_id not in user_chats_metadata:
        raise HTTPException(status_code=404, detail=f"聊天 ID {chat_id} 未找到")
    
    new_title = req.title.strip()
    if not new_title:
        raise HTTPException(status_code=400, detail="標題不能為空")
    
    now_iso = datetime.now().isoformat()
    user_chats_metadata[chat_id]["title"] = new_title
    user_chats_metadata[chat_id]["updated_at"] = now_iso
    _save_user_chats_metadata(username, user_chats_metadata)
    logger.info(f"✅ 用戶 {username} 的聊天已重命名: ID={chat_id}, New Title='{new_title}'")
    return ChatListItem(id=chat_id, title=new_title, updated_at=now_iso)

@api_router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat_for_user(chat_id: str, username: str = Depends(get_current_username)):
    user_chats_metadata = _load_user_chats_metadata(username)
    if chat_id not in user_chats_metadata:
        raise HTTPException(status_code=404, detail=f"聊天 ID {chat_id} 未找到")
    
    del user_chats_metadata[chat_id]
    _save_user_chats_metadata(username, user_chats_metadata)
    
    message_file = get_user_chat_messages_dir(username) / f"{chat_id}.json"
    if message_file.exists():
        try:
            message_file.unlink()
            logger.info(f"用戶 {username} 的聊天訊息文件 {message_file} 已刪除。")
        except Exception as e:
            logger.error(f"❌ 刪除用戶 {username} 的聊天訊息文件 {message_file} 失敗: {e}", exc_info=True)
    
    logger.info(f"🗑️ 用戶 {username} 的聊天已刪除: ID={chat_id}")
    if not user_chats_metadata:
        user_chat_dir = get_user_chat_data_dir(username)
        try:
            messages_dir = get_user_chat_messages_dir(username)
            if not any(messages_dir.iterdir()):
                shutil.rmtree(user_chat_dir)
                logger.info(f"用戶 {username} 的數據目錄 {user_chat_dir} 因無聊天記錄而被刪除。")
        except Exception as e:
            logger.error(f"❌ 刪除空的用戶 {username} 數據目錄 {user_chat_dir} 失敗: {e}", exc_info=True)
    return None

@app.post("/chat")
async def chat(req: ChatRequest, username: str = Depends(get_current_username)):
    start_all = time.time()
    session_id = req.session_id
    question = req.question.strip()
    selected_model = req.model if req.model in SUPPORTED_MODELS else DEFAULT_MODEL
    prompt_mode = req.prompt_mode if req.prompt_mode in ["default", "research"] else "default"

    if not question:
        logger.warning(f"⚠️ 用戶 {username} 收到空問題。")
        return {"error": "問題不能為空"}

    llm = get_model(selected_model)
    if llm is None: return {"error": f"無法載入語言模型 '{selected_model}'"}
    if vectordb is None:
        logger.error(f"❌ /chat error for user {username}: Vector database not available.")
        return {"error": "向量資料庫不可用"}

    user_chats_metadata = _load_user_chats_metadata(username)
    if session_id not in user_chats_metadata:
        now_iso = datetime.now().isoformat()
        default_title = f"對話 {session_id[:8]}"
        if question:
            first_message_title = question[:30] + "..." if len(question) > 30 else question
            default_title = first_message_title

        user_chats_metadata[session_id] = {"title": default_title, "created_at": now_iso, "updated_at": now_iso}
        _save_user_chats_metadata(username, user_chats_metadata)
        _save_user_chat_messages(username, session_id, [])
        logger.info(f"ℹ️ 自動為用戶 {username} 的 session_id '{session_id}' 創建了聊天元數據。標題: '{default_title}'")

    detected_format_mode = detect_format_mode(question)
    format_mode = detected_format_mode
    logger.info(f"🚀 Request - User: {username}, ChatID: {session_id}, Model: {selected_model}, PromptMode: {prompt_mode}, FormatMode: {format_mode}")
    logger.info(f"❓ Question: {question}")

    messages_for_prompt = _get_user_chat_messages(username, session_id)
    history_pairs = []
    temp_user_q = None
    for msg_dict in messages_for_prompt:
        if msg_dict["role"] == "user": temp_user_q = msg_dict["content"]
        elif msg_dict["role"] == "assistant" and temp_user_q:
            history_pairs.append((temp_user_q, msg_dict["content"]))
            temp_user_q = None
    history_text = "\n".join([f"使用者: {q}\n助理: {a}" for q, a in history_pairs[-MAX_HISTORY_PER_SESSION:]]) if history_pairs else "無歷史對話紀錄。"

    start_retrieve = time.time()
    retriever_k = 20; retriever_fetch_k = 50; retriever_lambda_mult = 0.8
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": retriever_k, "fetch_k": retriever_fetch_k, "lambda_mult": retriever_lambda_mult})
    context = ""; docs = []
    try:
        docs = retriever.invoke(question)
        if not docs: context = "沒有找到相關的背景資料。"
        else: context = "\n\n".join([doc.page_content for doc in docs])
        logger.info(f"⏱️ Retrieval time: {time.time() - start_retrieve:.2f}s, Found {len(docs)} docs.")
    except Exception as e:
        logger.error(f"❌ Retrieval error for user {username}, q='{question}': {str(e)}", exc_info=True)
        return {"error": f"向量資料庫檢索錯誤：{str(e)}"}
    
    selected_template = None; template_name = "N/A"
    if prompt_mode == "research":
        selected_template = RESEARCH_PROMPT_TEMPLATE
        template_name = f"Research ({'User Format' if format_mode == 'custom' else 'Default Format'})"
    elif format_mode == "custom":
        selected_template = CUSTOM_FORMAT_BASE_PROMPT
        template_name = "Custom Format Request"
    else:
        selected_template = random.choice(DEFAULT_PROMPT_OPTIONS)
        if selected_template == STRUCTURED_LIST_PROMPT: template_name = "Default Style (Random: Structured List)"
        elif selected_template == HIERARCHICAL_BULLETS_PROMPT: template_name = "Default Style (Random: Hierarchical Bullets)"
        elif selected_template == PARAGRAPH_EMOJI_LEAD_PROMPT: template_name = "Default Style (Random: Paragraph Emoji Lead)"
        else: template_name = "Default Style (Random: Unknown)"
    logger.info(f"Using template for user {username}: {template_name}")
    try:
        prompt = selected_template.format(context=context, question=question, history=history_text, format_mode=format_mode)
    except Exception as e:
        logger.error(f"❌ Prompt formatting error for user {username}: {e}", exc_info=True)
        return {"error": "內部錯誤：提示詞格式化失敗"}

    final_answer = None; llm_attempts = 0; start_llm_total = time.time()
    while llm_attempts <= MAX_LLM_RETRIES:
        try:
            start_llm_attempt = time.time()
            raw_answer = llm.invoke(prompt)
            processed_answer = post_process_answer(raw_answer.strip())
            logger.info(f"⏱️ LLM response (Attempt {llm_attempts + 1}) for user {username}: {time.time() - start_llm_attempt:.2f}s")
            if not processed_answer or processed_answer.isspace(): raise ValueError("LLM empty answer")
            final_answer = processed_answer; break
        except Exception as e:
            logger.error(f"❌ LLM error (Attempt {llm_attempts + 1}) for user {username}: {e}", exc_info=True)
            llm_attempts += 1
            if llm_attempts <= MAX_LLM_RETRIES: time.sleep(1)
            else:
                if final_answer is None: return {"error": "LLM 回應或處理失敗"}
    if final_answer is None: return {"error": "無法從 LLM 獲取回應"}

    current_chat_session_messages = _get_user_chat_messages(username, session_id)
    current_chat_session_messages.append({"role": "user", "content": question})
    current_chat_session_messages.append({"role": "assistant", "content": final_answer})
    _save_user_chat_messages(username, session_id, current_chat_session_messages)
    
    user_chats_metadata = _load_user_chats_metadata(username)
    if session_id in user_chats_metadata:
        user_chats_metadata[session_id]["updated_at"] = datetime.now().isoformat()
        _save_user_chats_metadata(username, user_chats_metadata)
    logger.info(f"用戶 {username} 的聊天 {session_id} 記錄已更新並保存。")

    if SAVE_QA:
        try:
            user_qa_log_dir = get_user_qa_log_path(username)
            today_str = datetime.now().strftime("%Y-%m-%d")
            qa_filename = user_qa_log_dir / f"qa_{today_str}.jsonl"
            qa_record = {
                "session_id": session_id, "timestamp": datetime.now().isoformat(),
                "model": selected_model, "prompt_mode": prompt_mode, "format_mode": format_mode,
                "question": question, "answer": final_answer, "llm_attempts": llm_attempts +1,
                "template_used": template_name, "retrieved_docs_count": len(docs)
            }
            with open(qa_filename, "a", encoding="utf-8") as f: f.write(json.dumps(qa_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"❌ Failed to save QA record for user {username}: {e}", exc_info=True)

    total_time = time.time() - start_all
    logger.info(f"⏱️ Total request time for user {username}: {total_time:.2f}s")
    return {
        "answer": final_answer, "model_used": selected_model,
        "prompt_mode_used": prompt_mode, "format_mode_used": format_mode,
        "template_style_used": template_name
    }

@app.post("/feedback")
async def submit_feedback_for_user(feedback: FeedbackRequest, username: str = Depends(get_current_username)):
    if not feedback.user_expected_answer:
        raise HTTPException(status_code=400, detail="預期正確回答不能為空")
    record = {
        "question": feedback.user_expected_question or feedback.question,
        "answer": feedback.user_expected_answer,
        "metadata": {
            "source": "manual_feedback", "original_question": feedback.question,
            "original_answer": feedback.original_answer, "session_id": feedback.session_id,
            "model_used": feedback.model, "timestamp": datetime.now().isoformat()
        }
    }
    try:
        user_feedback_dir = get_user_feedback_save_path(username)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = user_feedback_dir / f"feedback_{feedback.session_id}_{ts_str}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        logger.info(f"📝 用戶 {username} 的回饋已保存到 {filename}")
        return {"message": "✅ 使用者回饋已儲存", "filename": str(filename)}
    except Exception as e:
        logger.error(f"❌ 保存用戶 {username} 的回饋失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="儲存回饋失敗")

app.include_router(api_router)

# Command to run:
# uvicorn 5_10test:app --host 0.0.0.0 --port 8000 --reload 
# 目前最主要程式碼
