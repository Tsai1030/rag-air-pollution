<!--
title: RAG QA System for Air Pollution
description: A LangChain + Chroma + Ollama powered retrieval-augmented chatbot system for answering questions about air pollution, environmental policies, and health effects.
keywords: RAG, LangChain, air pollution, chatbot, QA system, fastapi, react, chromadb, gemma, qwen
-->

# 🌱 RAG 空氣污染問答系統 (RAG-based QA System on Air Pollution)

本專案為一套結合語意檢索（Retrieval-Augmented Generation, RAG）與生成式 AI 的智慧型問答系統，針對空氣污染政策、健康影響與環境法規等主題，提供即時、精準且具可查證性的回應。使用者可透過網頁前端介面自由提問，系統將透過語意向量檢索技術，從本地文件知識庫中找出相關內容，並結合大型語言模型進行答案生成。

本系統後端以 FastAPI 為基礎，整合 LangChain 框架與 Chroma 向量資料庫，實作語意檢索與知識融合邏輯。Embedding 向量模型由 HuggingFace 匯入（如 `BAAI/bge-m3`），具備高效中文語意理解能力；本地大型語言模型則透過 Ollama 部屬，支援如 Gemma 3:12B 模型進行回應生成，具備穩定、高效且離線可執行的優勢。

前端介面使用 React + Vite 開發，支援即時提問、Markdown 回應格式呈現、歷史紀錄與使用者回饋機制。整體系統採前後端分離架構，支援多模型切換、語意斷句、文件結構化與自動評估指標整合。

系統中特別設計了一套 Prompt 策略模組，可隨機選擇不同風格的回答格式（如條列式、段落式、Emoji 風格等），以提升回答多樣性與親和力；若使用者於提問中明確指定回應格式（如「請用條列式回覆」），系統將優先依照指令執行，確保符合需求並提升互動體驗。

---
## 線上RAG系統測試  歡迎提供任何建議 感謝
http://163.15.172.93:3000/

## 🔄 資料處理
資料切割上 設定QA的文本是不進行切割的，只切割長文本
本系統的長文本資料會先進行結構化處理，依照段落標題（`title`）進行分類與整理，轉換為標準化的 JSON 格式，包含 `title`、`content`、`doc_id`、`source` 等欄位，以利後續進行語意檢索與生成。

- 長篇資料會使用語意斷句工具進行分段，再依設計的 chunk 大小進行切割與向量化處理。
- QA 格式的資料不進行切割，完整保留原始問答對，維持語境連貫性與回應品質。
- 透過這種結構化與分類處理方式，有助於提升查詢準確性與系統回覆的可解釋性。

> 🔧 設定上，QA 的文本不進行切割，僅針對長文本進行 chunk 處理。

## 📁 專案結構

```
rag-air-project/
├── .git/                   # Git 版本控制資料夾（自動產生）
├── .gitignore              # Git 忽略規則設定檔
├── environment.yml         # Conda 環境建立設定檔
├── README.md               # 專案說明文件
├── backend/                # FastAPI 後端程式碼
├── frontend/               # React + Vite 前端程式碼
│   └── public/images/      # 前端展示用圖片（包含展示照片.png）
```

## 🐍 Conda 環境建立說明

請確認你已安裝 [Anaconda](https://www.anaconda.com/) 或 Miniconda，然後執行以下指令建立專案環境：


conda 建立說明 打包環境設定安裝

```bash
conda env create -f environment.yml
conda activate test2
```
🚀 快速啟動指南

🧩 前端（React + Vite）
進入前端目錄：
cd frontend

安裝依賴套件：
npm install

啟動開發伺服器：
npm run dev

打開瀏覽器前往：
http://localhost:5173

⚙️ 後端（FastAPI + LangChain + ChromaDB）
進入後端資料夾：
cd backend

啟動 Conda 環境（已於前面建立）：
conda activate test2

啟動後端伺服器：
uvicorn main:app --reload

打開 API Swagger 文件頁面：
http://127.0.0.1:8000/docs

🧠 技術架構說明

| 元件         | 說明                                                   |
| ---------- | ---------------------------------------------------- |
| **RAG 架構** | 使用 LangChain 整合 Chroma 向量資料庫與本地 LLM 模型（如 Gemma、Qwen） |
| **語意檢索**   | 採用 BGE-m3 或 all-mpnet-base-v2 向量模型進行文件嵌入與查詢          |
| **回答生成**   | 結合查詢結果與自訂 Prompt，透過本地部署 LLM 生成答案                     |
| **前端介面**   | 使用 Vite 建立的 React 前端，支援提問、回覆呈現與格式化 Markdown          |
| **模型部署**   | 搭配 Ollama 進行本地 LLM 部署，可支援 Gemma 3B、Qwen 14B 等模型      |
| **評估機制**   | 可搭配 RAGAS 框架進行忠實度、語意相關性等指標的自動評估                      |


## 🌈 系統展示畫面

![RAG 系統展示圖](https://github.com/Tsai1030/rag-air-pollution/blob/main/frontend/public/images/%E5%B1%95%E7%A4%BA%E7%85%A7%E7%89%87.png?raw=true)

## Contact

🐈 蔡承紘 Cheng-Hung, Tsai

📧 Email : pijh102511@gmail.com


