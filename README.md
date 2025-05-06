# 🌱 RAG 空氣污染問答系統 (RAG-based QA System on Air Pollution)

本專案為一套結合語意檢索與生成式 AI 的問答系統，針對空氣污染政策、健康影響與環境法規等主題，提供精準、可查證的回應。使用者可透過前端網頁介面進行提問，系統將利用 RAG 技術結合本地文件資料庫進行語意查詢與回答生成。

---

## 🔄 資料處理
資料切割上 設定QA的文本是不進行切割的，只切割長文本

## 📁 專案結構

rag-air-project/
├── .git/                   # Git 版本控制資料夾（自動產生）
├── .gitignore              # Git 忽略規則設定檔
├── environment.yml         # Conda 環境建立設定檔
├── README.md               # 專案說明文件
├── backend/                # FastAPI 後端程式碼
├── frontend/               # React + Vite 前端程式碼


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

