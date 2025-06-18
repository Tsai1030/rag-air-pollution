import torch
import pandas as pd
from datasets import Dataset
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import numpy as np # 新增導入 numpy 用於計算標準誤等

# from ragas.run_config import RunConfig # 如果之前有用到，確保導入

# 初始化設備
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 修改：LLM 初始化分離 ---
# 1. 用於答案生成和精煉的 LLM (可以選擇較大的模型)
answer_generation_llm_model_name = "gemma3:27b" # 這裡使用較大的模型 "gemma3:12b" -> 假設是筆誤，改回你之前的一個模型，或者確保 gemma3:12b 可用
# 你可以根據需要選擇一個更強大的模型用於生成
print(f"答案生成/精煉 LLM: {answer_generation_llm_model_name}")
answer_generation_llm = OllamaLLM( # 將原來的 model 變量更名以清晰化
    model=answer_generation_llm_model_name,
    timeout=1200, # 給生成模型更長的超時
    temperature=0,  # 常規要選0 如果是創新要提高
)

# 2. 用於 RAGAS 評估的 LLM (選擇一個穩定運行的較小模型)
ragas_evaluation_llm_model_name = "gemma2:9b-instruct-fp16" # 例如，你之前測試穩定的小模型  gemma2:9b-instruct-fp16
# ragas_evaluation_llm_model_name = "gemma2:2b-instruct-fp16" # 或者你提供的程式碼片段中使用的 "mistral:instruct" 如果你希望評估也用它
print(f"RAGAS 評估 LLM: {ragas_evaluation_llm_model_name}")
ragas_evaluation_llm = OllamaLLM(
    model=ragas_evaluation_llm_model_name,
    timeout=900, # 評估 LLM 的超時
    temperature=0,
)

# 將 RAGAS 評估 LLM 封裝起來
ragas_llm_for_evaluation = LangchainLLMWrapper(ragas_evaluation_llm) # <--- 使用 ragas_evaluation_llm

# --- 嵌入模型初始化 ---
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",  #"infgrad/stella-base-zh-v3-1792d",   #"BAAI/bge-m3"
    model_kwargs={"device": DEVICE}
)
ragas_emb = LangchainEmbeddingsWrapper(emb)

# --- 向量庫和檢索器 ---
vector_store = Chroma(
    persist_directory="6_17ragas",
    embedding_function=emb,
)
# retriever = vector_store.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"k": 10, "score_threshold": 0.5}
# )

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # 最終返回幾個片段 10
        "fetch_k": 30,     # 先抓幾個，再用 MMR 挑出 k 個     50
        "lambda_mult": 0.5 # 多樣性 vs 相似性 的平衡參數（0~1） 1
    }
)

#retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# 定義 QA prompt 模板
# template = """
# 使用下列上下文回答問題；若無答案，請回答「不知道」。

# 上下文:
# {context}

# 問題: {question}
# 回答:
# """

# template="""
# Please answer the following question **strictly based on** the information provided in the Context below.
# Do **not** add any external knowledge, inferred reasoning, or unstated assumptions.

# Write your answer in **3 to 4 short paragraphs**, each consisting of 2 to 3 well-formed sentences.
# Ensure the writing is **concise yet information-dense**, focusing on key facts, cause-and-effect relationships, and the core ideas explicitly mentioned in the context.

# Avoid using any formatting such as bullet points, numbered lists, titles, or headings.
# Use proper punctuation and complete sentences to maintain clear structure and readability.
# If the context includes trends, mechanisms, or international collaborations, briefly explain their significance and how they relate to the main subject.

# ⚠️ Your response must be written **only in Traditional Chinese**. Replies in English or Simplified Chinese will be considered invalid.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# 6/18使用
# template="""
# Please answer the following question **strictly based on** the information provided in the Context below.
# Do **not** add any external knowledge, inferred reasoning, or unstated assumptions.

# Write your answer in **3 to 4 short paragraphs**, each consisting of 2 to 3 well-formed sentences.
# Ensure the writing is **concise yet information-dense**, focusing on key facts, cause-and-effect relationships, and the core ideas explicitly mentioned in the context.

# Avoid using any formatting such as bullet points, numbered lists, titles, or headings.
# Use proper punctuation and complete sentences to maintain clear structure and readability.
# If the context includes trends, mechanisms, or international collaborations, briefly explain their significance and how they relate to the main subject.

# ⚠️ Your response must be written **only in Traditional Chinese**. Replies in English or Simplified Chinese will be considered invalid.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# 6/18測試
# template = """
# 請嚴格依據下方 Context 作答，不得加入任何推論或外部知識，也 **不可逐字貼出 Context 內容**。
# **思考【Question】中可能包含的多個信息點或子問題，並確保你的回答直接、簡潔且完整地覆蓋所有這些關鍵點。**

# - 答案長度：**最多 250 個中文字**，以 3 ~ 4 句完整句構成（不需分段落）。
# - 禁用格式：請勿使用項目符號、數字編號、標題，僅保留逗號、句號等基本標點。
# - 若 Context 涵蓋趨勢、機制或跨國合作，請簡要說明其與主題的關聯。
# - 回覆語言：**僅限繁體中文**，否則視為無效答案。

# 【Context】
# {context}

# 【Question】
# {question}

# 【Answer】
# """

# 6/18v2 (第一步生成初步答案的 Prompt)
initial_answer_template = """
請嚴格依據下方 Context 作答，不得加入任何推論或外部知識，也 **不可逐字貼出 Context 內容**。

**核心任務：針對【Question】中的每一個具體問題點和所有隱含的信息需求，從【Context】中提取相關信息，並組織成一個初步的答案。**

**輸出要求：**
- **直接性：** 答案應直接針對問題。
- **簡潔性：** **答案長度最多 250 個中文字**，以 3 ~ 4 句完整句構成（不需分段落）。
- **禁用格式：** 請勿使用項目符號、數字編號、標題，僅保留逗號、句號等基本標點。
- **特定信息處理：** 若 Context 涵蓋趨勢、機制或跨國合作，請在回答相關問題點時，簡要說明其與主題的關聯。
- **回覆語言：** **僅限繁體中文**，否則視為無效答案。

【Context】
{context}

【Question】
{question}

【Answer】
"""
initial_prompt = ChatPromptTemplate.from_template(initial_answer_template)
initial_chain = initial_prompt | answer_generation_llm # <--- 修改：使用 answer_generation_llm

# 新增：定義第二步精煉答案的 Prompt 模板
refinement_template = """
【原始問題】:
{original_question}

【相關上下文】:
{contexts_str}

【初步生成的答案】:
{initial_answer}

**任務：**
請仔細檢查上方的【初步生成的答案】。根據【相關上下文】，判斷【初步生成的答案】是否完整且準確地回答了【原始問題】的所有方面和所有關鍵信息點。

如果【初步生成的答案】有所遺漏或不夠完整：
1.  請從【相關上下文】中提取必要的補充信息。
2.  對【初步生成的答案】進行修正和擴充，生成一個更全面、更相關的最終答案。

如果【初步生成的答案】已經足夠完整和相關，則可以選擇直接使用它或做微小調整。

**最終答案輸出要求：**
- **完整性優先：** 最終答案必須全面回應【原始問題】的所有方面。
- **直接性：** 答案應直接針對問題。
- **簡潔性：** 在保證完整和直接的前提下，力求語言簡練，**最終答案長度仍需最多 250 個中文字**，以 3 ~ 4 句完整句構成（不需分段落）。
- **忠實性：** 所有信息必須來源於【相關上下文】，不得加入外部知識或推論，**不可逐字貼出上下文內容**。
- **禁用格式：** 請勿使用項目符號、數字編號、標題，僅保留逗號、句號等基本標點。
- **特定信息處理：** 若 Context 涵蓋趨勢、機制或跨國合作，請在回答相關問題點時，簡要說明其與主題的關聯。
- **回覆語言：** **僅限繁體中文**。

【最終答案】:
"""
refinement_prompt = ChatPromptTemplate.from_template(refinement_template)
refinement_chain = refinement_prompt | answer_generation_llm # <--- 修改：使用 answer_generation_llm


# 要評估的問題清單
questions = [
    "高醫USR空氣汙染計畫與哪所大學簽署合作協議共同推動多語言衛教教材？",
    "高醫USR空氣汙染計畫當前主持人是誰?",
    "高醫大USR空氣汙染計畫主要場域涵蓋的四個行政區名稱。",
    "以高雄醫學大學空氣污染 USR 計畫中的課程學分學程為例，請說明計畫中的「跨院系／跨校」課程設計如何影響學生的學習成效。",
    "高雄醫學大學空氣污染 USR 計畫在聯合國永續發展目標（SDGs）中，特別著重於「良好健康與福祉」（SDG 3）與「優質教育」（SDG 4）。請用說明為何這兩項目標會被作為本計畫的重點，並簡要說明其背後的意義。",
    "請說明「POWER」核心理念如何呼應大學社會責任（USR）的精神與目標。",
    "假設你是參與高雄醫學大學空氣污染 USR 課程的學生，請說明你如何運用課程中所學的知識與技能，協助社區推動「空污環境教育與健康促進」的衛教活動。請提出一個具體可行的活動流程。",
    "請根據高雄醫學大學空氣污染 USR 計畫的宗旨，說明你如何結合呼吸治療學系的專業知識，設計一項針對空污敏感族群（如高齡者、癌症存活者等）的健康介入服務方案。請說明服務對象、介入內容與執行方式。",
    "高雄醫學大學空氣污染 USR 計畫中已開發多種教育資源，例如多語繪本與肺健康衛教教材。假設你需要將這些素材推廣到非華語族群或不同文化背景社區，請設計三項具體的在地化應用策略，並簡要說明其操作方式與目的。",
    "比較「社區培力坊」與「空氣旅行」兩項方案在提升居民參與度上的異同，並指出各自的關鍵成功因素。",
    "高雄醫學大學空污 USR 計畫規劃了四大主軸：人才培育、社區健康、場域經營、創新產學。請將這四大主軸分別對應出其主要利害關係人（例如學生、教師、社區居民、企業等），並說明每一主軸如何回應其對象的需求或關注點。",
    "請解析「簡易經濟型油煙吸收桶」試辦前後對攤販作業環境的三項改善指標及其因果關係。",
    "請評估高雄醫學大學空污 USR 計畫對高中生「環境素養」與「公民參與度」的成效。",
    "從永續影響角度評估「高空污地區智慧肺健康門診」對地方公共衛生的長期效益，並提出佐證理由。",
    "請從「資源整合效率」、「在地實踐深度」與「永續發展潛力」三個面向，評估高雄醫學大學第四期空污 USR 計畫的整體推動成效。",
    "針對台灣新住民族群，提出一款多語化互動教材原型，能在 30 分鐘內完成空污防護知識學習並測驗學習成效。",
    "請根據USR計畫中數位教材應用經驗，設計一項結合空污即時資料與健康建議的數位互動工具。",
    "依照高醫大空氣污染 USR 計畫中已有的資源（如：智慧肺健康 App、肺健康圖卡、多語繪本、社區培力坊、跨校合作等），請設計一項三年期的創新行動方案，用於提升高風險社區居民在空氣污染下的健康素養與自我防護能力。"

]


# 逐題處理並生成樣本
samples = []
for question_text in questions: # 將 question 變量名改為 question_text 以避免與字典鍵衝突
    print(f"\n🔵 處理問題: {question_text}")
    query_text = f"query: {question_text}" # 雖然 bge-m3 通常不需要，但保留與你之前一致
    docs = retriever.invoke(query_text)
    context_docs_list = [d.page_content for d in docs]
    contexts_str_for_llm = "\n\n".join(context_docs_list) # 合併上下文為單一字符串給 LLM

    # 第一步：生成初步答案
    print("📝 第一步：生成初步答案...")
    initial_answer_content = initial_chain.invoke({ # initial_chain 現在使用 answer_generation_llm
        "context": contexts_str_for_llm,
        "question": question_text
    })
    print(f"💬 初步答案: {initial_answer_content}")

    # 第二步：精煉答案
    print("✨ 第二步：精煉答案...")
    final_answer_content = refinement_chain.invoke({ # refinement_chain 現在使用 answer_generation_llm
        "original_question": question_text,
        "contexts_str": contexts_str_for_llm,
        "initial_answer": initial_answer_content
    })
    print(f"✅ 最終答案: {final_answer_content}")
    print("📸 回答完成！")

    sample = {
        "question": question_text,
        "contexts": context_docs_list,
        "answer": final_answer_content,
        "reference": "\n\n".join(context_docs_list)
    }
    samples.append(sample)

# RAGAS 運行配置 (如果需要，從之前代碼複製過來)
# from ragas.run_config import RunConfig
# ragas_run_config = RunConfig(timeout=850, max_workers=1, max_retries=2)


# 安全評估函式
def safe_evaluate(sample_dataset, question_str): # 將 question 變量名改為 question_str
    try:
        print(f"\n🔍 評估問題: {question_str}")
        result = evaluate(
            sample_dataset,
            metrics=[faithfulness, context_precision, answer_relevancy],
            llm=ragas_llm_for_evaluation, # <--- 修改：使用分離的 ragas_llm_for_evaluation
            embeddings=ragas_emb
            # , run_config=ragas_run_config # 如果定義了 run_config 則取消註解
        )
        df_result = result.to_pandas() # 將 RagasResult 轉換為 DataFrame
        df_result["question"] = question_str # 為 DataFrame 添加 question 列
        return df_result
    except Exception as e:
        print(f"⚠️ 評估失敗 ({question_str})，自動啟動修正機制: {e}")
        import traceback
        traceback.print_exc()
        fallback_data = { # 創建一個包含所有指標和 question 的字典
            "faithfulness": [float("nan")],
            "context_precision": [float("nan")],
            "answer_relevancy": [float("nan")],
            "question": [question_str]
        }
        return pd.DataFrame(fallback_data) # 直接從字典創建 DataFrame

# 單題逐筆進行 RAGAS 評估
# (這部分代碼與之前相同，此處省略以節省篇幅，請確保它在你的腳本中)
print("\n⚙️ 開始單題逐筆 RAGAS 評估...")
all_results_dfs = [] # 改為存儲 DataFrame 對象
if not samples:
    print("⚠️ 沒有生成任何樣本，無法進行 RAGAS 評估。")
else:
    for i, sample_data in enumerate(samples):
        # 確保 RAGAS 需要的鍵存在且類型正確
        current_question = sample_data.get('question', f'未知問題_{i+1}')
        if not isinstance(current_question, str):
            print(f"  警告: 第 {i+1} 個樣本的 'question' 類型錯誤，跳過。")
            continue

        if "contexts" not in sample_data or not isinstance(sample_data["contexts"], list):
            print(f"  警告: 第 {i+1} 個樣本 ({current_question}) 的 'contexts' 不是列表或不存在，設為空列表。")
            sample_data["contexts"] = [] # 強制設為空列表以避免 Dataset 錯誤
        if "answer" not in sample_data or not isinstance(sample_data["answer"], str):
            print(f"  警告: 第 {i+1} 個樣本 ({current_question}) 的 'answer' 不是字符串或不存在，設為空字符串。")
            sample_data["answer"] = "" # 強制設為空字串

        # 確保 sample_data 包含 RAGAS 需要的所有基本列，即使是空值
        for required_col in ["question", "answer", "contexts", "reference"]:
            if required_col not in sample_data:
                if required_col == "contexts" or required_col == "reference":
                    sample_data[required_col] = [] if required_col == "contexts" else ""
                else:
                    sample_data[required_col] = "" # 或其他合適的默認值


        one_dataset = Dataset.from_list([sample_data])
        df_one = safe_evaluate(one_dataset, current_question)
        all_results_dfs.append(df_one)
        print(f"✅ 第 {i+1} 題完成 ({current_question})")


# --- 修改：合併並分析結果，加入更多統計數據 ---
if all_results_dfs:
    final_df = pd.concat(all_results_dfs, ignore_index=True)
    print("\n📊 完整 RAGAS 分數 (每題)：")
    cols_to_print = ["question"]
    metric_names_for_stats = []
    for metric_name in ["faithfulness", "context_precision", "answer_relevancy"]:
        if metric_name in final_df.columns:
            cols_to_print.append(metric_name)
            metric_names_for_stats.append(metric_name)
    print(final_df[cols_to_print])

    print("\n📈 各項指標統計分析：")
    summary_stats_list = []

    for metric in metric_names_for_stats:
        # 確保該列存在並且是數值類型，處理 NaN 值
        if metric in final_df.columns and pd.api.types.is_numeric_dtype(final_df[metric]):
            metric_series = final_df[metric].dropna() # 移除 NaN 值進行統計計算
            if not metric_series.empty:
                mean_val = metric_series.mean()
                std_val = metric_series.std()
                sem_val = metric_series.sem() if len(metric_series) > 0 else float('nan') # 標準誤
                median_val = metric_series.median()
                min_val = metric_series.min()
                max_val = metric_series.max()
                count_val = len(metric_series) # 有效數據點數量

                print(f"\n--- {metric} ---")
                print(f"  樣本數 (Count)    : {count_val}")
                print(f"  平均值 (Mean)      : {mean_val:.4f}")
                print(f"  標準差 (Std Dev)   : {std_val:.4f}")
                print(f"  標準誤 (SEM)       : {sem_val:.4f}")
                print(f"  中位數 (Median)    : {median_val:.4f}")
                print(f"  最小值 (Min)       : {min_val:.4f}")
                print(f"  最大值 (Max)       : {max_val:.4f}")

                summary_stats_list.append({
                    "Metric": metric,
                    "Count": count_val,
                    "Mean": mean_val,
                    "Std Dev": std_val,
                    "SEM": sem_val,
                    "Median": median_val,
                    "Min": min_val,
                    "Max": max_val
                })
            else:
                print(f"\n--- {metric} ---")
                print(f"  所有值均為 NaN 或為空，無法計算統計數據。")
                summary_stats_list.append({"Metric": metric, "Mean": float('nan')}) # 至少記錄平均值為 NaN
        else:
            print(f"\n--- {metric} ---")
            print(f"  指標列不存在或非數值類型，無法計算統計數據。")
            summary_stats_list.append({"Metric": metric, "Mean": float('nan')})


    # 計算總體平均分 (基於有效的平均值)
    valid_means_for_overall = [s['Mean'] for s in summary_stats_list if 'Mean' in s and not pd.isna(s['Mean'])]
    if valid_means_for_overall:
        overall_score = sum(valid_means_for_overall) / len(valid_means_for_overall)
        print(f"\n🏆 總分（所有指標的平均值的平均）: {overall_score:.4f}")
    else:
        print("\n🏆 總分（所有指標的平均值的平均）: 未計算 (無有效指標平均值)")

    # 將詳細統計數據也保存到 CSV (可選)
    if summary_stats_list:
        summary_df = pd.DataFrame(summary_stats_list)
        print("\n📋 各項指標統計摘要表：")
        print(summary_df)
        summary_df.to_csv("ragas_summary_stats.csv", index=False, encoding="utf-8-sig")
        print("\n💾 統計摘要已儲存到 ragas_summary_stats.csv")


    if not final_df.empty:
        final_df.to_csv("ragas_detailed_results_refined.csv", index=False, encoding="utf-8-sig") # 改個文件名
        print("\n💾 詳細結果已儲存到 ragas_detailed_results_refined.csv")
    else:
        print("\n💾 詳細結果為空，未儲存 CSV 文件。")
else:
    print("\n⚠️ 沒有成功的評估結果，無法合併！")
