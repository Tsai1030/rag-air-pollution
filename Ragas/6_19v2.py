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
answer_generation_llm_model_name = "gemma3:27b" # 使用更大更穩定的模型  # mistral-small3.1:latest 1,1,72還不錯
print(f"答案生成/精煉 LLM: {answer_generation_llm_model_name}")
answer_generation_llm = OllamaLLM(
    model=answer_generation_llm_model_name,
    timeout=1800, # 增加超時時間
    temperature=0,
    top_p=0.9,  # 加入 top_p 控制
    repeat_penalty=1.1  # 避免重複
)

# 2. 用於 RAGAS 評估的 LLM (選擇一個穩定運行的較小模型)
ragas_evaluation_llm_model_name = "llama3.1:8b-instruct-fp16"#"gemma2:9b-instruct-fp16"
print(f"RAGAS 評估 LLM: {ragas_evaluation_llm_model_name}")
ragas_evaluation_llm = OllamaLLM(
    model=ragas_evaluation_llm_model_name,
    timeout=1200,
    temperature=0,
    top_p=0.9,
    repeat_penalty=1.1
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

# 改善的階層式檢索策略
def hierarchical_retrieval(question, vector_store, top_k=3):
    """階層式檢索策略：先高閾值精確檢索，不足時降低閾值補充"""
    # 第一層：高閾值精確檢索
    high_threshold_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": 0.7}
    )
    
    docs_high = high_threshold_retriever.invoke(f"query: {question}")
    
    # 如果高閾值檢索結果不足，用中等閾值補充
    if len(docs_high) < top_k:
        medium_threshold_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": top_k*2, "score_threshold": 0.8}
        )
        docs_medium = medium_threshold_retriever.invoke(f"query: {question}")
        
        # 移除重複文檔
        existing_contents = {doc.page_content for doc in docs_high}
        docs_medium_filtered = [doc for doc in docs_medium if doc.page_content not in existing_contents]
        
        # 合併結果
        docs_combined = docs_high + docs_medium_filtered[:top_k-len(docs_high)]
        
        # 如果仍然不足，使用 MMR 補充
        if len(docs_combined) < top_k:
            mmr_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": 15,
                    "lambda_mult": 0.8  # 更重視相似性
                }
            )
            docs_mmr = mmr_retriever.invoke(f"query: {question}")
            
            # 移除重複文檔
            existing_contents = {doc.page_content for doc in docs_combined}
            docs_mmr_filtered = [doc for doc in docs_mmr if doc.page_content not in existing_contents]
            
            docs_combined.extend(docs_mmr_filtered[:top_k-len(docs_combined)])
        
        return docs_combined[:top_k]
    
    return docs_high

# 定義問題分析和關鍵要素提取的 Prompt
question_analysis_template = """
請分析以下問題，提取出所有關鍵信息點和子問題：

問題：{question}

請列出：
1. 主要詢問的核心概念
2. 具體的信息需求點
3. 可能的子問題

分析結果：
"""
question_analysis_prompt = ChatPromptTemplate.from_template(question_analysis_template)
question_analysis_chain = question_analysis_prompt | answer_generation_llm

# 優化的初步答案生成 Prompt (加強忠實度約束)
initial_answer_template = """
【嚴格遵循規則】：
1. 答案內容必須100%來自下方Context，絕對不得添加任何外部知識、推論或假設
2. 如果Context中沒有明確資訊回答問題的任何部分，必須明確說明「根據提供資料無法確定該部分」
3. 只能陳述Context中的明確事實，不得進行任何形式的推論或延伸
4. 不可逐字複製Context內容，需要重新組織語言但保持原意

【問題分析】：{question_analysis}

【核心任務】：針對【Question】中的每一個具體問題點，從【Context】中尋找對應的明確信息，組織成完整答案。

【Context】
{context}

【Question】
{question}

【輸出要求】：
- 直接回應問題的所有關鍵點
- 最多250個中文字，3-4句完整句子
- 僅使用基本標點符號
- 僅限繁體中文
- 如遇趨勢、機制或跨國合作等資訊，說明其與主題的關聯

【Answer】
"""
initial_prompt = ChatPromptTemplate.from_template(initial_answer_template)
initial_chain = initial_prompt | answer_generation_llm

# 答案品質檢查 Prompt
quality_check_template = """
請檢查以下答案是否完全基於提供的Context，沒有添加外部知識：

【Context】
{context}

【Question】
{question}

【Generated Answer】
{answer}

【檢查標準】：
1. 答案中的每個事實是否都能在Context中找到對應
2. 是否包含Context以外的資訊
3. 是否有推論或假設的內容

如果答案完全基於Context，回答「通過」
如果答案包含Context以外的資訊，回答「未通過：[具體指出問題所在]」

【檢查結果】：
"""
quality_check_prompt = ChatPromptTemplate.from_template(quality_check_template)
quality_check_chain = quality_check_prompt | answer_generation_llm

# 強化版精煉答案 Prompt
refinement_template = """
【原始問題】:
{original_question}

【問題分析】:
{question_analysis}

【相關上下文】:
{contexts_str}

【初步生成的答案】:
{initial_answer}

【品質檢查結果】:
{quality_check_result}

**任務：**
基於品質檢查結果，對初步答案進行必要的修正和完善：

1. 如果檢查結果為「通過」：檢視答案是否完整回應了問題分析中的所有關鍵點，如有遺漏請補充
2. 如果檢查結果為「未通過」：移除所有不基於Context的內容，重新基於Context組織答案
3. 確保答案直接、完整地回應原始問題的所有方面

**最終答案要求：**
- 100%基於【相關上下文】的明確資訊
- 完整回應【問題分析】中的所有關鍵點
- 直接針對【原始問題】
- 最多250個中文字，3-4句完整句子
- 僅使用基本標點符號，僅限繁體中文
- 語言簡練但信息完整

【最終答案】:
"""
refinement_prompt = ChatPromptTemplate.from_template(refinement_template)
refinement_chain = refinement_prompt | answer_generation_llm

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

# 優化的樣本生成流程
samples = []
for i, question_text in enumerate(questions):
    print(f"\n🔵 處理問題 {i+1}/{len(questions)}: {question_text}")
    
    # 步驟1：問題分析
    print("📋 步驟1：分析問題關鍵要素...")
    question_analysis_result = question_analysis_chain.invoke({
        "question": question_text
    })
    print(f"💡 問題分析: {question_analysis_result}")
    
    # 步驟2：階層式檢索
    print("🔍 步驟2：執行階層式檢索...")
    docs = hierarchical_retrieval(question_text, vector_store, top_k=3)
    context_docs_list = [d.page_content for d in docs]
    contexts_str_for_llm = "\n\n".join(context_docs_list)
    print(f"📄 檢索到 {len(docs)} 個相關文檔")

    # 步驟3：生成初步答案
    print("📝 步驟3：生成初步答案...")
    initial_answer_content = initial_chain.invoke({
        "context": contexts_str_for_llm,
        "question": question_text,
        "question_analysis": question_analysis_result
    })
    print(f"💬 初步答案: {initial_answer_content}")

    # 步驟4：品質檢查
    print("🔍 步驟4：執行品質檢查...")
    quality_check_result = quality_check_chain.invoke({
        "context": contexts_str_for_llm,
        "question": question_text,
        "answer": initial_answer_content
    })
    print(f"✅ 品質檢查: {quality_check_result}")

    # 步驟5：精煉最終答案
    print("✨ 步驟5：精煉最終答案...")
    final_answer_content = refinement_chain.invoke({
        "original_question": question_text,
        "question_analysis": question_analysis_result,
        "contexts_str": contexts_str_for_llm,
        "initial_answer": initial_answer_content,
        "quality_check_result": quality_check_result
    })
    print(f"🎯 最終答案: {final_answer_content}")
    print("=" * 80)

    sample = {
        "question": question_text,
        "contexts": context_docs_list,
        "answer": final_answer_content,
        "reference": "\n\n".join(context_docs_list)
    }
    samples.append(sample)

# 安全評估函式
def safe_evaluate(sample_dataset, question_str):
    try:
        print(f"\n🔍 評估問題: {question_str}")
        result = evaluate(
            sample_dataset,
            metrics=[faithfulness, context_precision, answer_relevancy],
            llm=ragas_llm_for_evaluation,
            embeddings=ragas_emb
        )
        df_result = result.to_pandas()
        df_result["question"] = question_str
        return df_result
    except Exception as e:
        print(f"⚠️ 評估失敗 ({question_str})，自動啟動修正機制: {e}")
        import traceback
        traceback.print_exc()
        fallback_data = {
            "faithfulness": [float("nan")],
            "context_precision": [float("nan")],
            "answer_relevancy": [float("nan")],
            "question": [question_str]
        }
        return pd.DataFrame(fallback_data)

# 分批評估策略 (先測試前3題)
print("\n⚙️ 開始分批 RAGAS 評估...")
print("🔬 第一階段：測試前3題...")

test_samples = samples[:3]  # 先測試前3個問題
test_results_dfs = []

for i, sample_data in enumerate(test_samples):
    current_question = sample_data.get('question', f'未知問題_{i+1}')
    
    # 數據驗證
    for required_col in ["question", "answer", "contexts", "reference"]:
        if required_col not in sample_data:
            if required_col == "contexts":
                sample_data[required_col] = []
            else:
                sample_data[required_col] = ""

    one_dataset = Dataset.from_list([sample_data])
    df_one = safe_evaluate(one_dataset, current_question)
    test_results_dfs.append(df_one)
    print(f"✅ 測試題 {i+1}/3 完成")

# 分析測試結果
if test_results_dfs:
    test_df = pd.concat(test_results_dfs, ignore_index=True)
    print("\n📊 測試結果分析：")
    
    metric_names = ["faithfulness", "context_precision", "answer_relevancy"]
    test_means = {}
    
    for metric in metric_names:
        if metric in test_df.columns:
            metric_series = test_df[metric].dropna()
            if not metric_series.empty:
                mean_val = metric_series.mean()
                test_means[metric] = mean_val
                print(f"{metric}: {mean_val:.4f}")
    
    # 判斷是否繼續完整評估
    target_threshold = 0.75  # 測試階段的目標閾值
    all_above_threshold = all(score >= target_threshold for score in test_means.values())
    
    if all_above_threshold:
        print(f"\n🎉 測試結果良好！所有指標均超過 {target_threshold}，開始完整評估...")
        
        # 完整評估
        print("\n🚀 第二階段：完整評估所有問題...")
        all_results_dfs = []
        
        for i, sample_data in enumerate(samples):
            current_question = sample_data.get('question', f'未知問題_{i+1}')
            
            # 數據驗證
            for required_col in ["question", "answer", "contexts", "reference"]:
                if required_col not in sample_data:
                    if required_col == "contexts":
                        sample_data[required_col] = []
                    else:
                        sample_data[required_col] = ""

            one_dataset = Dataset.from_list([sample_data])
            df_one = safe_evaluate(one_dataset, current_question)
            all_results_dfs.append(df_one)
            print(f"✅ 完整評估 {i+1}/{len(samples)} 完成")
        
        # 最終結果分析
        final_df = pd.concat(all_results_dfs, ignore_index=True)
        
    else:
        print(f"\n⚠️ 測試結果未達標準，建議進一步調整參數後重新測試")
        print("目前測試結果：")
        for metric, score in test_means.items():
            status = "✅" if score >= target_threshold else "❌"
            print(f"  {status} {metric}: {score:.4f}")
        final_df = test_df  # 使用測試結果作為最終結果

else:
    print("\n⚠️ 沒有成功的評估結果！")
    final_df = pd.DataFrame()

# 最終結果統計和保存
if not final_df.empty:
    print("\n📊 最終 RAGAS 評估結果：")
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
        if metric in final_df.columns and pd.api.types.is_numeric_dtype(final_df[metric]):
            metric_series = final_df[metric].dropna()
            if not metric_series.empty:
                mean_val = metric_series.mean()
                std_val = metric_series.std()
                sem_val = metric_series.sem() if len(metric_series) > 0 else float('nan')
                median_val = metric_series.median()
                min_val = metric_series.min()
                max_val = metric_series.max()
                count_val = len(metric_series)

                print(f"\n--- {metric} ---")
                print(f"  樣本數 (Count)    : {count_val}")
                print(f"  平均值 (Mean)      : {mean_val:.4f}")
                print(f"  標準差 (Std Dev)   : {std_val:.4f}")
                print(f"  標準誤 (SEM)       : {sem_val:.4f}")
                print(f"  中位數 (Median)    : {median_val:.4f}")
                print(f"  最小值 (Min)       : {min_val:.4f}")
                print(f"  最大值 (Max)       : {max_val:.4f}")
                
                # 目標達成檢查
                target_status = "🎯 達標" if mean_val >= 0.80 else "📈 需改善"
                print(f"  目標達成狀態        : {target_status}")

                summary_stats_list.append({
                    "Metric": metric,
                    "Count": count_val,
                    "Mean": mean_val,
                    "Std Dev": std_val,
                    "SEM": sem_val,
                    "Median": median_val,
                    "Min": min_val,
                    "Max": max_val,
                    "Target_80_Achieved": mean_val >= 0.80
                })

    # 計算總體平均分
    valid_means_for_overall = [s['Mean'] for s in summary_stats_list if 'Mean' in s and not pd.isna(s['Mean'])]
    if valid_means_for_overall:
        overall_score = sum(valid_means_for_overall) / len(valid_means_for_overall)
        overall_status = "🏆 目標達成！" if overall_score >= 0.80 else "📊 持續改善中"
        print(f"\n🏆 總分（所有指標平均）: {overall_score:.4f} - {overall_status}")
        
        # 達標統計
        achieved_count = sum(1 for s in summary_stats_list if s.get('Target_80_Achieved', False))
        total_metrics = len(summary_stats_list)
        print(f"📊 達標指標數量: {achieved_count}/{total_metrics}")

    # 保存結果
    if summary_stats_list:
        summary_df = pd.DataFrame(summary_stats_list)
        summary_df.to_csv("ragas_optimized_summary_stats.csv", index=False, encoding="utf-8-sig")
        print(f"\n💾 優化後統計摘要已儲存到 ragas_optimized_summary_stats.csv")

    final_df.to_csv("ragas_optimized_detailed_results.csv", index=False, encoding="utf-8-sig")
    print(f"💾 優化後詳細結果已儲存到 ragas_optimized_detailed_results.csv")
else:
    print("\n⚠️ 沒有有效的評估結果可供分析！")
    # 目前為主要最佳模型
