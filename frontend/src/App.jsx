import { useState, useEffect, useRef } from "react";
import axios from "axios";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("gemma3:12b-it-q4_K_M");
  const [promptMode, setPromptMode] = useState("default");
  const [expectedQuestion, setExpectedQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const bottomRef = useRef(null);
  const textAreaRef = useRef(null); // 新增 textarea 的 ref
  const sessionId = "user-session-001";

  // 新增自動調整 textarea 高度的函數，確保文字不會被按鈕覆蓋
  const adjustTextAreaHeight = () => {
    const textArea = textAreaRef.current;
    if (textArea) {
      textArea.style.height = "auto"; // 先重置高度
      
      // 計算實際內容高度並添加底部間距，確保文字不會被按鈕覆蓋
      // 這裡我們會額外加上足夠的空間，以確保最後兩行文字不會被按鈕覆蓋
      const contentHeight = textArea.scrollHeight;
      const buttonClearance = 56; // 約 56px 的按鈕高度空間
      
      const newHeight = Math.max(100, Math.min(200, contentHeight));
      textArea.style.height = `${newHeight}px`;
    }
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);
  
    try {
      const res = await axios.post("http://163.15.172.93:8000/chat", {
        session_id: sessionId,
        question: input,
        model: selectedModel,
        prompt_mode: promptMode,
      });
      const answer = res.data.answer;
      
      console.log("DEBUG: Received from backend:", answer); // <-- 加入 Log

      const cleaned = answer  
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>") // 如果後端沒轉 <strong>，這裡可以保留
      .replace(/^\s*\*\s+/gm, "• "); // 如果後端沒轉 •，這裡可以保留                  // * 列點轉項目符號 // 5/2修正到的部分
      
      console.log("DEBUG: Before setting state:", cleaned); // <-- 加入 Log
      
      setMessages([...newMessages, { role: "assistant", content: cleaned }]);
    } catch (err) {
      setMessages([...newMessages, { role: "assistant", content: "❌ 回答失敗，請稍後再試。" }]);
    } finally {
      setLoading(false);
    }
  };
  

  const submitFeedback = async () => {
    if (!expectedAnswer.trim()) return;

    const original = messages[messages.length - 1];
    try {
      await axios.post("http://163.15.172.93:8000/feedback", {
        session_id: sessionId,
        question: input,
        model: selectedModel,
        original_answer: original?.content || "",
        user_expected_question: expectedQuestion,
        user_expected_answer: expectedAnswer,
      });
      alert("✅ 已送出回饋");
      setExpectedQuestion("");
      setExpectedAnswer("");
    } catch (err) {
      alert("❌ 無法提交回饋");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 監聽 input 變化時調整 textarea 高度
  useEffect(() => {
    adjustTextAreaHeight();
  }, [input]);

  return (
    <div className="flex flex-col h-screen bg-[#F9F8F3] text-black">
      <header className="p-4 text-xl font-bold bg-[#F9F8F3] shadow flex items-center gap-3">
        <img src="/images/KMU_logo.png" alt="KMU Logo" className="w-10 h-10 object-contain" />
        空氣汙染檢索增強(目前系統測試中較不穩定)請仔細閱讀輸出結果
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4 flex flex-col font-serif">
        {messages.map((msg, i) => (
          <div key={i} className="flex justify-center">
            <div className="flex items-start gap-2 max-w-[800px] w-full">
              {msg.role === "user" && (
                <img src="/images/KMU_logo.png" alt="KMU" className="w-6 h-6 mt-1 rounded-full object-contain shrink-0" />
              )}
              <div
                className={
                  "px-4 py-2 rounded-lg whitespace-pre-wrap prose text-black max-w-[90%] "+
                  (msg.role === "user" ? "bg-[#F2EFE7]" : "bg-[#F9F8F3]")
                }
                dangerouslySetInnerHTML={{ __html: msg.content }}
              ></div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-center">
            <div className="flex items-start gap-2 max-w-[800px] w-full justify-start">
              <div className="text-gray-400 flex items-center space-x-2 font-serif max-w-[90%]">
                <img src="/images/claude-color.png" alt="loading icon" className="w-5 h-5 animate-spin" />
                <span>AI 正在思考中...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <footer className="p-4 bg-[#F9F8F3] flex flex-col items-center font-serif" style={{ backgroundColor: 'transparent' }}>
        <div className="relative w-[800px]" style={{ backgroundColor: 'transparent' }}>

            {/* ✅ 使用說明下載區塊 */}
            <div className="fixed bottom-14 right-4 text-xs bg-white px-4 py-2 rounded shadow border border-gray-300 z-50">
              📄 <a
                href="/documents/空氣污染RAG系統使用說明書.pdf"
                download
                className="text-[#D38A74] underline hover:text-[#b34c32]"
              >
                點我下載使用說明書 (PDF)
              </a>
            </div>

          {/* ✅ 聯絡資訊 */}
          <div className="fixed bottom-3 right-4 text-xs text-gray-600 bg-white px-3 py-1 rounded shadow border border-gray-300 z-50">
            如有系統建議與錯誤回報，也歡迎來信：<br />
            📧 <a href="mailto:your_email@example.com" className="underline hover:text-[#D38A74]">pijh102511@gmail.com</a>
          </div>

          {/* ✅ 模型下拉選單 */}
          <div className="absolute bottom-[20px] right-[60px] z-10" style={{ backgroundColor: 'transparent', boxShadow: 'none' }}>
            <div className="relative w-[320px]" style={{ backgroundColor: 'transparent' }}>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="appearance-none bg-transparent text-black text-sm pr-8 py-2 rounded-md w-full focus:outline-none focus:ring-0 focus:border-0 hover:ring-0 hover:bg-transparent transition-all"
              style={{ 
                textAlignLast: "right", 
                backgroundColor: 'transparent', 
                boxShadow: 'none',
                border: 'none'
              }}
            >
                <option value="gemma3:12b-it-q4_K_M">gemma3:12b-it-q4_K_M</option>
                <option value="gemma3:12b">Gemma 3 12B</option>
                <option value="qwen2.5:14b">Qwen 2.5 14B</option>
                <option value="mistral-small3.1:24b-instruct-2503-q4_K_M">mistral-small3.1:24b-instruct-2503-q4_K_M</option>
                <option value="qwen2.5:14b-instruct-q5_K_M">qwen2.5:14b-instruct-q5_K_M(測試)</option>
                <option value="phi4-mini-reasoning:3.8b">phi4-mini-reasoning:3.8b(測試)</option>
              </select>
              <div className="pointer-events-none absolute top-1/2 right-2 transform -translate-y-1/2" style={{ backgroundColor: 'transparent', boxShadow: 'none' }}>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-4 h-4 text-gray-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  style={{ backgroundColor: 'transparent', boxShadow: 'none' }}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                </svg>
              </div>
            </div>
          </div>
          <div className="absolute bottom-[15px] left-[10px] z-10" style={{ backgroundColor: 'transparent' }}>
            <button
              onClick={() => setPromptMode(promptMode === "research" ? "default" : "research")}
              className={`px-4 py-1 rounded-full text-sm font-medium border transition-all ${
                promptMode === "research"
                  ? "bg-[#D38A74] text-white border-[#D38A74]"
                  : "bg-transparent text-gray-700 border-gray-400 hover:bg-gray-100"
              }`}
              style={{ boxShadow: 'none' }}
            >
              {promptMode === "research" ? "Research Mode" : "Default Mode"}
            </button>
          </div>

          <div className="flex" style={{ backgroundColor: 'transparent' }}>
            <div className="flex-1" style={{ backgroundColor: 'transparent' }}>
              <div className="relative" style={{ backgroundColor: 'transparent' }}>
                <textarea
                  ref={textAreaRef} // 添加 ref
                  rows={1}
                  onKeyDown={handleKeyDown}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  className="w-full p-2 pb-16 pr-14 text-lg leading-relaxed tracking-wide rounded-xl bg-[#FCFBF9] text-black resize-none border-2 border-gray-300
                    focus:outline-none focus:ring-0 focus:border-gray-300 font-serif overflow-hidden"
                  placeholder="請輸入問題"
                  style={{ minHeight: "100px", maxHeight: "200px" }}
                ></textarea>
                <div className="absolute h-14 bottom-0 left-0 right-0" style={{ backgroundColor: 'transparent' }}></div>
              </div>
            </div>
          </div>

          {/* ✅ 回饋輸入區塊，獨立放到右側固定區塊 */}
          <div className="fixed bottom-32 right-12 w-[340px] p-4 rounded-xl bg-white shadow-md border border-gray-300 text-sm space-y-3">
            <div>
              <label className="block mb-1">📝 輸入您的問題(遇到錯誤資訊或者不滿意)：</label>
              <input
                type="text"
                value={expectedQuestion}
                onChange={(e) => setExpectedQuestion(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>
            <div>
              <label className="block mb-1">✅ 預期正確回答：</label>
              <textarea
                rows={3}
                value={expectedAnswer}
                onChange={(e) => setExpectedAnswer(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              ></textarea>
            </div>
            <button
              onClick={submitFeedback}
              className="w-full px-4 py-2 bg-[#D38A74] text-white rounded-lg hover:bg-[#c15c3a]"
            >送出修正建議</button>
          </div>

          <button
            onClick={sendMessage}
            className="absolute bottom-[15px] right-[10px] px-2 py-2 rounded-lg z-10"
            style={{ 
              backgroundColor: '#D38A74', 
              boxShadow: 'none',
              outline: 'none',
              border: 'none'
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" className="w-6 h-6">
              <path d="M12 21c-.55 0-1-.45-1-1V7.83L6.41 12.41a.996.996 0 1 1-1.41-1.41l6.3-6.29a1 1 0 0 1 1.41 0l6.29 6.29a.996.996 0 1 1-1.41 1.41L13 7.83V20c0 .55-.45 1-1 1z" />
            </svg>
          </button>
        </div>
      </footer>
    </div>
  );
}