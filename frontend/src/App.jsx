import { useState, useEffect, useRef, useCallback }
from "react";
import axios from "axios";

// --- SVG 圖標組件 (保持不變) ---
const MenuIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
  </svg>
);
const ChevronDownIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
  </svg>
);
const ChevronUpIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 15.75 7.5-7.5 7.5 7.5" />
  </svg>
);
const PlusIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
  </svg>
);
const ChevronLeftIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
    </svg>
);
const FeedbackIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5 mr-2">
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-6.75 3h9M3.375 7.375c0-1.036.84-1.875 1.875-1.875h13.5c1.036 0 1.875.84 1.875 1.875v9.75c0 1.036-.84 1.875-1.875-1.875H3.375c-1.036 0-1.875-.84-1.875-1.875V7.375Z" />
  </svg>
);
const DownloadIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-2">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
    </svg>
);
const MailIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-2 shrink-0">
        <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 0 1-2.25 2.25h-15a2.25 2.25 0 0 1-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0 0 19.5 4.5h-15a2.25 2.25 0 0 0-2.25 2.25m19.5 0v.243a2.25 2.25 0 0 1-1.07 1.916l-7.5 4.615a2.25 2.25 0 0 1-2.36 0L3.32 8.91a2.25 2.25 0 0 1-1.07-1.916V6.75" />
    </svg>
);
const EllipsisHorizontalIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4">
        <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM12.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0ZM18.75 12a.75.75 0 1 1-1.5 0 .75.75 0 0 1 1.5 0Z" />
    </svg>
);
const PencilIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-2">
        <path strokeLinecap="round" strokeLinejoin="round" d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10" />
    </svg>
);
const TrashIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-2">
        <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
    </svg>
);
// --- SVG User Icon ---
const UserIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-2 shrink-0">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 1 1-7.5 0 3.75 3.75 0 0 1 7.5 0ZM4.501 20.118a7.5 7.5 0 0 1 14.998 0A1.875 1.875 0 0 1 18 22.5H6c-.98 0-1.844-.796-1.875-1.882Z" />
    </svg>
);

// --- 新增：SVG Logout Icon ---
const LogoutIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4 mr-2 shrink-0">
        <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15m0-3H6.75M15.75 12l-3-3m3 3-3 3" />
    </svg>
);
// --- SVG 圖標組件結束 ---

// --- Modal 組件定義 (保持不變) ---
// Rename Modal
function RenameChatModal({ chatId, currentTitle, onSave, onCancel }) {
    const [newTitle, setNewTitle] = useState(currentTitle || "");
    const inputRef = useRef(null);
    const modalContentRef = useRef(null);

    useEffect(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
    }, []);

    const handleContentClick = (e) => e.stopPropagation();

    const handleSave = () => {
        if (newTitle.trim()) {
          onSave(chatId, newTitle);
        } else {
          alert("標題不能為空。");
        }
    };

    const handleKeyDown = (e) => {
      if (e.key === 'Enter') handleSave();
      else if (e.key === 'Escape') onCancel();
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 transition-opacity duration-150 ease-in-out" onClick={onCancel} >
            <div ref={modalContentRef} onClick={handleContentClick} className="bg-white rounded-xl p-6 w-full max-w-md shadow-xl transform transition-all duration-150 ease-in-out scale-100" >
                <h3 className="text-lg font-semibold leading-6 text-gray-900 mb-4">Rename chat</h3>
                <input ref={inputRef} type="text" value={newTitle} onChange={(e) => setNewTitle(e.target.value)} onKeyDown={handleKeyDown} className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-1 focus:ring-[#D38A74] focus:border-[#D38A74] text-sm" />
                <div className="mt-5 sm:mt-6 flex justify-end space-x-3">
                    <button type="button" onClick={onCancel} className="inline-flex justify-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400" > Cancel </button>
                    <button type="button" onClick={handleSave} className="inline-flex justify-center rounded-lg border border-transparent bg-black px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black" > Save </button>
                </div>
            </div>
        </div>
    );
  }

// Delete Modal
function DeleteChatModal({ chatId, onConfirm, onCancel }) {
    const confirmButtonRef = useRef(null);
    const modalContentRef = useRef(null);

    useEffect(() => { confirmButtonRef.current?.focus(); }, []);
    const handleContentClick = (e) => e.stopPropagation();
    const handleDelete = () => onConfirm(chatId);

    return (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 transition-opacity duration-150 ease-in-out" onClick={onCancel} >
            <div ref={modalContentRef} onClick={handleContentClick} className="bg-white rounded-xl p-6 w-full max-w-md shadow-xl transform transition-all duration-150 ease-in-out scale-100" >
                <h3 className="text-lg font-semibold leading-6 text-gray-900">Delete chat?</h3>
                <div className="mt-2"> <p className="text-sm text-gray-600">Are you sure you want to delete this chat?</p> </div>
                <div className="mt-5 sm:mt-6 flex justify-end space-x-3">
                    <button type="button" onClick={onCancel} className="inline-flex justify-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400" > Cancel </button>
                    <button ref={confirmButtonRef} type="button" onClick={handleDelete} className="inline-flex justify-center rounded-lg border border-transparent bg-[#c15c3a] px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-[#a94a2a] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#c15c3a]" > Delete </button>
                </div>
            </div>
        </div>
    );
}
// --- Modal 組件定義結束 ---

// --- 新增：Username Input Modal ---
function UsernameModal({ onSetUsername }) {
    const [tempUsername, setTempUsername] = useState("");
    const inputRef = useRef(null);

    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (tempUsername.trim()) {
            onSetUsername(tempUsername.trim());
        } else {
            alert("用戶名稱不能為空。");
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-[100]"> {/* z-index 提高以覆蓋所有內容 */}
            <form onSubmit={handleSubmit} className="bg-white rounded-xl p-8 shadow-2xl w-full max-w-sm">
                <h2 className="text-xl font-semibold text-gray-800 mb-6 text-center">請輸入您的用戶名稱</h2>
                <input
                    ref={inputRef}
                    type="text"
                    value={tempUsername}
                    onChange={(e) => setTempUsername(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-[#D38A74] focus:border-[#D38A74] text-base mb-6"
                    placeholder="例如：KMUUser"
                />
                <button
                    type="submit"
                    className="w-full inline-flex justify-center rounded-lg border border-transparent bg-black px-6 py-3 text-base font-medium text-white shadow-sm hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black"
                >
                    開始使用
                </button>
            </form>
        </div>
    );
}
// --- Username Input Modal 結束 ---


const API_BASE_URL = "http://163.15.172.93:8000";

export default function App() {
  const CONTROL_HEIGHT = 56;
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("gemma3:12b");
  const [promptMode, setPromptMode] = useState("default");
  const [expectedQuestion, setExpectedQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const bottomRef = useRef(null);
  const textAreaRef = useRef(null);

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isChatsOpen, setIsChatsOpen] = useState(true);
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);
  const [menuOpenForChat, setMenuOpenForChat] = useState(null);
  const menuRef = useRef(null);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [modalState, setModalState] = useState(null);

  const [username, setUsername] = useState(null);
  const [showUsernameModal, setShowUsernameModal] = useState(false);

  const apiClient = axios.create({
    baseURL: API_BASE_URL,
  });

  apiClient.interceptors.request.use(
    (config) => {
      if (username) {
        config.headers['X-Username'] = username;
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  useEffect(() => {
    const storedUsername = localStorage.getItem("chatAppUsername");
    if (storedUsername) {
      setUsername(storedUsername);
    } else {
      setShowUsernameModal(true);
    }
  }, []);

  useEffect(() => {
    if (!username) return;

    const fetchChatHistory = async () => {
      setIsLoadingHistory(true);
      setChatHistory([]);
      setCurrentChatId(null);
      setMessages([]);
      try {
        const response = await apiClient.get(`/api/chats`);
        setChatHistory(response.data || []);
      } catch (error)
      {
        console.error("Failed to fetch chat history:", error);
        setChatHistory([]);
      } finally {
        setIsLoadingHistory(false);
      }
    };
    fetchChatHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username]);
  // --- 修改結束 --- (此處為用戶程式碼原有的註釋位置)

  // --- 新增：在用戶登入且無選中聊天時，自動創建新聊天 ---
  useEffect(() => {
    if (username && !isLoadingHistory && currentChatId === null && chatHistory.length === 0) {
      handleNewChat();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username, isLoadingHistory, currentChatId, chatHistory.length]); // chatHistory.length 確保在歷史記錄實際為空時觸發
  // --- 新增結束 ---

  const handleSetUsername = (newUsername) => {
    const sanitizedUsername = newUsername.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase() || `user_${Date.now()}`;
    setUsername(sanitizedUsername);
    localStorage.setItem("chatAppUsername", sanitizedUsername);
    setShowUsernameModal(false);
    // --- 新增：用戶設定後，如果側邊欄是關閉的，則打開它 ---
    if (!isSidebarOpen) {
        setIsSidebarOpen(true);
    }
    // --- 新增結束 ---
  };

  // --- 新增：處理切換使用者 (登出) ---
  const handleLogout = () => {
    localStorage.removeItem("chatAppUsername");
    setUsername(null);
    setShowUsernameModal(true);
    // 重置應用程式狀態
    setMessages([]);
    setChatHistory([]);
    setCurrentChatId(null);
    setInput("");
    setIsLoadingHistory(false); // 確保加載狀態重置
    setIsSidebarOpen(false); // 登出時關閉側邊欄
    // 根據需要重置其他狀態，例如：
    // setMenuOpenForChat(null);
    // setModalState(null);
    // setIsChatsOpen(true);
    // setIsFeedbackOpen(false);
  };
  // --- 新增結束 ---

  const adjustTextAreaHeight = () => {
    const textArea = textAreaRef.current;
    if (textArea) {
      textArea.style.height = "auto";
      const minHeight = 100;
      const maxHeight = 400;
      const scrollHeight = textArea.scrollHeight;
      let newHeight = scrollHeight;
      if (scrollHeight <= minHeight) newHeight = minHeight;
      else if (scrollHeight > maxHeight) newHeight = maxHeight;
      textArea.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden';
      textArea.style.height = `${newHeight}px`;
    }
  };

  const handleScroll = (e) => {
    const ta = e.target;
    const maxScroll = ta.scrollHeight - ta.clientHeight - CONTROL_HEIGHT;
    if (ta.scrollTop > maxScroll) ta.scrollTop = maxScroll;
  };

  const sendMessage = async () => {
    if (!username) {
        alert("請先設置您的用戶名稱。");
        setShowUsernameModal(true);
        return;
    }
    if (!input.trim()) return;

    let chatIdToUse = currentChatId;
    if (!chatIdToUse) {
        alert("請先選擇或建立一個新的對話。");
        return;
    }

    setMessages((prev) => [...prev, { role: "user", content: input }]);
    const currentInput = input;
    setInput("");
    setLoading(true);
    try {
      const res = await apiClient.post(`/chat`, {
        session_id: chatIdToUse,
        question: currentInput,
        model: selectedModel,
        prompt_mode: promptMode,
      });
      setMessages((prev) => [...prev, { role: "assistant", content: res.data.answer }]);
    } catch (error) {
      console.error("Send message error:", error)
      setMessages((prev) => [...prev, { role: "assistant", content: "❌ 回答失敗，請稍後再試。" }]);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (!username) {
        alert("請先設置您的用戶名稱。");
        setShowUsernameModal(true);
        return;
    }
    if (!expectedAnswer.trim() && !expectedQuestion.trim()) {
        alert("請至少輸入問題或預期回答。"); return;
    }
    let lastUserMessage = null, lastAssistantMessage = null;
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'assistant') {
            lastAssistantMessage = messages[i];
            if (i > 0 && messages[i-1].role === 'user') lastUserMessage = messages[i-1];
            break;
        }
    }
    const questionForFeedback = lastUserMessage ? lastUserMessage.content : "用戶未提問或來自新對話";
    const originalAnswerForFeedback = lastAssistantMessage ? lastAssistantMessage.content : "無原始回答或來自新對話";
    const chatIdToUse = currentChatId;
    if (!chatIdToUse) {
        alert("無法確定當前對話以提交回饋。");
        return;
    }

    try {
      await apiClient.post(`/feedback`, {
        session_id: chatIdToUse,
        question: questionForFeedback,
        model: selectedModel,
        original_answer: originalAnswerForFeedback,
        user_expected_question: expectedQuestion,
        user_expected_answer: expectedAnswer,
      });
      alert("✅ 已送出回饋，感謝您的建議！");
      setExpectedQuestion(""); setExpectedAnswer(""); setIsFeedbackOpen(false);
    } catch (err) {
      console.error("Submit feedback error:", err);
      alert("❌ 無法提交回饋，請稍後再試。");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);
  useEffect(() => { adjustTextAreaHeight(); }, [input]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpenForChat(null);
      }
    };
    if (menuOpenForChat) document.addEventListener("mousedown", handleClickOutside);
    else document.removeEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [menuOpenForChat]);

  const handleNewChat = async () => {
    if (!username) {
        // alert("請先設置您的用戶名稱。"); // 在自動創建時，若無username，則由上層useEffect處理
        // setShowUsernameModal(true); // 不在這裡顯示，讓UsernameModal自己處理
        return;
    }
    setMessages([]);
    const newChatId = `chat_${Date.now()}`;
    const newChatTitle = `新對話 ${chatHistory.filter(chat => chat.title.startsWith("新對話")).length + 1}`;


    const newChatOptimistic = { id: newChatId, title: newChatTitle, updated_at: new Date().toISOString() };
    setChatHistory(prevHistory => [newChatOptimistic, ...prevHistory].sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at)));
    setCurrentChatId(newChatId);
    setInput("");

    try {
      const response = await apiClient.post(`/api/chats`, { id: newChatId, title: newChatTitle });
      const actualChat = response.data;
      setChatHistory(prevHistory =>
        prevHistory.map(chat => (chat.id === newChatId ? actualChat : chat)).sort((a,b) => new Date(b.updated_at) - new Date(a.updated_at))
      );
    } catch (error) {
      console.error("Failed to create new chat on backend:", error);
      // alert("建立新對話失敗。"); // 自動創建時，減少alert打擾
      setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== newChatId));
      if (currentChatId === newChatId) {
        const nextChat = chatHistory.length > 0 ? chatHistory[0] : null;
        if (nextChat) {
            handleSelectChat(nextChat.id, false);
        } else {
            setCurrentChatId(null);
            setMessages([]);
        }
      }
    }
  };

  const handleSelectChat = async (chatId, userInitiated = true) => {
    if (!username) return;
    if (currentChatId === chatId && messages.length > 0 && userInitiated) {
      setMenuOpenForChat(null);
      return;
    }
    if (currentChatId === chatId && !userInitiated && messages.length > 0) {
        setMenuOpenForChat(null);
        return;
    }

    setMessages([]);
    setCurrentChatId(chatId);
    setInput("");
    setMenuOpenForChat(null);
    setLoading(true);

    try {
      const response = await apiClient.get(`/api/chats/${chatId}/messages`);
      setMessages(response.data || []);
    } catch (error) {
      console.error(`Failed to fetch messages for chat ${chatId}:`, error);
      setMessages([{ role: "assistant", content: "❌ 載入對話記錄失敗。" }]);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleChatMenu = (chatId, event) => {
    event.stopPropagation();
    setMenuOpenForChat(prevOpenChatId => (prevOpenChatId === chatId ? null : chatId));
  };

  const handleRenameChat = (chatId) => {
    setMenuOpenForChat(null);
    const chatToRename = chatHistory.find(chat => chat.id === chatId);
    if (chatToRename) {
      setModalState({ type: 'rename', chatId: chatId, currentTitle: chatToRename.title });
    }
  };

  const handleDeleteChat = (chatId) => {
    setMenuOpenForChat(null);
    setModalState({ type: 'delete', chatId: chatId });
  };

  const closeModal = () => {
    setModalState(null);
  };

  const confirmRename = async (chatId, newTitle) => {
    if (!username) return;
    if (newTitle && newTitle.trim() !== "") {
      const originalChat = chatHistory.find(chat => chat.id === chatId);
      // const originalTitle = originalChat?.title; // Not strictly needed for current logic but good for reference
      const updated_at = new Date().toISOString();

      setChatHistory(prevHistory =>
        prevHistory.map(chat =>
          chat.id === chatId ? { ...chat, title: newTitle.trim(), updated_at } : chat
        ).sort((a,b) => new Date(b.updated_at) - new Date(a.updated_at))
      );
      closeModal();

      try {
        await apiClient.put(`/api/chats/${chatId}`, { title: newTitle.trim() });
      } catch (error) {
        console.error(`Failed to rename chat ${chatId} on backend:`, error);
        alert("重命名失敗。");
        if (originalChat) {
          setChatHistory(prevHistory =>
            prevHistory.map(chat =>
              chat.id === chatId ? originalChat : chat
            ).sort((a,b) => new Date(b.updated_at) - new Date(a.updated_at))
          );
        }
      }
    } else {
      alert("標題不能為空。");
    }
  };

  const confirmDelete = async (chatId) => {
    if (!username) return;
    const chatToDelete = chatHistory.find(chat => chat.id === chatId);
    if (!chatToDelete) return;

    const oldChatHistory = [...chatHistory];
    setChatHistory(prevHistory => prevHistory.filter(chat => chat.id !== chatId));

    if (currentChatId === chatId) {
      setMessages([]);
      const remainingChats = oldChatHistory.filter(c => c.id !== chatId);
      if (remainingChats.length > 0) {
        const sortedRemaining = [...remainingChats].sort((a,b) => new Date(b.updated_at) - new Date(a.updated_at));
        handleSelectChat(sortedRemaining[0].id, false);
      } else {
        setCurrentChatId(null);
        // The useEffect for auto new chat will handle creating a new one if needed
      }
      setInput("");
    }
    closeModal();

    try {
      await apiClient.delete(`/api/chats/${chatId}`);
    } catch (error) {
      console.error(`Failed to delete chat ${chatId} from backend:`, error);
      alert("刪除失敗。");
      setChatHistory(oldChatHistory);
      if (currentChatId === null && oldChatHistory.some(c => c.id === chatId)) {
         handleSelectChat(chatId, false);
      }
    }
  };

  return (
    <>
      {showUsernameModal && <UsernameModal onSetUsername={handleSetUsername} />}

      <div className={`flex h-screen bg-[#F9F8F3] text-black overflow-hidden ${showUsernameModal ? 'filter blur-sm pointer-events-none' : ''}`}>
        <div
          className={`
            transition-all duration-300 ease-in-out
            bg-[#F0EBE3] border-r border-gray-300 flex flex-col
            ${isSidebarOpen && username ? "w-72" : "w-0"}
            overflow-hidden
          `}
        >
          {isSidebarOpen && username && (
            <div className="p-4 flex flex-col h-full">
              <div>
                 <div className="flex items-center justify-between mb-4"> <button onClick={() => setIsSidebarOpen(false)} className="text-gray-500 hover:text-gray-700 p-1 rounded-md hover:bg-gray-200 -ml-1" aria-label="收合側邊欄"> <ChevronLeftIcon /> </button> <h2 className="text-xl font-semibold text-gray-800">KMU</h2> <div className="w-6 h-6"></div> </div>
                 <button onClick={handleNewChat} className="w-full flex items-center justify-center gap-2 px-4 py-2 mb-4 text-sm font-medium text-black bg-white border border-gray-300 rounded-lg hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-[#D38A74]"> <PlusIcon /> New chat </button>
              </div>

              <div className="flex-grow mb-4 overflow-y-auto" style={{ minHeight: '100px' }}>
                <button onClick={() => setIsChatsOpen(!isChatsOpen)} className="flex items-center justify-between w-full px-2 py-1 text-sm font-semibold text-gray-700 rounded hover:bg-gray-200 mb-1"> Chats {isChatsOpen ? <ChevronUpIcon /> : <ChevronDownIcon />} </button>
                {isChatsOpen && (
                  <ul className="space-y-1">
                    {isLoadingHistory ? (
                        <li className="px-2 py-1.5 text-xs text-gray-400">載入對話記錄中...</li>
                    ) : chatHistory.length === 0 && currentChatId === null ? ( // 避免在剛創建新聊天但後端未確認時顯示"尚無對話"
                        <li className="px-2 py-1.5 text-xs text-gray-400">尚無對話紀錄</li>
                    ) : (
                        chatHistory.map((chat) => (
                        <li key={chat.id} className={`relative flex items-center justify-between rounded ${currentChatId === chat.id ? 'bg-gray-300/70' : 'hover:bg-gray-200/60'}`}>
                            <button onClick={() => handleSelectChat(chat.id)} className="flex-grow text-left px-2 py-1.5 text-xs text-gray-700 truncate" title={chat.title}> {chat.title} </button>
                            <button id={`menu-button-${chat.id}`} onClick={(e) => handleToggleChatMenu(chat.id, e)} className="p-1.5 text-gray-500 hover:text-gray-800 rounded mr-1 focus:outline-none focus:ring-1 focus:ring-gray-400" aria-haspopup="true" aria-expanded={menuOpenForChat === chat.id} aria-label={`聊天選項 ${chat.title}`}> <EllipsisHorizontalIcon /> </button>
                            {menuOpenForChat === chat.id && (
                            <div ref={menuRef} className="absolute right-1 top-full mt-1 w-32 bg-white rounded-md shadow-lg border border-gray-200 z-20" role="menu" aria-orientation="vertical" aria-labelledby={`menu-button-${chat.id}`}>
                                <div className="py-1" role="none">
                                <button onClick={() => handleRenameChat(chat.id)} className="flex items-center w-full px-3 py-1.5 text-left text-xs text-gray-700 hover:bg-gray-100 hover:text-gray-900" role="menuitem"> <PencilIcon /> Rename </button>
                                <button onClick={() => handleDeleteChat(chat.id)} className="flex items-center w-full px-3 py-1.5 text-left text-xs text-red-600 hover:bg-red-50 hover:text-red-700" role="menuitem"> <TrashIcon /> Delete </button>
                                </div>
                            </div>
                            )}
                        </li>
                        ))
                    )}
                  </ul>
                )}
              </div>

              <div className="mt-auto">
                <div className="border-t border-gray-300 pt-3">
                    <button onClick={() => setIsFeedbackOpen(!isFeedbackOpen)} className="flex items-center justify-between w-full px-2 py-2 text-sm font-semibold text-gray-700 rounded hover:bg-gray-200"> <div className="flex items-center"> <FeedbackIcon /> <span>修正建議</span> </div> {isFeedbackOpen ? <ChevronUpIcon /> : <ChevronDownIcon />} </button>
                    {isFeedbackOpen && ( <div className="mt-2 p-3 bg-white/50 rounded-md shadow-sm border border-gray-200 text-xs space-y-2"> <div> <label className="block mb-0.5 text-gray-700">您的問題 (選填)：</label> <input type="text" value={expectedQuestion} onChange={(e) => setExpectedQuestion(e.target.value)} className="w-full px-2 py-1.5 border border-gray-300 rounded-md text-xs" placeholder="例如：關於PM2.5的影響..."/> </div> <div> <label className="block mb-0.5 text-gray-700">預期正確回答 (必填)：</label> <textarea rows={3} value={expectedAnswer} onChange={(e) => setExpectedAnswer(e.target.value)} className="w-full px-2 py-1.5 border border-gray-300 rounded-md text-xs resize-y min-h-[60px]" placeholder="請提供您認為更精確或完整的回答..."></textarea> </div> <button onClick={submitFeedback} className="w-full px-3 py-1.5 bg-[#D38A74] text-white rounded-md hover:bg-[#c15c3a] text-xs">送出建議</button> </div> )}
                </div>
                <div className="border-t border-gray-300 mt-3 pt-3"> <a href="/documents/空氣污染RAG系統使用說明書.pdf" download className="flex items-center w-full px-2 py-2 text-xs text-gray-700 rounded hover:bg-gray-200 hover:text-[#D38A74] focus:outline-none focus:ring-1 focus:ring-[#D38A74]"> <DownloadIcon /> <span>下載使用說明書</span> </a> </div>
                <div className="border-t border-gray-300 mt-3 pt-3 text-xs text-gray-600"> <div className="flex items-start px-2 py-1"> <MailIcon /> <div className="flex flex-col"> <span>系統建議與錯誤回報：</span> <a href="mailto:pijh102511@gmail.com" className="underline hover:text-[#D38A74] break-all"> pijh102511@gmail.com </a> </div> </div> </div>
                {username && (
                  <>
                    <div className="border-t border-gray-300 mt-3 pt-3 text-xs text-gray-600">
                        <div className="flex items-center px-2 py-1">
                            <UserIcon />
                            <span className="truncate" title={username}>用戶: {username}</span>
                        </div>
                    </div>
                    {/* --- 新增：切換使用者按鈕 --- */}
                    <div className="border-t border-gray-300 mt-3 pt-3">
                        <button
                            onClick={handleLogout}
                            aria-label="切換使用者"
                            className="flex items-center w-full px-2 py-2 text-xs text-gray-700 rounded hover:bg-gray-200 hover:text-red-600 focus:outline-none focus:ring-1 focus:ring-red-400"
                        >
                            <LogoutIcon />
                            <span>切換使用者</span>
                        </button>
                    </div>
                    {/* --- 新增結束 --- */}
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="flex-1 flex flex-col h-screen overflow-hidden">
          <header className="p-4 text-xl font-bold bg-[#F9F8F3] shadow flex items-center gap-3 relative">
             {!isSidebarOpen && username && ( <button onClick={() => setIsSidebarOpen(true)} className="p-1 mr-2 text-gray-600 hover:text-black hover:bg-gray-200 rounded-md" aria-label="展開側邊欄"> <MenuIcon /> </button> )}
             <img src="/images/KMU_logo.png" alt="KMU Logo" className="w-10 h-10 object-contain" />
             空氣汙染檢索增強(目前系統測試中較不穩定)請仔細閱讀輸出結果
          </header>
          <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4 flex flex-col font-serif">
             {loading && messages.length === 0 && currentChatId && (
                <div className="flex justify-center items-center h-full">
                    <p className="text-gray-500">正在載入對話內容...</p>
                </div>
             )}
             {messages.map((msg, i) => ( <div key={i} className="flex justify-center"> <div className="flex items-start gap-2 max-w-[800px] w-full"> {msg.role === "user" && ( <img src="/images/KMU_logo.png" alt="KMU" className="w-6 h-6 mt-1 rounded-full object-contain shrink-0" /> )} <div className={ "px-4 py-2 rounded-lg whitespace-pre-wrap prose text-black max-w-[90%] "+ (msg.role === "user" ? "bg-[#F2EFE7]" : "bg-[#F9F8F3]") } dangerouslySetInnerHTML={{ __html: msg.content }} ></div> </div> </div> ))}
             {loading && input === "" && messages.length > 0 && (
                <div className="flex justify-center"> <div className="flex items-start gap-2 max-w-[800px] w-full justify-start"> <div className="text-gray-400 flex items-center space-x-2 font-serif max-w-[90%]"> <img src="/images/claude-color.png" alt="loading icon" className="w-5 h-5 animate-spin" /> <span>AI 正在思考中...</span> </div> </div> </div>
             )}
             <div ref={bottomRef} />
          </main>
          <footer className="px-4 pt-0 pb-[25px] bg-[#F9F8F3] flex flex-col items-center font-serif" style={{ backgroundColor: 'transparent' }}>
            <div className="relative w-[730px] transform -translate-y-0 -translate-x-8" style={{ backgroundColor: 'transparent' }}>
              <div className="absolute bottom-[20px] right-[60px] z-10" style={{ backgroundColor: 'transparent', boxShadow: 'none' }}>
                <div className="relative w-[320px]" style={{ backgroundColor: 'transparent' }}>
                  <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="appearance-none bg-transparent text-black text-sm pr-8 py-2 rounded-md w-full focus:outline-none focus:ring-0 focus:border-0 hover:ring-0 hover:bg-transparent transition-all" style={{ textAlignLast: "right", backgroundColor: 'transparent', boxShadow: 'none', border: 'none' }}>
                      <option value="gemma3:12b-it-q4_K_M">gemma3:12b-it-q4_K_M</option>
                      <option value="gemma3:12b">Gemma 3 12B</option>
                      <option value="qwen2.5:14b">Qwen 2.5 14B</option>
                      <option value="qwen2.5:14b-instruct-q5_K_M">qwen2.5:14b-instruct-q5_K_M(測試)</option>
                  </select>
                  <div className="pointer-events-none absolute top-1/2 right-2 transform -translate-y-1/2" style={{ backgroundColor: 'transparent', boxShadow: 'none' }}> <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" style={{ backgroundColor: 'transparent', boxShadow: 'none' }}> <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /> </svg> </div>
                </div>
              </div>
              <div className="absolute bottom-[15px] left-[10px] z-10" style={{ backgroundColor: 'transparent' }}>
                <button onClick={() => setPromptMode(promptMode === "research" ? "default" : "research")} className={`px-4 py-1 rounded-full text-sm font-medium border transition-all ${ promptMode === "research" ? "bg-[#D38A74] text-white border-[#D38A74]" : "bg-transparent text-gray-700 border-gray-400 hover:bg-gray-100" }`} style={{ boxShadow: 'none' }}> {promptMode === "research" ? "Research Mode" : "Default Mode"} </button>
              </div>
              <div className="flex" style={{ backgroundColor: 'transparent' }}>
                <div className="flex-1" style={{ backgroundColor: 'transparent' }}>
                  <div className="relative border-[1.5px] border-gray-400 rounded-xl bg-[#FCFBF9] overflow-hidden">
                    <textarea
                      ref={textAreaRef}
                      rows={1}
                      onKeyDown={handleKeyDown}
                      onScroll={handleScroll}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      className="w-full p-2 pb-16 pr-10 text-lg leading-relaxed tracking-wide resize-none bg-transparent border-none focus:outline-none focus:ring-0 focus:border-gray-300 font-serif"
                      placeholder={!username ? "請先設置用戶名稱" : (currentChatId ? "How can I help you today?" : "請先建立或選擇一個對話")}
                      disabled={!username || (!currentChatId && chatHistory.length === 0 && !input && !loading)} // Adjusted disabled logic slightly
                      style={{ minHeight: "100px", maxHeight: "400px", overflowY: "hidden" }}
                    />
                    <div className="absolute h-14 bottom-[-8px] left-0 right-0 bg-[#FCFBF9] pointer-events-none rounded-b-xl" />
                  </div>
                </div>
              </div>
              <button
                onClick={sendMessage}
                disabled={!username || (!currentChatId && !input.trim()) || loading}
                className="absolute bottom-[20px] right-[20px] px-3 py-3 bg-[#D38A74] text-white rounded-lg hover:bg-[#c15c3a] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#D38A74] disabled:opacity-50"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" className="w-3 h-3">
                  <path d="M12 21c-.55 0-1-.45-1-1V7.83L6.41 12.41a.996.996 0 1 1-1.41-1.41l6.3-6.29a1 1 0 0 1 1.41 0l6.29 6.29a.996.996 0 1 1-1.41 1.41L13 7.83V20c0 .55-.45 1-1 1z" />
                </svg>
              </button>
            </div>
          </footer>
        </div>
      </div>

      {modalState?.type === 'rename' && (
        <RenameChatModal
          chatId={modalState.chatId}
          currentTitle={modalState.currentTitle}
          onSave={confirmRename}
          onCancel={closeModal}
        />
      )}
      {modalState?.type === 'delete' && (
        <DeleteChatModal
          chatId={modalState.chatId}
          onConfirm={confirmDelete}
          onCancel={closeModal}
        />
      )}
    </>
  );
}
