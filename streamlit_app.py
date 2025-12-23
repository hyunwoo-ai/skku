import streamlit as st
import time
import random
import ast
from datetime import datetime

import instructions

# =======================================================
# 1. Backend Libraries & Classes
# =======================================================
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from FlagEmbedding import BGEM3FlagModel
    import chromadb
except ImportError:
    # 로컬 환경에서 라이브러리가 없을 경우를 대비한 안내
    st.error("필수 라이브러리(vllm, transformers, chromadb 등)가 설치되지 않았습니다.")
    st.stop()

class QueryRewriter:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

        # 창의성보다는 정확성을 위해 temperature를 낮춤
        self.sampling_params = SamplingParams(temperature=0.2, max_tokens=32768, repetition_penalty=1.05,)
        
        # 검색 쿼리 변환을 위한 시스템 프롬프트
        self.system_prompt = (
            "당신은 검색 쿼리 최적화 AI입니다. "
            "사용자의 질문을 벡터 데이터베이스에서 검색하기 좋은 '1~3개의 구체적인 질문'으로 분해하거나 재작성하세요. "
            "불필요한 사족 없이 오직 예시에 나와있는 json 형식으로만 출력하세요."
            "재작성된 쿼리는 리스트 형식으로 작성하세요."
            ""
            "예시)"
            "{"
            "  \"original_query\": 이번 대회의 우승자와 저의 차이점이 뭡니까?"
            "  \"rewritten_queries\": [\"이 대회의 우승자는 누구입니까?\", \"이 대회의 우승자의 특징은 무엇입니까?\"]"
            "}"
        )

    def rewrite(self, user_query):
        """
        사용자 질문 -> 검색 쿼리 리스트 변환
        """
        # 1. 프롬프트 구성
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"질문: {user_query}\n\n검색 쿼리:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. 추론
        outputs = self.llm.generate([prompt], self.sampling_params)
        raw_output = outputs[0].outputs[0].text.strip()

        # <think> 태그를 기준으로 내부 사고 과정과 최종 답변을 분리
        thinking_content = ""
        final_response = raw_output

        if "</think>" in raw_output:
            parts = raw_output.split("</think>")
            thinking_content = parts[0].replace("<think>", "").strip()
            if len(parts) > 1:
                final_response = parts[1].strip()
            else:
                final_response = "" # 사고 과정만 출력된 경우 예외 처리

        # 3. 결과 파싱
        try:
            # JSON 파싱 시도 (LLM이 가끔 형식을 어길 수 있으므로 예외처리)
            queries = ast.literal_eval(final_response)['rewritten_queries']
        except:
            # 파싱 실패 시 원본 쿼리 사용
            queries = [user_query]
            
        print(f"🔄 [Rewriter] 원본: '{user_query}' -> 변환: {queries}")
        return list(queries)


class QwenVLLMChatbotWithRAG:
    def __init__(self, 
                 model_name="Qwen/Qwen3-30B-A3B-FP8", 
                 llm=None, 
                 tokenizer=None, 
                 query_rewriter=None,
                 embedding_model=None,
                 collection=None,
                 system_instructions=None):
        """
        RAG(Retrieval-Augmented Generation) 기능을 탑재한 Qwen 챗봇 초기화
        """
        print(f"Initializing Chatbot with model: {model_name}...")
        
        # -------------------------------------------------------
        # [1. 생성 모델(Generator) 설정] - LLM & Tokenizer
        # -------------------------------------------------------
        if llm:
            print(">> [Generator] Existing LLM instance detected. Using provided engine.")
            self.llm = llm
        else:
            print(">> [Generator] Loading new vLLM engine...")
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.90,
                dtype="auto",
                trust_remote_code=True
            )

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # -------------------------------------------------------
        # [2. RAG 컴포넌트 설정] - Rewriter, Embedder, Vector DB
        # -------------------------------------------------------
        
        if query_rewriter:
            print(">> [RAG] Existing Query Rewriter detected.")
            self.query_rewriter = query_rewriter
        else:
            print(">> [RAG] Loading new Query Rewriter...")
            self.query_rewriter = QueryRewriter(self.llm, self.tokenizer)

        if embedding_model:
            print(">> [RAG] Existing Embedding Model detected.")
            self.embedding_model = embedding_model
        else:
            print(">> [RAG] Loading new BGE-M3 Embedding Model...")
            self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

        if collection:
            print(">> [RAG] Existing ChromaDB Collection detected.")
            self.collection = collection
        else:
            print(">> [RAG] Connecting to ChromaDB...")
            # Streamlit 캐시 문제 방지를 위해 경로 확인 필요
            client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = client.get_or_create_collection("persona_memory")
        
        self.system_instructions = system_instructions
        self.clear_history() 

    def search_relatives(self, query, top_k=3):
        print(f"\n   [Retrieval] 검색 수행: '{query}'")
        start_time = time.time()
        
        try:
            query_embeddings = self.embedding_model.encode(query, batch_size=1)['dense_vecs']
            
            # 3. 결과에서 우리가 필요한 첫 번째(진짜 질문) 결과만 가져옴
            query_embedding = query_embeddings[0]
            
        except Exception as e:
            print(f"   [Error] 임베딩 우회 처리 중 치명적 오류: {e}")
            # 최악의 경우를 대비한 2차 방어선 (비효율적이지만 작동은 하게 함)
            # 여기서는 어쩔 수 없이 에러가 나더라도 진행되도록 예외 처리
            return {"ids": [[]], "documents": [[]], "distances": [[]]}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        search_time = time.time() - start_time
        print(f"   검색 소요 시간: {search_time:.3f}초")
        return results

    def synthesize_input_with_contexts(self, user_input, contexts):
        reformatted_contexts = ""
        for context in contexts:
            reformatted_contexts += f"- {context}\n            "
        
        template = f"""
        <reference_documents>
            {reformatted_contexts}
        </reference_documents>

        사용자 질문: {user_input}
        
        위 <reference_documents>의 내용을 바탕으로 사용자의 질문에 답변하세요.
        """
        return template
        
    def generate_response(self, user_input):
        """
        [Main Flow] RAG 파이프라인 실행 및 답변 생성
        """
        print("##### [RAG Pipeline Start] #####")
        print(f"1. 유저 원본 입력: {user_input}")
        
        # Step 1: Rewrite
        print("\n### Step 1: Query Rewriting ###")
        queries_for_rag = self.query_rewriter.rewrite(user_input)
        print(f"-> 재작성된 쿼리 목록: {queries_for_rag}")

        # Step 2: Retrieve
        print("\n### Step 2: Information Retrieval ###")
        retrieved_results = []
        unique_docs = set()

        for query in queries_for_rag:
            relatives = self.search_relatives(query)
            for idx, (doc_id, document, distance) in enumerate(zip(relatives['ids'][0], relatives['documents'][0], relatives['distances'][0])):
                if document not in unique_docs:
                    print(f"   문서 발견 (ID: {doc_id}, Dist: {distance:.4f}): {document}")
                    retrieved_results.append(document)
                    unique_docs.add(document)

        # Step 3: Synthesize
        print("\n### Step 3: Context Synthesis ###")
        synthesized_input = self.synthesize_input_with_contexts(user_input, retrieved_results)
        print(f"-> 최종 프롬프트 길이: {len(synthesized_input)}자 생성됨")
        
        # Step 4: Generate
        print("\n### Step 4: LLM Generation ###")
        current_messages = self.history + [{"role": "user", "content": synthesized_input}]
        
        prompt_str = self.tokenizer.apply_chat_template(
            current_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=2048,
            repetition_penalty=1.05,
        )

        outputs = self.llm.generate([prompt_str], sampling_params)
        raw_output = outputs[0].outputs[0].text.strip()

        # Step 5: Parsing
        thinking_content = ""
        final_response = raw_output

        if "</think>" in raw_output:
            parts = raw_output.split("</think>")
            thinking_content = parts[0].replace("<think>", "").strip()
            if len(parts) > 1:
                final_response = parts[1].strip()
            else:
                final_response = ""

        # History Update (Clean version)
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": final_response})

        # =======================================================
        # [수정된 부분] Streamlit UI에 맞게 Return 값 구조 변경
        # =======================================================
        now_str = datetime.now().strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후")
        
        return {
            "response": final_response,      # 실제 답변
            "timestamp": now_str,            # 타임스탬프
            "details": {
                "thought": thinking_content,     # 사고 과정
                "rewritten_queries": queries_for_rag, # RAG용 재작성 쿼리
                "retrieved_docs": retrieved_results   # RAG용 검색 문서
            },
            "raw": raw_output                # 원본 (디버깅용)
        }
        
    def clear_history(self):
        if self.system_instructions:
            self.history = [{"role": "system", "content": self.system_instructions}]
        else:
            self.history = []
        print(">> Chat history cleared.")

# ==========================================
# 2. Streamlit UI Logic
# ==========================================

# CSS 로드 함수
def load_css():
    st.markdown("""
    <style>
        .chat-container { display: flex; flex-direction: column; gap: 10px; padding: 10px; }
        .chat-bubble { max-width: 70%; padding: 12px 16px; border-radius: 15px; position: relative; font-size: 16px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-bottom: 5px; }
        .bot-row { display: flex; justify-content: flex-start; align-items: flex-end; margin-bottom: 10px; }
        .bot-bubble { background-color: #F2F2F2; color: #000000; border-top-left-radius: 2px; }
        .user-row { display: flex; justify-content: flex-end; align-items: flex-end; margin-bottom: 10px; }
        .user-bubble { background-color: #FEE500; color: #000000; border-top-right-radius: 2px; }
        .timestamp { font-size: 10px; color: #888888; margin: 0 5px; min-width: 40px; }
        .streamlit-expanderHeader { font-size: 14px; color: #555; background-color: #fafafa; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 페이지 설정
st.set_page_config(page_title="나만의 페르소나 챗봇 메신저", layout="centered")
load_css()

# 초기 환영 메시지 정의
initial_message = {
    "role": "assistant",
    "content": "여러분의 페르소나와 이야기해보세요!",
    "timestamp": datetime.now().strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후"),
    "details": None
}

# ------------------------------------------------
# [중요] 봇 초기화 (vLLM 로딩은 세션당 1회만 수행)
# ------------------------------------------------
if "bot" not in st.session_state:
    with st.spinner("AI 모델을 로딩 중입니다... (시간이 걸릴 수 있습니다)"):
        # 실제 봇 클래스 인스턴스화
        # 주의: system_instructions 등은 필요에 따라 수정하세요.
        st.session_state.bot = QwenVLLMChatbotWithRAG(
            system_instructions=instructions.ahn_sungjae
        )

# 대화 기록 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [initial_message]

st.title("Persona AI Chatbot")

# ------------------------------------------------
# Sidebar: 설정 및 초기화
# ------------------------------------------------
with st.sidebar:
    if st.button("대화 내용 초기화", type="primary"):
        # 봇 내부 히스토리도 초기화 필요
        st.session_state.bot.clear_history()
        # 화면용 히스토리 리셋
        st.session_state.chat_history = [initial_message]
        st.rerun()

# ------------------------------------------------
# UI 렌더링: 채팅 기록 출력
# ------------------------------------------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            # 유저 메시지
            st.markdown(f"""
            <div class="user-row">
                <span class="timestamp">{msg['timestamp']}</span>
                <div class="chat-bubble user-bubble">
                    {msg['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            # 봇 메시지
            st.markdown(f"""
            <div class="bot-row">
                <div class="chat-bubble bot-bubble">
                    {msg['content']}
                </div>
                <span class="timestamp">{msg['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # [상세 보기 Expander] - RAG 정보 표시
            if msg.get("details"):
                with st.expander("🔍 AI 사고 과정 및 근거 보기"):
                    # 1. 쿼리 재구성
                    if msg['details'].get('rewritten_queries'):
                        st.markdown("**1. 질문 재구성 (Rewriting)**")
                        for q in msg['details']['rewritten_queries']:
                            st.code(q, language='text')

                    # 2. 참고 문서
                    if msg['details'].get('retrieved_docs'):
                        st.markdown("**2. 참고 문서 (Context)**")
                        for doc in msg['details']['retrieved_docs']:
                            st.success(doc)

                    # 3. 사고 과정
                    if msg['details'].get('thought'):
                        st.markdown("**3. 사고 과정 (Thinking)**")
                        st.info(msg['details']['thought'])

# ------------------------------------------------
# 입력 처리
# ------------------------------------------------
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 1. 유저 메시지 즉시 추가
    now_time = datetime.now().strftime("%p %I:%M").replace("AM", "오전").replace("PM", "오후")
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": now_time,
        "details": None
    })
    st.rerun() 

# ------------------------------------------------
# 답변 생성
# ------------------------------------------------
if st.session_state.chat_history[-1]["role"] == "user":
    with st.spinner("생각하는 중..."):
        # 실제 봇 로직 실행
        result = st.session_state.bot.generate_response(st.session_state.chat_history[-1]["content"])
        
        # 결과 저장
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"],
            "timestamp": result["timestamp"],
            "details": result["details"]
        })
        st.rerun()
