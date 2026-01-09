from typing import TypedDict , List , Literal
from langchain_core.messages import BaseMessage , HumanMessage , AIMessage
from pydantic import BaseModel , Field
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig 
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from Config import GROQ_API_KEY , TAVILY_API_KEY , PINECONE_API_KEY
from vector_store import get_retriever

os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

tavily = TavilySearch(max_results=3, topic="general")

class RouteDecision(BaseModel):
    route : Literal['rag','web','answer','end']
    reply : str | None = Field(None,description="Filled only when route == 'end' ")

class RagJudge(BaseModel):
    suffiecient : bool = Field(...,description="True if retrieved information is sufficient to answer the user's question , False otherwise")

# llms
llm_router = ChatGroq(model="llama3-70b-8192",temperature=0).with_structured_output(RouteDecision)
llm_judger = ChatGroq(model="llama3-70b-8192",temperature=0).with_structured_output(RagJudge)
llm_answerer = ChatGroq(model="llama3-70b-8192",temperature=0.7)

# AgentState
class AgentState(TypedDict) : 
    messages : List[BaseMessage]
    route: Literal['rag','web','answer','end']
    rag: str
    web: str
    web_search_enabled: bool


@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        result = tavily.invoke({"query": query})
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            return str(result)
    except Exception as e:
        return f"WEB_ERROR::{e}"

@tool
def rag_search_tool(query: str) -> str:
    """Top-K chunks from KB (empty string if none)"""
    try:
        retriever_instance = get_retriever()
        docs = retriever_instance.invoke(query, k=5) # Increased from 3 to 5
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"
    

def router_node(state:AgentState) -> AgentState :
    print("Entering router node")
    query = next((m.content for m in reversed (state['messages']) if isinstance(m,HumanMessage)),"") 

    web_search_enabled = state.get("web_search_enabled",True)
    print(f"Router recieved web search info : {web_search_enabled}")

    sys_prompt = (
        "You are an intelligent routing agent designed to direct the user queries to the most appropriate tool."
        "Your primary goal is to provide accurate and relevant information by selecting the best source."
        "Prioritize using the **internal knowladge base (RAG)** for factual information that is likely "
        "to be contained within pre-uploaded documents or for common, well-established facts."
    )

    if web_search_enabled:
        sys_prompt+=(
            "You can use web search for queries tha require very current,real-time,or broad general knowladge "
            "that is unlikely to be in a specific , static knowladge base (e.g., today's news , live data , very recet events)"
            "\n\nChoose one of the following routes:"
            "\n- 'rag' For queries about specific entities , historical facts , product details , procedures , or any information that would typically be found in a accurated document collection"
            "\n- 'web' For queries up-to-date internet access"
        )
    else : 
        sys_prompt+=(
            "**WEB Search is currently DISABLED.**You **MUST NOT** choose the 'web' route."
            "if a query would normally require web search , you should attempt to answer it using RAG if applicable pr directly from genral knowladge"
            "\n\nChoose one of the following routes:"
            "\n- 'rag' For queries about specific entities , historical facts , product details , procedures , or any information that would typically be found in a accurated document collection"
            "\n- 'answer' For very simple,direct questions you can answer without any external lookup"
        ) 

    system_prompt += (
    "\n- 'answer': For very simple, direct questions you can answer without any external lookup (e.g., 'What is your name?')."
    "\n- 'end': For pure greetings or small-talk where no factual answer is expected (e.g., 'Hi', 'How are you?'). If choosing 'end', you MUST provide a 'reply'."
    "\n\nExample routing decisions:"
    "\n- User: 'What are the treatment of diabetes?' -> Route: 'rag' (Factual knowledge, likely in KB)."
    "\n- User: 'What is the capital of France?' -> Route: 'rag' (Common knowledge, can be in KB or answered directly if LLM knows)."
    "\n- User: 'Who won the NBA finals last night?' -> Route: 'web' (Current event, requires live data)."
    "\n- User: 'How do I submit an expense report?' -> Route: 'rag' (Internal procedure)."
    "\n- User: 'Tell me about quantum computing.' -> Route: 'rag' (Foundational knowledge can be in KB. If KB is sparse, judge will route to web if enabled)."
    "\n- User: 'Hello there!' -> Route: 'end', reply='Hello! How can I assist you today?'")


    messages = [
        ('system',sys_prompt),
        ('user',query)
    ]

    result : RouteDecision = llm_router.invoke(messages)
    initial_router_decision = result.route

    router_override_reason = None

    if not web_search_enabled and result.route == "web":
        # If web search is disabled, force it to try RAG instead
        result.route = "rag" 
        router_override_reason = "Web search disabled by user; redirected to RAG."
        print(f"Router decision overridden: changed from 'web' to 'rag' because web search is disabled.")


    print(f"Router final decision: {result.route}, Reply (if 'end'): {result.reply}")

    out = {
        "messages": state["messages"], 
        "route": result.route,
        "web_search_enabled": web_search_enabled # Pass the flag along in the state
    }
    if router_override_reason: # Add override info for tracing
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello!")]
    
    print("--- Exiting router_node ---")
    return out



def rag_node(state: AgentState,config:RunnableConfig) -> AgentState:
    print("\n--- Entering rag_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    # MODIFIED: Get web_search_enabled directly from the config
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True) # <-- CHANGED LINE
    print(f"Router received web search info : {web_search_enabled}")
    print(f"RAG query: {query}")
    chunks = rag_search_tool.invoke(query)
    
    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error: {chunks}. Checking web search enabled status.")
        # If RAG fails, and web search is enabled, try web. Otherwise, go to answer.
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route}

    if chunks:
        print(f"Retrieved RAG chunks (first 500 chars): {chunks[:500]}...")
    else:
        print("No RAG chunks retrieved.")

    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "Consider if the retrieved text directly addresses the question's core and provides enough detail."
            "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient."
            "\n\nRespond ONLY with a JSON object: {\"sufficient\": true/false}"
            "\n\nExample 1: Question: 'What is the capital of France?' Retrieved: 'Paris is the capital of France.' -> {\"sufficient\": true}"
            "\nExample 2: Question: 'What are the symptoms of diabetes?' Retrieved: 'Diabetes is a chronic condition.' -> {\"sufficient\": false} (Doesn't answer symptoms)"
            "\nExample 3: Question: 'How to fix error X in software Y?' Retrieved: 'No relevant information found.' -> {\"sufficient\": false}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
    ]
    verdict: RagJudge = llm_judger.invoke(judge_messages)
    print(f"RAG Judge verdict: {verdict.sufficient}")
    print("--- Exiting rag_node ---")
    
    # NEW LOGIC: Decide next route based on sufficiency AND web_search_enabled
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer" # If not sufficient, only go to web if enabled
        print(f"RAG not sufficient. Web search enabled: {web_search_enabled}. Next route: {next_route}")

    return {
        **state,
        "rag": chunks,
        "route": next_route,
        "web_search_enabled": web_search_enabled # Pass the flag along
    }



def web_node(state: AgentState,config:RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    # Check if web search is actually enabled before performing it
    # MODIFIED: Get web_search_enabled directly from the config
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True) # <-- CHANGED LINE
    print(f"Router received web search info : {web_search_enabled}")
    if not web_search_enabled:
        print("Web search node entered but web search is disabled. Skipping actual search.")
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    print(f"Web search query: {query}")
    snippets = web_search_tool.invoke(query)
    
    if snippets.startswith("WEB_ERROR::"):
        print(f"Web Error: {snippets}. Proceeding to answer with limited info.")
        return {**state, "web": "", "route": "answer"}

    print(f"Web snippets retrieved: {snippets[:200]}...")
    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}



def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    
    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        # If web search was disabled, the 'web' field might contain a message like "Web search was disabled..."
        # We should only include actual search results here.
        if state["web"] and not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
    
    context = "\n\n".join(ctx_parts)
    if not context.strip():
        context = "No external context was available for this query. Try to answer based on general knowledge if possible."

    prompt = f"""Please answer the user's question using the provided context.
If the context is empty or irrelevant, try to answer based on your general knowledge.

Question: {user_q}

Context:
{context}

Provide a helpful, accurate, and concise response based on the available information."""

    print(f"Prompt sent to answer_llm: {prompt[:500]}...")
    ans = llm_answerer.invoke([HumanMessage(content=prompt)]).content
    print(f"Final answer generated: {ans[:200]}...")
    print("--- Exiting answer_node ---")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }


def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_) -> Literal["answer"]:
    return "answer"

def build_agent():
    """Builds and compiles the LangGraph agent."""
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")
    
    g.add_conditional_edges(
        "router",
        from_router,
        {
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END
        }
    )
    
    g.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {
            "answer": "answer",
            "web": "web_search"
        }
    )
    
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    agent = g.compile(checkpointer=MemorySaver())
    return agent

rag_agent = build_agent()