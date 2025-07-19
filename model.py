import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Initialize Gemini LLM
llm = None

def initialize_llm(api_key):
    """Initialize the Gemini LLM with API key"""
    global llm
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest", 
        google_api_key=api_key,
        temperature=0.2
    )

def classify_symptom(state: dict) -> dict:
    """Node to classify the symptom"""
    prompt = (
        "You are a helpful Medical Assistant. Classify the symptoms below into one of the categories:\n"
        "- General\n"
        "- Emergency\n" 
        "- Mental Health\n"
        f"Symptom: {state['symptom']}\n"
        "Respond only with one word: General, Emergency, or Mental Health\n"
        "Example: Input: I have fever, Output: General"
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip()
    state["category"] = category
    return state

def symptom_router(state: dict) -> str:
    """Router logic to route to the correct node"""
    cat = state["category"].lower()
    if "general" in cat:
        return "general"
    elif "emergency" in cat:
        return "emergency" 
    elif "mental" in cat:
        return "mental_health"
    else:
        return "general"

def general_node(state: dict) -> dict:
    """Handle general medical symptoms"""
    state["answer"] = f"'{state['symptom']}' seems general. We're directing you to the general ward for consultation with a doctor."
    state["recommendation"] = "Visit General Ward"
    state["urgency"] = "Low"
    return state

def emergency_node(state: dict) -> dict:
    """Handle emergency medical symptoms"""
    state["answer"] = f"'{state['symptom']}' indicates a medical emergency. Please seek immediate medical help!"
    state["recommendation"] = "Emergency Room - Immediate Attention Required"
    state["urgency"] = "Critical"
    return state

def mental_health_node(state: dict) -> dict:
    """Handle mental health symptoms"""
    state["answer"] = f"'{state['symptom']}' seems like a mental health issue. We recommend speaking with our counselor."
    state["recommendation"] = "Mental Health Counseling"
    state["urgency"] = "Medium"
    return state

def build_graph():
    """Build and compile the LangGraph"""
    builder = StateGraph(dict)
    
    # Add nodes
    builder.add_node("classify", classify_symptom)
    builder.add_node("general", general_node)
    builder.add_node("emergency", emergency_node)
    builder.add_node("mental_health", mental_health_node)
    
    # Set entry point
    builder.set_entry_point("classify")
    
    # Add conditional edges
    builder.add_conditional_edges("classify", symptom_router, {
        "general": "general",
        "emergency": "emergency",
        "mental_health": "mental_health"
    })
    
    # Add edges to END
    builder.add_edge("general", END)
    builder.add_edge("emergency", END)
    builder.add_edge("mental_health", END)
    
    return builder.compile()

def process_symptom(symptom: str, api_key: str = None):
    """Main function to process a symptom and return classification"""
    if api_key:
        initialize_llm(api_key)
    
    if llm is None:
        raise ValueError("LLM not initialized. Please provide API key.")
    
    # Build graph
    graph = build_graph()
    
    # Process symptom
    initial_state = {"symptom": symptom}
    final_state = graph.invoke(initial_state)
    
    return {
        "symptom": final_state.get("symptom"),
        "category": final_state.get("category"),
        "answer": final_state.get("answer"),
        "recommendation": final_state.get("recommendation", ""),
        "urgency": final_state.get("urgency", "")
    }