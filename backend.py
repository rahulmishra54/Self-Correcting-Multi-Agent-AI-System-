
import os
import json
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


# ==========================================
# 1. STATE
# ==========================================

class State(TypedDict):
    topic: str
    answer: str
    revision_count: int
    feedback: str
    decision: str


MAX_REVISIONS = 3


# ==========================================
# 2. LLM
# ==========================================

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


# ==========================================
# 3. WRITER AGENT
# ==========================================

def writer(state: State):

    print("\n--- WRITER AGENT ---")

    response = get_llm().invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the WRITER agent in a self-correcting multi-agent system. "
                    "You are a beginner-friendly teacher. Explain the topic in 120-160 words. "
                    "Use simple language, one everyday analogy, and one tiny example."
                ),
            },
            {
                "role": "user",
                "content": f"Explain: {state['topic']}",
            },
        ]
    )

    return {
        "answer": response.content,
        "feedback": "",
        "decision": "",
        "revision_count": 0,
    }


# ==========================================
# 4. REVIEWER AGENT
# ==========================================

def reviewer(state: State):

    print("\n--- REVIEWER AGENT ---")

    response = get_llm().invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the REVIEWER agent in a self-correcting multi-agent system. "
                    "Check the answer using ONLY these rules:\n"
                    "1. It is easy for a beginner.\n"
                    "2. It contains an everyday analogy.\n"
                    "3. It contains a tiny concrete example.\n"
                    "4. It stays focused on the requested topic.\n"
                    "If any rule fails, choose REVISE and give one or two precise improvements.\n\n"
                    "Return ONLY valid JSON in exactly this shape:\n"
                    '{"decision":"PASS|REVISE","feedback":"..."}\n'
                    "When the decision is PASS, feedback must be an empty string."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {state['topic']}\n\n"
                    f"Answer:\n{state['answer']}"
                ),
            },
        ]
    )

    try:
        review = json.loads(response.content)

        decision = review["decision"].upper()

        if decision not in ["PASS", "REVISE"]:
            raise ValueError("Invalid reviewer decision")

        feedback = review["feedback"]

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        decision = "REVISE"
        feedback = "Return valid JSON with a PASS or REVISE decision."

    print("Decision:", decision)
    print("Feedback:", feedback)

    return {
        "decision": decision,
        "feedback": feedback,
    }


# ==========================================
# 5. REVISER AGENT
# ==========================================

def reviser(state: State):

    print("\n--- REVISER AGENT ---")

    response = get_llm().invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are the REVISER agent in a self-correcting multi-agent system. "
                    "Improve the existing answer using reviewer feedback. "
                    "Keep it beginner-friendly and concise. "
                    "Return only the improved answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {state['topic']}\n\n"
                    f"Current answer:\n{state['answer']}\n\n"
                    f"Reviewer feedback:\n{state['feedback']}"
                ),
            },
        ]
    )

    print("Revision completed.")

    return {
        "answer": response.content,
        "revision_count": state["revision_count"] + 1,
    }


# ==========================================
# 6. CONDITIONAL ROUTING
# ==========================================

def route_after_review(state: State) -> Literal["done", "revise"]:

    if state["decision"] == "PASS":
        return "done"

    if state["revision_count"] >= MAX_REVISIONS:
        print("\nMaximum revision limit reached.")
        return "done"

    return "revise"


# ==========================================
# 7. BUILD GRAPH
# ==========================================

builder = StateGraph(State)


# Register nodes
builder.add_node("writer", writer)
builder.add_node("reviewer", reviewer)
builder.add_node("reviser", reviser)


# Starting point
builder.add_edge(START, "writer")


# Writer -> Reviewer
builder.add_edge("writer", "reviewer")


# Conditional routing
builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "revise": "reviser",
        "done": END,
    }
)


# Reviser -> Reviewer
builder.add_edge("reviser", "reviewer")


# Compile graph
graph = builder.compile()


# ==========================================
# 8. EXECUTE GRAPH
# ==========================================

if __name__ == "__main__":

    topic = input("Enter your topic: ")

    initial_state: State = {
        "topic": topic,
        "answer": "",
        "revision_count": 0,
        "feedback": "",
        "decision": "",
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 20}
    )

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nFinal Decision:", result["decision"])
    print("Total Revisions:", result["revision_count"])

    if result["decision"] == "PASS":
        print("Status: APPROVED")
    else:
        print("Status: NOT APPROVED (revision limit reached)")
