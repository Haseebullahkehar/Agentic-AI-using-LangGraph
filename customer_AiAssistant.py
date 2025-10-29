import streamlit as st
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

# --- Mock models for local testing ---
class MockStructuredModel:
    def invoke(self, prompt):
        class Sentiment:
            sentiment = "positive" if "love" in prompt.lower() or "great" in prompt.lower() else "negative"
        return Sentiment()

class MockStructuredModel2:
    def invoke(self, prompt):
        class DiagnosisSchema:
            def model_dump(self):
                if "late" in prompt.lower() or "delay" in prompt.lower():
                    return {"issue_type": "delivery delay", "tone": "frustrated", "urgency": "high"}
                elif "broken" in prompt.lower():
                    return {"issue_type": "damaged product", "tone": "angry", "urgency": "high"}
                else:
                    return {"issue_type": "general dissatisfaction", "tone": "upset", "urgency": "medium"}
        return DiagnosisSchema()

class MockModel:
    def invoke(self, prompt):
        class Response:
            content = (
                "Dear Customer,\n\n"
                "Thank you for your wonderful feedback! We truly appreciate your support. "
                "Please consider leaving a review on our website!\n\n"
                "Warm regards,\nTeam"
                if "thank-you message" in prompt
                else "We’re sorry to hear about your issue. We’ve escalated this to our support team for quick resolution."
            )
        return Response()

# --- Initialize mock models ---
structured_model = MockStructuredModel()
structured_model2 = MockStructuredModel2()
model = MockModel()


# --- Define the state schema ---
class ReviewState(TypedDict):
    review: str
    sentiment: str
    diagnosis: dict
    response: str


# --- Node 1: Find Sentiment ---
def find_sentiment(state: ReviewState):
    prompt = f"For the following review find out the sentiment:\n{state['review']}"
    sentiment = structured_model.invoke(prompt).sentiment
    return {"sentiment": sentiment}


# --- Conditional function ---
def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"


# --- Node 2: Positive Response ---
def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    "{state['review']}"
    Also, kindly ask the user to leave feedback on our website.
    """
    response = model.invoke(prompt).content
    return {"response": response}


# --- Node 3: Run Diagnosis ---
def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:
    {state['review']}
    Return issue_type, tone, and urgency.
    """
    response = structured_model2.invoke(prompt)
    return {"diagnosis": response.model_dump()}


# --- Node 4: Negative Response ---
def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]
    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', 
and marked urgency '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
    """
    response = model.invoke(prompt).content
    return {"response": response}


# --- Build the graph ---
graph = StateGraph(ReviewState)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()


# --- Streamlit App ---
st.set_page_config(page_title="Review Sentiment Workflow", page_icon="💬", layout="centered")

st.title("💬 Customer Review Sentiment Workflow")
st.write("Enter a review below to see how the LangGraph workflow analyzes and responds.")

# User input
user_review = st.text_area("✍️ Paste a review here:", height=150)

if st.button("Analyze Review"):
    if not user_review.strip():
        st.warning("Please enter a review first.")
    else:
        # Run the workflow
        initial_state = {"review": user_review}
        result = workflow.invoke(initial_state)

        # Display results
        st.subheader("🧠 Workflow Output")
        st.write("**Sentiment:**", result.get("sentiment", "N/A"))

        if "diagnosis" in result:
            st.write("**Diagnosis:**")
            st.json(result["diagnosis"])

        st.subheader("💌 Generated Response")
        st.success(result["response"])
