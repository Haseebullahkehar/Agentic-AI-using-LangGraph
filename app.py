import streamlit as st
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv

# ---------------------- Load Environment ----------------------
load_dotenv()

# ---------------------- Model Setup ----------------------
model = ChatOpenAI(model="gpt-4o-mini")

# Pydantic structure for LLM output
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# structured output model
structured_model = model.with_structured_output(EvaluationSchema)

# ---------------------- Graph State ----------------------
class CSSState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# ---------------------- Nodes ----------------------
def evaluate_language(state: CSSState):
    prompt = f"Evaluate the language quality of the following essay and provide feedback and assign a score out of 10:\n\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"language_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_analysis(state: CSSState):
    prompt = f"Evaluate the depth of analysis of the following essay and provide feedback and assign a score out of 10:\n\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"analysis_feedback": output.feedback, "individual_scores": [output.score]}

def evaluate_thought(state: CSSState):
    prompt = f"Evaluate the clarity of thought of the following essay and provide feedback and assign a score out of 10:\n\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback": output.feedback, "individual_scores": [output.score]}

def final_evaluation(state: CSSState):
    prompt = (
        f"Summarize the overall performance of the essay based on the following feedbacks:\n"
        f"Language Feedback: {state['language_feedback']}\n"
        f"Depth of Analysis: {state['analysis_feedback']}\n"
        f"Clarity of Thought: {state['clarity_feedback']}"
    )
    overall_feedback = model.invoke(prompt).content
    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {"overall_feedback": overall_feedback, "avg_score": avg_score}

# ---------------------- Graph ----------------------
graph = StateGraph(CSSState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# edges
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="CSS Essay Evaluator", layout="centered")

st.title("🧠 AI Essay Evaluator (CSS Exam Style)")
st.markdown("Paste your essay below and click **Evaluate** to get structured feedback and a score.")

essay_input = st.text_area("✍️ Your Essay:", height=300, placeholder="Type or paste your essay here...")

if st.button("Evaluate Essay"):
    if not essay_input.strip():
        st.warning("Please enter an essay before evaluating.")
    else:
        with st.spinner("Evaluating essay... Please wait ⏳"):
            initial_state = {"essay": essay_input}
            result = workflow.invoke(initial_state)

        # ---------------------- Display Results ----------------------
        st.subheader("🧩 Evaluation Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Language", result["individual_scores"][0])
        col2.metric("Analysis", result["individual_scores"][1])
        col3.metric("Clarity", result["individual_scores"][2])

        st.markdown("---")
        st.markdown("### 🗣️ Feedback by Category")

        with st.expander("Language Feedback"):
            st.write(result["language_feedback"])

        with st.expander("Depth of Analysis Feedback"):
            st.write(result["analysis_feedback"])

        with st.expander("Clarity of Thought Feedback"):
            st.write(result["clarity_feedback"])

        st.markdown("---")
        st.subheader("🏁 Overall Feedback")
        st.write(result["overall_feedback"])

        st.success(f"✅ **Average Score:** {result['avg_score']:.2f}/10")
