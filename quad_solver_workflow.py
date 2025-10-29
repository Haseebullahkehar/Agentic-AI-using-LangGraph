import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
import math
import numpy as np
import matplotlib.pyplot as plt

# --- Define the state schema ---
class QuadState(TypedDict):
    a: float
    b: float
    c: float
    equation: str
    discriminant: float
    result: str


# --- Node 1: Show the quadratic equation ---
def show_equation(state: QuadState):
    equation = f"{state['a']}x² + {state['b']}x + {state['c']} = 0"
    return {'equation': equation}


# --- Node 2: Calculate discriminant ---
def calculate_discriminant(state: QuadState):
    discriminant = state['b']**2 - (4 * state['a'] * state['c'])
    return {'discriminant': discriminant}


# --- Node 3: Real roots ---
def real_roots(state: QuadState):
    root1 = (-state['b'] + math.sqrt(state['discriminant'])) / (2 * state['a'])
    root2 = (-state['b'] - math.sqrt(state['discriminant'])) / (2 * state['a'])
    result = f"The roots are {root1:.2f} and {root2:.2f}"
    return {'result': result}


# --- Node 4: Repeated roots ---
def repeated_roots(state: QuadState):
    root = -state['b'] / (2 * state['a'])
    result = f"The repeated root is {root:.2f}"
    return {'result': result}


# --- Node 5: No real roots ---
def no_real_roots(state: QuadState):
    result = "No real roots exist (discriminant < 0)."
    return {'result': result}


# --- Conditional branching ---
def check_condition(state: QuadState) -> Literal['real_roots', 'repeated_roots', 'no_real_roots']:
    if state['discriminant'] > 0:
        return 'real_roots'
    elif state['discriminant'] == 0:
        return 'repeated_roots'
    else:
        return 'no_real_roots'


# --- Build LangGraph workflow ---
graph = StateGraph(QuadState)

graph.add_node('show_equation', show_equation)
graph.add_node('calculate_discriminant', calculate_discriminant)
graph.add_node('real_roots', real_roots)
graph.add_node('repeated_roots', repeated_roots)
graph.add_node('no_real_roots', no_real_roots)

graph.add_edge(START, 'show_equation')
graph.add_edge('show_equation', 'calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant', check_condition, {
    'real_roots': 'real_roots',
    'repeated_roots': 'repeated_roots',
    'no_real_roots': 'no_real_roots'
})

graph.add_edge('real_roots', END)
graph.add_edge('repeated_roots', END)
graph.add_edge('no_real_roots', END)

workflow = graph.compile()


# --- STREAMLIT UI IMPLEMENTATION ---
st.set_page_config(page_title="Quadratic Solver - LangGraph", page_icon="🧮", layout="centered")
st.title("🧮 Quadratic Equation Solver using LangGraph")
st.markdown(
    """
    This interactive app demonstrates how **LangGraph workflows** can automate mathematical reasoning.
    
    👉 Enter values of coefficients **a**, **b**, and **c** below to compute:
    - The equation
    - The discriminant
    - The nature of roots (real, repeated, or complex)
    """
)

# Input section
st.subheader("Enter Coefficients")
a = st.number_input("Coefficient a:", value=1.0, step=0.5)
b = st.number_input("Coefficient b:", value=-3.0, step=0.5)
c = st.number_input("Coefficient c:", value=2.0, step=0.5)

if st.button("🔍 Solve Equation"):
    try:
        with st.spinner("Running LangGraph workflow..."):
            initial_state = {'a': a, 'b': b, 'c': c}
            result = workflow.invoke(initial_state)

        st.success("✅ Workflow Completed Successfully!")

        st.markdown("### 🧾 Equation")
        st.latex(result['equation'].replace("x²", "x^2"))

        st.markdown("### 🔢 Discriminant")
        st.code(f"Δ = {result['discriminant']:.2f}", language="text")

        st.markdown("### 🧠 Result")
        st.info(result['result'])

        # Optional plot for real roots visualization
        if result['discriminant'] >= 0:

            x = np.linspace(-10, 10, 400)
            y = a * x**2 + b * x + c

            fig, ax = plt.subplots()
            ax.plot(x, y, label=f"{a}x² + {b}x + {c}")
            ax.axhline(0, color="black", linestyle="--", lw=1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Graph of Quadratic Equation")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

