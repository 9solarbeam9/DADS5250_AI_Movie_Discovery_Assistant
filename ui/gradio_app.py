import gradio as gr

from agents.intake.agent import ask
from core.llm import MODEL


def chat(message: str, history: list[dict]) -> str:
    return ask(message, [{"role": m["role"], "content": m["content"]} for m in history])


demo = gr.ChatInterface(chat, title=f"Boston 311 Intake ({MODEL})")
