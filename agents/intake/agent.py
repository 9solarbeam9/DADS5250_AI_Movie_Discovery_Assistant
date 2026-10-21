from langchain_core.messages import AIMessage, HumanMessage, convert_to_openai_messages
from langgraph.graph import START, MessagesState, StateGraph

from core.llm import MODEL, client
from db.request_types import search

PROMPT = """You are a Boston 311 intake assistant. Pick the request type below that best matches the resident's problem,
and reply with its type, department and queue. If none fits or the message is too vague, ask one short question instead.
Only use request types from this list.

{context}"""


class State(MessagesState):
    context: str


def retrieve(state: State) -> dict:
    # joins every user turn into one query, add an LLM query rewrite if follow-ups drift
    query = " ".join(m.text for m in state["messages"] if m.type == "human")
    hits = search(query, k=10)
    return {"context": "\n".join(f"- {h.type} ({h.reason}, {h.subject}): {h.department_name} ({h.department}), queue {h.queue}" for h in hits.itertuples())}


def generate(state: State) -> dict:
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": PROMPT.format(context=state["context"])}, *convert_to_openai_messages(state["messages"])],
    )
    return {"messages": [AIMessage(reply.choices[0].message.content)]}


graph = (
    StateGraph(State)
    .add_node(retrieve)
    .add_node(generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "generate")
    .compile()
)


def ask(question: str, history: list[dict]) -> str:
    """`history` holds prior turns as {"role", "content"} dicts, so the caller owns the conversation."""
    state = graph.invoke({"messages": [*history, HumanMessage(question)]})
    return state["messages"][-1].content


if __name__ == "__main__":
    history = []
    while question := input("\nYou: ").strip():
        answer = ask(question, history)
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        print(f"\nBot: {answer}")
