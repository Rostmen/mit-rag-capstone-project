import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's questions concisely."

USE_HISTORY = False  # set False to run the no-memory version


def require_api_key() -> None:
    """Exit early with a clear message if OPENROUTER_API_KEY is not set, instead of
    failing later with a KeyError when the model client is created."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "\n[setup] OPENROUTER_API_KEY is not set.\n"
            "  1. Get a free key at https://openrouter.ai/keys\n"
            "  2. Create a file named '.env' in this folder with one line:\n"
            "         OPENROUTER_API_KEY=sk-or-your-key-here\n"
            "     or set it in your shell  (Windows: setx OPENROUTER_API_KEY sk-or-... ;\n"
            "     macOS/Linux: export OPENROUTER_API_KEY=sk-or-...).\n"
        )


def chat_loop(response):
    """Minimal command-line chat loop: Reads input, prints response(input). (Provided.)"""
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        try:
            result = response(user_input)
        except Exception as e:
            print(f"Error: {e}")
            continue
        print(f"\nAssistant: {result}\n")


def make_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def build_no_history_respond():
    llm = make_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    chain = prompt | llm
    def respond(user_input: str) -> str:
        reply = chain.invoke({"question": user_input})
        return reply.content

    return respond


def build_history_respond():
    llm = make_llm()
    history = [SystemMessage(content=SYSTEM_PROMPT)]

    def respond(user_input: str) -> str:
        history.append(HumanMessage(content=user_input))

        response = llm.invoke(history)
        reply = response.content

        history.append(AIMessage(content=reply))
        return reply

    return respond


if __name__ == "__main__":
    require_api_key()
    respond = build_history_respond() if USE_HISTORY else build_no_history_respond()
    chat_loop(respond)
