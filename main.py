import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# LLM and Agent Setup
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# Web Search Tool
@tool
def web_search_tool(question: str):
    """Search the web for information."""
    
    search = DuckDuckGoSearchRun()
    result = search.run(question)

    return result


def get_LLM_response(question):

    try:
        
        model = init_chat_model(
            # "openrouter:google/gemma-3-27b-it",
            "openrouter:openai/gpt-4o-mini",
            # groq:llama-3.3-70b-versatile
            temperature=0.5
        )

        agent = create_agent(
            model=model,
            tools=[web_search_tool],

            system_prompt=
            '''
            You are a smart, friendly, and helpful AI assistant.

Explain everything in very simple and beginner-friendly words.

Rules:
- Keep responses clean and easy to understand.
- Give short answers first, then details if needed.
- Use bullet points and step-by-step explanations.
- For coding questions:
  1. Explain the concept
  2. Show simple code
  3. Add comments in code
  4. Mention important notes

- Use real-world examples whenever possible.
- Avoid overly technical jargon.
- If the question needs recent information, use the web search tool.
- If you do not know something, clearly say so.
- Be conversational and supportive.
You are a helpful AI assistant.
            '''
        )

        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

        return response["messages"][-1].content

    except Exception as e:

        return "⚠️ Response too large. Try asking in smaller parts."

# res=get_LLM_response("what is the current temperature in delhi india?")
# print(res["messages"][-1].content)


# Streamlit UI

import streamlit as st


st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Chatbot")

user_input = st.text_input("Ask something")

if user_input:
    with st.spinner("Thinking..."):
        res = get_LLM_response(user_input)
    st.write(res)