from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph

from typing_extensions import Annotated, TypedDict
from typing import Sequence 

# ================== CONSTANTS DEFINITION =================

APP_CONFIG = {"configurable": {"thread_id": "acd612"}}
SECRET_KEY = config('GEMINI_API_KEY')
CHATBOT_MODEL_NODE = 'model'

# =========================================================

ASSISTANT_SETUP_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a helpful assistant." +\
            "Answer all the following questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

GOOGLE_LLM = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=SECRET_KEY)

# ==================================================

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# ==================================================

class Chatbot:
    def __init__(self, model, prompt, config) -> None:
        self._model = model
        self._prompt = prompt
        self._config = config
        self._app = self._build_app()

    def _call_model(self, state: State) -> dict:
        chain = self._prompt | self._model
        response = chain.invoke(state)
        return {"messages": [response]}
    
    def _build_app(self) -> StateGraph:
        """"""        
        memory = MemorySaver()
        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, CHATBOT_MODEL_NODE)
        workflow.add_node(CHATBOT_MODEL_NODE, self._call_model)
        app = workflow.compile(checkpointer=memory)
        return app

    def _process(self, state):
        output = self._app.invoke(state, self._config)
        return output
    
    def response(self, query: str) -> None:
        state = {"messages": [HumanMessage(query)]}
        output = self._process(state)
        output["messages"][-1].pretty_print()

# ==================================================

chat = Chatbot(
    model = GOOGLE_LLM,
    prompt = ASSISTANT_SETUP_PROMPT,
    config = APP_CONFIG
)

query = "Hi! I'm Bob."
chat.response(query)

query = "What is my name?"
chat.response(query)