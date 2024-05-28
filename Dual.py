import asyncio
import dataclasses
import inspect
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import anthropic
import streamlit as st
from openai import AsyncOpenAI
from streamlit.delta_generator import DeltaGenerator

st.set_page_config(layout="wide")
st.title("Dual")

with st.sidebar:
    models = st.multiselect("Models", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240229", "gpt-4o", "gpt-4-turbo"], default=["claude-3-opus-20240229", "gpt-4o"])

def custom_logger(log):
    caller_frame = inspect.currentframe().f_back
    caller_filename = caller_frame.f_code.co_filename
    caller_funcname = caller_frame.f_code.co_name
    caller_lineno = caller_frame.f_lineno
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"{timestamp} - {caller_filename} - {caller_funcname} - {caller_lineno} - {log}"
    print(log_message, file=sys.stdout)


@dataclasses.dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionEngine(ABC):
    def __init__(self, model: str):
        self.model = model

    @classmethod
    def for_model(cls, model: str) -> "ChatCompletionEngine":
        return OpenAIEngine(model) if model.startswith("gpt") else AnthropicEngine(model)
        
    @abstractmethod
    async def complete(self, messages: list[Message], model: str) -> Message:
        pass


class OpenAIEngine(ChatCompletionEngine):
    async def complete(self, messages: list[Message]) -> Message:
        c = AsyncOpenAI()
        resp = await c.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[dataclasses.asdict(m) for m in messages]
        )
        content = resp.choices[0].message.content
        custom_logger(f"Got completion {content} from OpenAIEngine.")
        return Message(role="assistant", content=content)


class AnthropicEngine(ChatCompletionEngine):

    async def complete(self, messages: list[Message]) -> Message:
        c = anthropic.AsyncAnthropic()
        messages_ = [dataclasses.asdict(m) for m in messages if m.role != "system"]
        system = next(iter(m.content for m in messages if m.role == "system"), None)
        completion = await c.messages.create(
            max_tokens=1024,
            messages=messages_,
            model=self.model,
            system=system
        )
        content = completion.content[0].text
        custom_logger(f"Got completion {content} from AnthropicEngine.")
        return Message(role="assistant", content=content)


if 'messages' not in st.session_state:
    st.session_state.messages = [
        Message(
            role="system",
            content="Please write all responses as markdown. Thanks in advance."
        )
    ]

if 'completions' not in st.session_state:
    st.session_state.completions = ()


def printm(m: Message, out: DeltaGenerator | None = None):
    if m.role == "system":
        return
    if out is None:
        out = st.empty()
    with out.container():
        with st.chat_message(m.role):
            st.markdown(m.content)


for message in st.session_state.messages:
    printm(message)


def chat_input() -> Message | None:
    user_prompt = st.chat_input("Type a message...")
    custom_logger(f"Got {user_prompt} from user_prompt component")
    if not user_prompt:
        return
    return Message(role="user", content=user_prompt)


def get_user_selection():
    with st.container():
        selection = st.radio("Choose the best completion", models, index=None, horizontal=True)
        custom_logger(f"Got user selection {selection} from radio component.")
        return selection


def run_gather(*coros):
    async def f():
        return await asyncio.gather(*coros)

    return asyncio.run(f())


if prompt_msg := chat_input():
    st.session_state.messages.append(prompt_msg)
    printm(prompt_msg)
    st.session_state.completions = run_gather(*(ChatCompletionEngine.for_model(model).complete(st.session_state.messages) for model in models))
if any(x is not None for x in st.session_state.completions):
    columns = st.columns(len(models))
    for model_, completion_, col_ in zip(models, st.session_state.completions, columns):
        with col_:
            st.markdown(f"<h3 style='text-align: center;'>{model_}</h3>", unsafe_allow_html=True)
            printm(completion_)
    if user_selection := get_user_selection():
        st.session_state.messages.append(
            next(cmp for cmp, model in zip(st.session_state.completions, models) if model == user_selection)
        )
        st.session_state.completions = ()
        st.rerun()
