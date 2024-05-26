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
    oai_model = st.selectbox("OpenAI Model", ["gpt-4o", "gpt-4-turbo"])
    ant_model = st.selectbox("Anthropic Model",
                             ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240229"])


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
    @abstractmethod
    async def complete(self, messages: list[Message], model: str) -> Message:
        pass


class OpenAIEngine(ChatCompletionEngine):
    async def complete(self, messages: list[Message], model: str) -> Message:
        c = AsyncOpenAI()
        resp = await c.chat.completions.create(
            model='gpt-4o',
            max_tokens=1024,
            messages=[dataclasses.asdict(m) for m in messages]
        )
        content = resp.choices[0].message.content
        custom_logger(f"Got completion {content} from OpenAIEngine.")
        return Message(role="assistant", content=content)


class AnthropicEngine(ChatCompletionEngine):

    async def complete(self, messages: list[Message], model: str) -> Message:
        c = anthropic.AsyncAnthropic()
        messages_ = [dataclasses.asdict(m) for m in messages if m.role != "system"]
        system = next(iter(m.content for m in messages if m.role == "system"), None)
        completion = await c.messages.create(
            max_tokens=1024,
            messages=messages_,
            model='claude-3-opus-20240229',
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

if 'openai' not in st.session_state:
    st.session_state.openai = OpenAIEngine()
if 'anthropic' not in st.session_state:
    # golden gate bridge claude coming soon
    st.session_state.anthropic = AnthropicEngine()

if 'oai_cmp' not in st.session_state:
    st.session_state.oai_cmp = None

if 'ant_cmp' not in st.session_state:
    st.session_state.ant_cmp = None


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
        selection = st.radio("Choose the best completion", ["OpenAI", "Anthropic"], index=None, horizontal=True)
        custom_logger(f"Got user selection {selection} from radio component.")
        return selection


def run_gather(*coros):
    async def f():
        return await asyncio.gather(*coros)

    return asyncio.run(f())


if prompt_msg := chat_input():
    st.session_state.messages.append(prompt_msg)
    printm(prompt_msg, None)
    st.session_state.oai_cmp, st.session_state.ant_cmp = run_gather(
        st.session_state.openai.complete(st.session_state.messages, model=oai_model),
        st.session_state.anthropic.complete(st.session_state.messages, model=ant_model),
    )
if any(x is not None for x in (st.session_state.oai_cmp, st.session_state.ant_cmp)):
    oai, ant = st.columns(2)
    with oai:
        st.markdown(f"<h3 style='text-align: center;'>OpenAI ({oai_model})</h3>", unsafe_allow_html=True)
        printm(st.session_state.oai_cmp)
    with ant:
        st.markdown(f"<h3 style='text-align: center;'>Anthropic ({ant_model})</h3>", unsafe_allow_html=True)
        printm(st.session_state.ant_cmp)
    if user_selection := get_user_selection():
        st.session_state.messages.append(
            st.session_state.oai_cmp if user_selection == "OpenAI" else st.session_state.ant_cmp
        )
        st.session_state.oai_cmp, st.session_state.ant_cmp = None, None
        st.rerun()
