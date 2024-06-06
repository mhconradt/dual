import asyncio
import dataclasses
import inspect
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import anthropic
import streamlit as st
from anthropic.types import ContentBlockDeltaEvent
from openai import AsyncOpenAI
from streamlit.delta_generator import DeltaGenerator

st.set_page_config(layout="wide")
st.title("Dual")

ANTHROPIC_MODELS = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
OPENAI_MODELS = ["gpt-4-0125-preview", "gpt-4o", "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-1106-preview",
                 "gpt-4-32k-0613", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"]


def get_selected_models(*model_lists: list[str]) -> list[str]:
    choices, defaults = [], []
    for model_list in model_lists:
        choices.extend(model_list)
        # If you pass an empty list in here, you're fired.
        defaults.append(model_list[0])
    return st.multiselect("Models", choices, default=defaults)


with st.sidebar:
    models = get_selected_models(ANTHROPIC_MODELS, OPENAI_MODELS)


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
        if model.startswith("gpt"):
            return OpenAIEngine(model)
        elif model.startswith("claude"):
            return AnthropicEngine(model)
        elif model.startswith("mistral") or model.startswith("codestral"):
            return MistralEngine(model)
        raise NotImplementedError(model)

    @abstractmethod
    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        pass


class OpenAIEngine(ChatCompletionEngine):
    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        c = AsyncOpenAI()
        content = ""
        async with (await c.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[dataclasses.asdict(m) for m in messages],
                stream=True,
        )) as stream:
            async for chunk in stream:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk:
                    content += content_chunk
                mprintm(Message(role="assistant", content=content), self.model, out=out)
        custom_logger(f"Got completion {content} from OpenAIEngine.")
        return Message(role="assistant", content=content)


class AnthropicEngine(ChatCompletionEngine):

    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        c = anthropic.AsyncAnthropic()
        messages_ = [dataclasses.asdict(m) for m in messages if m.role != "system"]
        system = next(iter(m.content for m in messages if m.role == "system"), None)
        content = ""
        async with (await c.messages.create(
                max_tokens=1024,
                messages=messages_,
                model=self.model,
                system=system,
                stream=True,
        )) as stream:
            async for chunk in stream:
                if isinstance(chunk, ContentBlockDeltaEvent):
                    content += chunk.delta.text
                mprintm(Message(role="assistant", content=content), self.model, out=out)
        custom_logger(f"Got completion {content} from AnthropicEngine.")
        return Message(role="assistant", content=content)


class MistralEngine(ChatCompletionEngine):
    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        from mistralai.async_client import MistralAsyncClient
        c = MistralAsyncClient()
        content = ""
        async for chunk in c.chat_stream(
                model=self.model,
                messages=[dataclasses.asdict(m) for m in messages],
        ):
            content += chunk.choices[0].delta.content
            mprintm(Message(role="assistant", content=content), self.model, out=out)
        custom_logger(f"Got completion {content} for model {self.model} from MistralEngine.")
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


def mprintm(m: Message, model: str, out: DeltaGenerator | None = None):
    if out is None:
        out = st.empty()
    with out.container():
        st.markdown(f"<h3 style='text-align: center;'>{model}</h3>", unsafe_allow_html=True)
        printm(m)


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


def get_placeholders(n: int):
    return [col.empty() for col in st.columns(n)]


placeholders = None

if prompt_msg := chat_input():
    st.session_state.messages.append(prompt_msg)
    printm(prompt_msg)
    placeholders = get_placeholders(len(models))
    st.session_state.completions = run_gather(
        *(ChatCompletionEngine.for_model(model).complete(st.session_state.messages, out=ph) for model, ph in
          zip(models, placeholders)))
if any(x is not None for x in st.session_state.completions):
    if placeholders is None:
        placeholders = get_placeholders(len(models))
    for model_, completion_, col_ in zip(models, st.session_state.completions, placeholders):
        mprintm(completion_, model_, out=col_)
    if user_selection := get_user_selection():
        st.session_state.messages.append(
            next(cmp for cmp, model in zip(st.session_state.completions, models) if model == user_selection)
        )
        st.session_state.completions = ()
        st.rerun()
