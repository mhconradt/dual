import asyncio
import dataclasses
import inspect
import json
import sys
import tempfile
from abc import ABC, abstractmethod
from asyncio import Future
from collections import defaultdict
from datetime import datetime
from typing import Literal, Callable, Any

import anthropic
import streamlit as st
from anthropic.types import ContentBlockDeltaEvent, RawContentBlockStartEvent, TextDelta, ToolUseBlock, InputJsonDelta, \
    RawMessageDeltaEvent
from anthropic.types.raw_content_block_delta_event import Delta
from openai import AsyncOpenAI
from streamlit.delta_generator import DeltaGenerator

TOOL_ALIASES = {'run_shell_script': 'Shell'}

st.set_page_config(layout="wide")
st.title("Dual")

ANTHROPIC_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
OPENAI_MODELS = ["gpt-4-0125-preview", "gpt-4o", "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-1106-preview",
                 "gpt-4-32k-0613", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"]


def get_selected_models(*model_lists: list[str]) -> list[str]:
    choices, defaults = [], []
    for model_list in model_lists:
        choices.extend(model_list)
        # If you pass an empty list in here, you're fired.
        defaults.append(model_list[0])
    return st.multiselect("Models", choices, default=defaults)


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
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: "list[ToolCall] | None" = None


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    # OpenAPI schema for the function. Should include types, required parameters, etc. based on annotations and defaults
    schema: dict
    fn: Callable[[Any, ...], Future[str]]

    @classmethod
    def from_function(cls, fn: Callable[[Any, ...], Future[str]]) -> "Tool":
        annotations = inspect.get_annotations(fn)

        try:
            annotations.pop('return')
        except KeyError:
            pass

        required = [k for k, v in inspect.signature(fn).parameters.items() if v.default is inspect.Parameter.empty]
        schema = {
            "type": "object",
            "properties": {
                k: {
                    "type": {str: "string", float: "number", int: "integer", bool: "boolean"}[v]
                }
                for k, v in annotations.items()
            },
            "required": required
        }

        return cls(
            name=fn.__name__,
            description=inspect.getdoc(fn) or "",
            schema=schema,
            fn=fn,
        )

    async def execute(self, **kwargs) -> str:
        return await self.fn(**kwargs)

    async def __call__(self, *args, **kwargs):
        return await self.fn(*args, **kwargs)


async def shell(command: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmpdir,
        )
        stdout, stderr = await proc.communicate()
        return stdout, stderr, proc.returncode


def quote(text: str, sep: str = '```') -> str:
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    return '\n'.join([sep, text, sep])

def get_program_output(stdout, stderr, exit_code) -> str:
    parts = []
    if stdout:
        parts.extend(['## stdout', quote(stdout)])
    if stderr:
        parts.extend(['## stderr', quote(stderr)])
    if exit_code:
        parts.extend([f"Process exited with code {exit_code}."])
    return '\n'.join(parts)


async def run_shell_script(script: str) -> str:
    """
Runs a shell program, returning its stdout, stderr, and exit code.
It's highly encouraged to use this to compile and run code.
1. This sequence of shell commands writes, compiles, and runs a C++ program:
cat <<EOF> scratch.cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
}
EOF
c++ scratch.cpp -std=c++20 -o ./scratch
./scratch
2. This program writes and runs a Python program:
cat <<EOF> scratch.py
if __name__ == '__main__':
    print('Hello world')
EOF
python3 ./scratch.py
3. If there are import errors using the default Python interpreter, you may need to install additional dependencies as a last resort, always doing so in a virtual environment:
python3 -m venv ./venv
source ./venv/bin/activate
pip install pandas
cat <<EOF> scratch.py
import numpy as np
import pandas as pd
if __name__ == '__main__':
    print(pd.DataFrame({'n': np.random.randn(100), 'x': np.arange(100)}).describe())
EOF
python3 ./scratch.py
deactivate
    """
    stdout, stderr, code = await shell(script)
    return get_program_output(stdout, stderr, code)


TOOLS = {
    "run_shell_script": Tool.from_function(run_shell_script),
}


class OpenAISerDe:
    def messages(self, messages: list[Message]) -> list[dict]:
        return [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": [self.encode_tool_call(tc) for tc in m.tool_calls] if m.tool_calls else None,
                "tool_call_id": m.tool_call_id
            }
            for m in messages
        ]

    def encode_tool_call(self, tool_call: "ToolCall") -> dict:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(tool_call.arguments)
            }
        }

    def encode_tool(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema,
            }
        }

    def decode_tool_call(self, tool_call: dict) -> "ToolCall":
        return ToolCall(id=tool_call['id'], name=tool_call['name'], arguments=json.loads(tool_call['arguments']))


class AnthropicSerDe:
    def messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

    def encode_tool(self, tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
        }

    def decode_tool_call(self, tool_call: dict) -> "ToolCall":
        return ToolCall(id=tool_call['id'], name=tool_call['name'],
        arguments=json.loads(tool_call['input']))


class ChatCompletionEngine(ABC):
    def __init__(self, model: str, tools: list[Tool]):
        self.model = model
        self.tools = tools

    @classmethod
    def for_model(cls, model: str, tools: list[Tool]) -> "ChatCompletionEngine":
        if model.startswith("gpt"):
            return OpenAIEngine(model, tools)
        elif model.startswith("claude"):
            return AnthropicEngine(model, tools)
        elif model.startswith("mistral") or model.startswith("codestral"):
            return MistralEngine(model, tools)
        raise NotImplementedError(model)

    @abstractmethod
    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        pass


@dataclasses.dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

    async def execute(self) -> str:
        return await TOOLS[self.name](**self.arguments)


class OpenAIEngine(ChatCompletionEngine):
    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        c = AsyncOpenAI()
        messages = list(messages)
        content = ""
        serde = OpenAISerDe()
        tool_choices = [serde.encode_tool(tool) for tool in self.tools]
        while not content:
            messages_ = serde.messages(messages)
            async with (await c.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=messages_,
                stream=True,
                tools=tool_choices
            )) as stream:
                # We need to go deeper.
                tool_calls = defaultdict(lambda: defaultdict(str))
                async for chunk in stream:
                    custom_logger(chunk)
                    delta = chunk.choices[0].delta
                    if not delta:
                        continue
                    if delta and delta.content:
                        content_chunk = delta.content
                        if content_chunk:
                            content += content_chunk
                        mprintm(Message(role="assistant", content=content), self.model, out=out)
                    elif delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            index = tool_call.index
                            if tool_call.id:
                                tool_calls[index]['id'] = tool_call.id
                            if tool_call.function:
                                if tool_call.function.name:
                                    tool_calls[index]['name'] = tool_call.function.name
                                if tool_call.function.arguments:
                                    custom_logger(f"Got tool fragment {tool_call.function.arguments} from OpenAIEngine")
                                    tool_calls[index]['arguments'] += tool_call.function.arguments
                if tool_calls:
                    custom_logger(list(tool_calls.values()))
                    custom_logger(f"Intermediate completion {content}")
                    parsed_calls: list[ToolCall] = [serde.decode_tool_call(tool_call) for tool_call in
                                                    tool_calls.values()]
                    outputs = await asyncio.gather(*[call.execute() for call in parsed_calls])
                    messages.append(Message(role="assistant", content=content, tool_calls=parsed_calls))
                    messages.extend([
                        Message(tool_call_id=call.id, name=call.name, role="tool", content=output)
                        for call, output in zip(parsed_calls, outputs)
                    ])

        custom_logger(f"Got completion {content} from OpenAIEngine.")
        return Message(role="assistant", content=content)


class AnthropicEngine(ChatCompletionEngine):

    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        c = anthropic.AsyncAnthropic()
        serde = AnthropicSerDe()
        system = next(iter(m.content for m in messages if m.role == "system"), None)
        stop_reason = None
        messages = list(messages)
        messages_ = serde.messages(messages)
        content = ""
        while stop_reason != "end_turn" or not content:
            content = ''
            custom_logger(messages_)
            async with (await c.messages.create(
                messages=messages_,
                max_tokens=4096,
                model=self.model,
                system=system,
                stream=True,
                tools=[serde.encode_tool(tool) for tool in self.tools],
            )) as stream:
                tool_calls = defaultdict(lambda: defaultdict(str))
                k = None
                async for chunk in stream:
                    if isinstance(chunk, RawContentBlockStartEvent) and isinstance(chunk.content_block, ToolUseBlock):
                        k = chunk.content_block.id
                        tool_calls[k]['id'] = k
                        tool_calls[k]['name'] = chunk.content_block.name
                    custom_logger(chunk)
                    if isinstance(chunk, ContentBlockDeltaEvent) and isinstance(chunk.delta, TextDelta):
                        content += chunk.delta.text
                    if isinstance(chunk, ContentBlockDeltaEvent) and isinstance(chunk.delta, InputJsonDelta):
                        tool_calls[k]['input'] += chunk.delta.partial_json
                    if isinstance(chunk, RawMessageDeltaEvent):
                        stop_reason = chunk.delta.stop_reason
                    mprintm(Message(role="assistant", content=content), self.model, out=out)
                if tool_calls:
                    all_calls = list(tool_calls.values())
                    custom_logger(all_calls)
                    parsed_calls = [serde.decode_tool_call(call) for call in all_calls]
                    content_ = [{"type": "text", "text": content}]
                    content_.extend([{"type": "tool_use", "id": call.id,
                    "name": call.name, "input": call.arguments} for call in
                    parsed_calls])
                    messages_.append({"role": "assistant", "content":
                    content_})
                    outputs = await asyncio.gather(*[call.execute() for call in parsed_calls])
                    custom_logger(outputs)
                    content_ = [{"type": "tool_result", "content": output,
                    "tool_use_id": call.id} for output, call in zip(outputs,
                    parsed_calls)]
                    messages_.append({"role": "user", "content": content_})
                else:
                    break

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
    user_prompt = st.chat_input("Type a message...", disabled=bool(st.session_state.completions))
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


def get_selected_tools() -> list[Tool]:
    tools_raw = st.multiselect(
        'Tools',
        options=list(TOOLS),
        default=list(TOOLS),
        format_func=lambda k: TOOL_ALIASES.get(k, k)
    )
    return [TOOLS[k] for k in tools_raw]


placeholders = None


def get_current_system_prompt() -> str:
    return st.session_state.messages[0].content


with st.sidebar:
    models = get_selected_models(ANTHROPIC_MODELS, OPENAI_MODELS)
    tools = get_selected_tools()
    system_prompt = st.text_area('System Prompt', get_current_system_prompt())

if prompt_msg := chat_input():
    st.session_state.messages.append(prompt_msg)
    printm(prompt_msg)
    placeholders = get_placeholders(len(models))
    st.session_state.completions = run_gather(
        *(
            ChatCompletionEngine.for_model(model, tools).complete(st.session_state.messages, out=ph)
            for model, ph in zip(models, placeholders)
        )
    )
if st.session_state.completions:
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
