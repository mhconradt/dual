import asyncio
import dataclasses
import inspect
import sys
import tempfile
from abc import ABC, abstractmethod
from asyncio import Future
from datetime import datetime
from typing import Literal, Callable, Any

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


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    # OpenAPI schema for the function. Should include types, required parameters, etc. based on annotations and defaults
    schema: dict
    fn: Callable[[Any, ...], Future[str]]

    @classmethod
    def from_function(cls, fn: Callable) -> "Tool":
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


async def exec_subprocess(program, *args):
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = await asyncio.create_subprocess_exec(
            program,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmpdir,
        )
        stdout, stderr = await proc.communicate()
        return stdout, stderr, proc.returncode


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


def get_program_output(exit_code, stderr, stdout):
    parts = []
    if stdout:
        parts.extend(['## stdout', stdout])
    if stderr:
        parts.extend(['## stderr', stderr])
    if exit_code:
        parts.extend([f"Process exited with code {exit_code}."])
    return '\n'.join(parts)


async def compile_cxx_code(code: str, *, outfile: str | None = None) -> tuple[str, str, int]:
    """
    Compiles C++ code, returning stdout, stderr, and exit code.
    """
    with tempfile.NamedTemporaryFile(mode='w') as f:
        # asyncio would be better for files.
        f.write(code)
        f.flush()
        args = [f.name]
        if outfile:
            args.append('')
        return await exec_subprocess('g++', *args)


async def cxx_compile(code: str) -> str:
    """
    Compiles C++ code and returns the compiler output, errors, and exit code.
    Useful for iteratively debugging compilation errors.
    """
    stdout, stderr, exit_code = await compile_cxx_code(code)
    return get_program_output(exit_code, stderr, stdout)


async def cxx_run(code: str) -> str:
    """
    Compiles and runs C++ code.
    If compilation failed, i.e. exit code != 0, returns the compiler output, errors, and exit code.
    Otherwise, returns the output, errors, and exit code of the program.
    """
    with tempfile.NamedTemporaryFile(mode='r') as f:
        stdout, stderr, exit_code = await compile_cxx_code(code)
        if exit_code:
            return get_program_output(exit_code, stderr, stdout)
        return ...


async def run_shell_command(script: str) -> str:
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
    """
    stdout, stderr, code = await shell(script)
    return get_program_output(stdout, stderr, code)


TOOLS = {
    "C++ Compiler": Tool.from_function(cxx_compile)
}


class OpenAIEncoder:
    def tool(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        }


class AnthropicEncoder:
    def tool(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.schema,
        }


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


class OpenAIEngine(ChatCompletionEngine):
    async def complete(self, messages: list[Message], out: DeltaGenerator | None = None) -> Message:
        c = AsyncOpenAI()
        content = ""
        async with (await c.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[dataclasses.asdict(m) for m in messages],
            stream=True,
            tools=[OpenAIEncoder().tool(tool) for tool in self.tools]
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
            max_tokens=4096,
            messages=messages_,
            model=self.model,
            system=system,
            stream=True,
            tools=[AnthropicEncoder().tool(tool) for tool in self.tools],
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
    tools_raw = st.multiselect('Tools', options=list(TOOLS), default=list(TOOLS))
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
