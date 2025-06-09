from abc import abstractmethod
from asyncio import create_task, get_running_loop, sleep, Future, Queue, Task
from dataclasses import dataclass
import math
import re
import time
from typing import List, Optional, Tuple

import aiohttp

import torch

from transformers import AutoModelForCausalLM

from ReES.aggregates.language import Tokens
from ReES.env import OPENAI_API_KEY, OPENROUTER_API_KEY
from ReES.tokenizer import HuggingFaceTokenizer, Tokenizer


@dataclass
class ChatResponse:
    message: str
    time: float
    tokens: Tokens


async def openai_compliant_instruction(
    end_point: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    authorization: Optional[str] = None,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    body = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
    }

    headers = {
        "Content-Type": "application/json",
    }

    if authorization is not None:
        headers["Authorization"] = authorization

    exceptions: List[Exception] = []

    while len(exceptions) < 3:
        try:
            start = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.post(url=end_point, json=body, headers=headers) as request:
                    response = await request.json()
                    input_tokens = response["usage"]["prompt_tokens"]
                    output_tokens = response["usage"]["completion_tokens"]
                    content = response["choices"][0]["message"]["content"]
                    message = re.sub(r"<think>(.|\n)*?<\/think>\n\n", "", content)

                    end = time.time()

                    return ChatResponse(
                        message=message,
                        time=end - start,
                        tokens=Tokens(input=input_tokens, output=output_tokens),
                    )

        except Exception as e:
            exceptions.append(e)

    raise ExceptionGroup(f"Failed to retrieve message from API {len(exceptions)} times...", exceptions)


class LanguageModel:

    @abstractmethod
    async def instruct(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        pass


class HuggingFaceLanguageModel(LanguageModel):

    def __init__(self, model_name: str):
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        self._tokenizer = HuggingFaceTokenizer(model_name)

    async def instruct(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        messages = [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt }
        ]

        start = time.time()

        text = self._tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            temperature=0.0,
        )

        model_inputs = self._tokenizer.tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        message = self._tokenizer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        end = time.time()

        return ChatResponse(
            message=message,
            time=end-start,
            tokens=Tokens(
                input=len(text),
                output=self._tokenizer.len(message)
            ),
        )


class LocalLanguageModel(LanguageModel):

    end_point = "http://0.0.0.0:8000/v1/chat/completions"

    def __init__(self, model_name: str):
        self._model_name = model_name

    async def instruct(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        return await openai_compliant_instruction(
            LocalLanguageModel.end_point,
            self._model_name,
            system_prompt,
            user_prompt,
        )


class RateLimiter(LanguageModel):

    @dataclass
    class _LanguageEntry:

        system_prompt: str
        user_prompt: str

    def __init__(self, model: LanguageModel, rpm: int, tpm: int):
        self._model = model
        self._rpm = rpm
        self._tpm = tpm

        self._queue: Queue[Tuple[RateLimiter._LanguageEntry, Future]] = Queue()

        self._task: Optional[Task] = None

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    async def _poll_queue(self):
        def _task_callback(task: Task, future: Future):
            try:
                result = task.result()
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        tpm_buffer = self._tpm

        while True:
            entry, future = await self._queue.get()

            estimated_token_usage = min(
                math.ceil((len(entry.system_prompt) + len(entry.user_prompt)) * 15 / 2), # 0.75 * 10
                self._tpm, # allow any requests if buffer is full
            )

            if tpm_buffer < estimated_token_usage:
                await sleep(60 * estimated_token_usage / self._tpm)
                tpm_buffer = estimated_token_usage

            tpm_buffer -= estimated_token_usage

            instruction_task = create_task(self._model.instruct(entry.system_prompt, entry.user_prompt))
            instruction_task.add_done_callback(lambda task, f=future: _task_callback(task, f))

            sleep_duration = 60 / self._rpm
            await sleep(sleep_duration)

            tpm_buffer = min(tpm_buffer + math.floor(self._tpm / 60 * sleep_duration), self._tpm)

    async def instruct(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        if self._task is None:
            self._task = create_task(self._poll_queue())

        entry = RateLimiter._LanguageEntry(system_prompt, user_prompt)

        loop = get_running_loop()
        future = loop.create_future()

        await self._queue.put((entry, future))

        return await future


class OpenAILanguageModel(LanguageModel):

    end_point = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._tokenizer = Tokenizer.from_tiktoken_encoder(model_name)

    async def instruct(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        return await openai_compliant_instruction(
            OpenAILanguageModel.end_point,
            self._model_name,
            system_prompt,
            user_prompt,
            f"Bearer {OPENAI_API_KEY}",
        )


class OpenRouterModel(LanguageModel):

    end_point = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_name: str):
        self._model_name = model_name
        _model_type = self._model_name.split(":")[0]
        self._tokenizer = Tokenizer.from_huggingface_tokenizer(_model_type)

    async def instruct(self, system_prompt: str, user_prompt: str) -> ChatResponse:
        return await openai_compliant_instruction(
            OpenRouterModel.end_point,
            self._model_name,
            system_prompt,
            user_prompt,
            f"Bearer {OPENROUTER_API_KEY}",
        )
