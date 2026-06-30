# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI

__all__ = ["compile_llm", "create_llm", "recovery_llm", "task_llm"]

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------
# os.environ["ALL_PROXY"] = ""
# os.environ["all_proxy"] = ""
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model="gpt-4o"):
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_URL"),
    )


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------


# Initialize LLM instances, but handle errors gracefully for documentation builds
def _create_llm_safe(*, temperature=0.0, model="gpt-4o"):
    try:
        return create_llm(temperature=temperature, model=model)
    except Exception:
        return None


task_llm = _create_llm_safe(temperature=0.0, model="gpt-5")
recovery_llm = _create_llm_safe(temperature=0.0, model="gpt-5")
compile_llm = _create_llm_safe(temperature=0.0, model="gpt-5")


def _health_check() -> None:
    """Run a minimal LLM connectivity check without printing credentials."""
    missing_env = [
        env_name for env_name in ("OPENAI_API_KEY", "LLM_URL") if not os.getenv(env_name)
    ]
    if missing_env:
        missing = ", ".join(missing_env)
        raise SystemExit(f"Missing required environment variable(s): {missing}")

    model = os.getenv("AGENTCHORD_LLM_HEALTHCHECK_MODEL", "gpt-5")
    llm = create_llm(temperature=0.0, model=model)
    response = llm.invoke("Reply with exactly: AgentChord LLM health check OK")
    content = getattr(response, "content", str(response)).strip()

    print(f"LLM health check succeeded with model={model}.")
    print(content)


if __name__ == "__main__":
    _health_check()
