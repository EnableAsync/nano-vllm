from nanovllm.engine.llm_engine import LLMEngine


# LLM 是 LLMEngine 的别名，保持与 vLLM 的 API 接口一致
class LLM(LLMEngine):
    pass
