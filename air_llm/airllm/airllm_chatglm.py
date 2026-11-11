from .airllm_base import AirLLMBaseModel


class AirLLMChatGLM(AirLLMBaseModel):
    """ChatGLM2/3 (``ChatGLMModel``, remote code): decoder under ``transformer.encoder``.

    The checkpoint also carries ``transformer.rotary_pos_emb.inv_freq``; it is not streamed
    because the model computes the same buffer in fp32 when it is built.
    """

    LAYER_NAMES = {'embed': 'transformer.embedding.word_embeddings',
                   'layer_prefix': 'transformer.encoder.layers',
                   'norm': 'transformer.encoder.final_layernorm',
                   'lm_head': 'transformer.output_layer'}
