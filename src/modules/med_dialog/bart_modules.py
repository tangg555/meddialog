from transformers.models.bart.modeling_bart import (BartForConditionalGeneration, BartConfig, nn)
from transformers.models.t5.modeling_t5 import (T5ForConditionalGeneration,T5Config)

class BartWithTermsForCG(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.encoder_fc = nn.Linear(config.d_model, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

class T5WithTermsForCG(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.encoder_fc = nn.Linear(config.d_model, 2, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
