import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks.mmlu.mmlu import MMLU
from deepeval.benchmarks.hellaswag.hellaswag import HellaSwag
from deepeval.benchmarks.big_bench_hard.big_bench_hard import BigBenchHard
from deepeval.benchmarks.drop.drop import DROP
from deepeval.benchmarks.truthful_qa.truthful_qa import TruthfulQA
from deepeval.benchmarks.human_eval.human_eval import HumanEval
from deepeval.benchmarks.gsm8k.gsm8k import GSM8K
import config as ctf

class LLM(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        # model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model.config.name_or_path

model_name = ctf.model_name
peft_name = ctf.peft_name
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    # quantization_config=bnb_config #Enable 4 bit quantization
    # load_in_8bit=True, #Enable 8 bit quantization
    torch_dtype=torch.bfloat16, #Diable is 4 or 8 bit quantization is enabled
    attn_implementation="flash_attention_2", #Flass Attention 2 for faster Inference
    max_length = ctf.GENERATION_LENGTH
)


tokenizer = AutoTokenizer.from_pretrained(model_name)

if ctf.peft_name:
    model = PeftModel.from_pretrained(
        model, 
        peft_name, 
        device_map="auto"
    )
model.eval()

evalModel = LLM(model=model, tokenizer=tokenizer)
benchmark = MMLU()
results = benchmark.evaluate(model=evalModel)
print('Score:: ',results)