from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

device = torch.device("cuda:0")
model_info = "/staff/liqi/Aquila"
tokenizer = AutoTokenizer.from_pretrained(model_info, trust_remote_code=True)
quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
model = AutoModelForCausalLM.from_pretrained(model_info, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                # quantization_config=quantization_config, # Uncomment this line for 4bit quantization
                                                )
model.eval()
model.to(device)
text = "请给出10个要到北京旅游的理由。"
from predict import predict
out = predict(model, text, tokenizer=tokenizer, max_gen_len=200, top_p=0.9,
              seed=123, topk=15, temperature=1.0, sft=True, device=device,
              model_name="AquilaChat2-34B")
print(out)