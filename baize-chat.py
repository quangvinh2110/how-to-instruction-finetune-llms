import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="7"

def sample_decode(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    stop_words: list,
    max_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 25,
):
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        # if top_k is not None:
        #    probs_sort1, _ = torch.topk(probs_sort, top_k)
        #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
        #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)
        yield text
        if any([x in text for x in stop_words]):
            return

def stream_answer(prompt, bot_name, user_name, max_new_tokens=256):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    text_generator = sample_decode(
        input_ids,
        model,
        tokenizer,
        stop_words=["[|Human|]"],
        max_length=max_new_tokens,
        temperature= 0.2,
        top_p=1.0,
        top_k=25,
    )
    next_text = ""
    current_text = ""
    sys.stdout.flush()
    for text in text_generator:
        next_text = text[len(current_text):]
        current_text = text
        print(next_text, end="")
        sys.stdout.flush()
    return text 

def chat():
    user_name, bot_name = "[|Human|]","[|AI|]"
    system_prompt = f"""The conversation between human and AI assistant.
[|Human|] Hi, my name is Jeff.
[|AI|] Hi Jeff, I'm a assistant chatbot. How can I help you?
[|Human|]
"""
    history = system_prompt.strip()
    # print(f"{history.strip()}")
    print(user_name)
    while True:
        query = input()
        history += f"{query}\n{bot_name}"
        print(bot_name)
        answer = stream_answer(history, bot_name=bot_name, user_name=user_name)
        answer = answer.strip(user_name)
        history += f"{bot_name}{answer}"

load_in_8bit = False
# BASE_MODEL = "VietAI/gpt-neo-1.3B-vietnamese-news"
BASE_MODEL = "../llm/models/bloom-7b1/"
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=load_in_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# PEFT_WEIGHTS = "../llm/chat-bloom-7b/adapter"
PEFT_WEIGHTS = "../baize-chatbot-main/ckpts/baize/pt_lora_model/"
model = PeftModel.from_pretrained(model, PEFT_WEIGHTS)

chat()
