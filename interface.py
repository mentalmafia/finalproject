import os
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

llama_path = "luodian/llama-7b-hf"
checkpoint_path = "/home/grads/dps6276/HaELM-master/checkpoint"
data_path = "/home/grads/dps6276/HaELM-master/LLM_output/mPLUG_caption.jsonl"  # Path to your new dataset

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(llama_path)

model = LlamaForCausalLM.from_pretrained(
    llama_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    checkpoint_path,
    force_download=True,
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

# Load data
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Generation configuration
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
)

# Store ground truth and predictions
ground_truth = []
predictions = []

print(model.base_model.model.model.embed_tokens.weight.device)
model = model.to(torch.float16)

# Evaluate the model
for d in data:
    prompt = d["instruction"]
    expected_output = d["output"].strip().lower()  # Ground truth label (yes/no)
    ground_truth.append(expected_output)

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.base_model.model.model.embed_tokens.weight.device).to(torch.int64)
    
    # Generate the model's response
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1,
        )

    # Decode the response
    sentence = generation_output.sequences
    result = tokenizer.decode(sentence.tolist()[0], skip_special_tokens=True)
    predicted_output = result.split("\n")[-1].strip().lower()  # Model's prediction (yes/no)
    predictions.append(predicted_output)
    print(predictions)

# Compute metrics
accuracy = accuracy_score(ground_truth, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, predictions, average="binary", pos_label="yes")

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
