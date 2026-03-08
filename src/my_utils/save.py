import torch
def save_lora_weights(model, save_path):
    lora_state_dict = {
        k: v.cpu() for k, v in model.state_dict().items()
        if 'lora' in k.lower()
    }
    torch.save(lora_state_dict, save_path)

def load_lora_weights(model, lora_state_dict):
    model_state_dict = model.state_dict()
    # Only update matching LoRA keys
    matched_keys = []
    for k in lora_state_dict.keys():
        if k in model_state_dict:
            model_state_dict[k].copy_(lora_state_dict[k])
            matched_keys.append(k)
    print(f"Loaded {len(matched_keys)} LoRA weights.")