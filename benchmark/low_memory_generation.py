import torch
import torch.nn.functional as F


def forward_last_logits(model, input_ids, attention_mask=None, past_key_values=None):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )

    last_hidden = outputs.last_hidden_state[:, -1:, :]
    if getattr(model.config, "pretraining_tp", 1) > 1:
        slices = model.lm_head.weight.split(
            model.vocab_size // model.config.pretraining_tp, dim=0
        )
        logits = torch.cat(
            [F.linear(last_hidden, slices[i]) for i in range(model.config.pretraining_tp)],
            dim=-1,
        )
    else:
        logits = model.lm_head(last_hidden)

    return logits.float(), outputs.past_key_values


def greedy_generate(model, inputs, max_new_tokens, progress_interval=None, progress_prefix="Generated"):
    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    past_key_values = None

    for token_idx in range(max_new_tokens):
        current_input_ids = input_ids if past_key_values is None else input_ids[:, -1:]
        logits, past_key_values = forward_last_logits(
            model,
            current_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        if progress_interval and (token_idx + 1) % progress_interval == 0:
            print(f"   {progress_prefix} {token_idx + 1}/{max_new_tokens} tokens...")

    return {
        "sequences": input_ids,
        "past_key_values": past_key_values,
        "attention_mask": attention_mask,
    }
