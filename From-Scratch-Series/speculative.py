import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ... [Your autoregressive_generate function goes here] ...
def autoregressive_generate(model, tokenizer, prompt, max_new_tokens):
    print("Running Baseline (Standard Autoregressive)...")
    device = model.device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    original_input_len = input_ids.shape[-1] # Save this once

    start_time = time.time()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        next_token_logits = outputs.logits[:, -1, :]
        
        # (A) This is how you correctly get a (1, 1) tensor
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # Shape (1, 1)
        
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token_id)], 
            dim=-1
        )

    end_time = time.time()
    
    # Trim to exactly max_new_tokens
    final_input_ids = input_ids[0, :original_input_len + max_new_tokens]
    generated_text = tokenizer.decode(final_input_ids)
    
    # print(f"Generated text: {generated_text}")
    print("Results for k={}:".format(k))

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print("-" * 30)
    return generated_text, end_time - start_time


# ======================================================
#            Speculative (Task 5 - Final Fix)
# ======================================================
def speculative_generate(target_model, draft_model, tokenizer, prompt, max_new_tokens, k):
    """
    Generates text using speculative decoding with corrected tensor logic.
    """
    print(f"Running Speculative Decoding (k={k})...")
    device = target_model.device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    original_input_len = input_ids.shape[-1]
    generated_tokens_count = 0
    start_time = time.time()

    with torch.no_grad():
        while generated_tokens_count < max_new_tokens:
            
            # --- 1. Drafting Phase ---
            draft_input_ids = input_ids
            draft_attention_mask = attention_mask
            draft_tokens = [] # This will be a list of 0D scalar tensors

            for _ in range(k):
                draft_outputs = draft_model(input_ids=draft_input_ids, attention_mask=draft_attention_mask)
                next_token_logits = draft_outputs.logits[:, -1, :]
                
                # (B) Get 1D tensor (1,) and unsqueeze to 2D (1, 1)
                next_token_id_2D = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                # Store the 0D scalar
                draft_tokens.append(next_token_id_2D[0, 0]) # Get the 0D scalar

                draft_input_ids = torch.cat([draft_input_ids, next_token_id_2D], dim=-1)
                draft_attention_mask = torch.cat(
                    [draft_attention_mask, torch.ones_like(next_token_id_2D)],
                    dim=-1
                )
            
            # --- 2. Verification Phase ---
            draft_tokens_tensor = torch.stack(draft_tokens, dim=0).unsqueeze(0) # Shape (1, k)
            
            verification_input_ids = torch.cat([input_ids, draft_tokens_tensor], dim=-1)
            verification_attention_mask = torch.cat(
                [attention_mask, torch.ones_like(draft_tokens_tensor)],
                dim=-1
            )
            
            target_outputs = target_model(
                input_ids=verification_input_ids, 
                attention_mask=verification_attention_mask
            )
            target_logits = target_outputs.logits 
            
            # --- 3. Acceptance/Rejection Phase ---
            newly_generated_tokens = []
            current_input_len = input_ids.shape[-1]

            for i in range(k):
                target_token_logits = target_logits[:, current_input_len + i - 1, :]
                target_token_id = torch.argmax(target_token_logits, dim=-1)[0] # 0D scalar
                draft_token_id = draft_tokens[i] # 0D scalar
                
                if target_token_id == draft_token_id:
                    newly_generated_tokens.append(draft_token_id)
                else:
                    newly_generated_tokens.append(target_token_id)
                    break
            
            # --- 4. Append All New Tokens At Once ---
            if not newly_generated_tokens:
                outputs = target_model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                
                # (C) Correctly get 2D (1, 1) tensor
                new_tokens_tensor = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                new_tokens_tensor = torch.stack(newly_generated_tokens, dim=0).unsqueeze(0)

            n_new = new_tokens_tensor.shape[-1]
            
            # Handle "bonus token" case
            if n_new == k and newly_generated_tokens[-1] == draft_tokens[-1]:
                last_target_logits = target_logits[:, -1, :]
                
                # (C) Correctly get 2D (1, 1) tensor
                bonus_token_id = torch.argmax(last_target_logits, dim=-1).unsqueeze(-1) 
                new_tokens_tensor = torch.cat([new_tokens_tensor, bonus_token_id], dim=-1)
                n_new += 1

            # Update the main state ONCE
            input_ids = torch.cat([input_ids, new_tokens_tensor], dim=-1)
            new_attention = torch.ones_like(new_tokens_tensor)
            attention_mask = torch.cat([attention_mask, new_attention], dim=-1)
            
            generated_tokens_count += n_new
            
            if generated_tokens_count >= max_new_tokens:
                break

    end_time = time.time()
    
    final_input_ids = input_ids[0, :original_input_len + max_new_tokens]
    generated_text = tokenizer.decode(final_input_ids)
    
    # print(f"Generated text: {generated_text}")
    print("Results for k={}:".format(k))
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print("-" * 30)
    return generated_text, end_time - start_time

# ======================================================
#            MAIN EXECUTION (Unchanged)
# ======================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)
    draft_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    
    target_model.eval()
    draft_model.eval()
    print("Models loaded and in eval mode.")

    prompt = "The field of Artificial Intelligence has seen"
    max_new_tokens = 100
    K = [3,5,7,10]
    for k in K:
        autoregressive_generate(target_model, tokenizer, prompt, max_new_tokens)

        speculative_generate(target_model, draft_model, tokenizer, prompt, max_new_tokens, k=k)