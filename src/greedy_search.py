import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def greedy_search_generation(model, tokenizer, input_txt, n_steps=8):
    iterations = []
    choices_per_step = 5

    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        for _ in range(n_steps):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(input_ids[0])
            output = model(input_ids=input_ids)
            # Seleccionamos los logits del primer batch y el último token y aplicamos softmax
            next_token_logits = output.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
            # Guardamos los tokens con mayor probabilidad :>
            for choice_idx in range(choices_per_step):
                token_id = sorted_ids[choice_idx]
                token_prob = next_token_probs[token_id].cpu().numpy()
                token_choice = (
                    f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
                )
                iteration[f"Choice {choice_idx+1}"] = token_choice
            # Concatenamos el token predicho a nuestro input para la siguiente iteración
            input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
            iterations.append(iteration)
    return iterations
