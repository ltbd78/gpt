import torch


def generate_batch(model, encode_fn, decode_fn, initial_texts, n_tokens, print_batch_num=0): # TODO: print batches simultaneously
    model.eval()
    if print_batch_num is not None:
        print(initial_texts[print_batch_num], end='')

    encoded_texts = []
    for text in initial_texts:
        encoded_text = encode_fn(text)
        tensor = torch.tensor(encoded_text, dtype=torch.int64)
        tensor = torch.nn.functional.pad(tensor, (model.sequence_dim-len(encoded_text), 0))
        encoded_texts.append(tensor)
    x = torch.stack(encoded_texts, dim=0).to(model.device)
    for i in range(n_tokens):
        x_cropped = x[:, -model.sequence_dim:] # crop s.t. it's <= sequence_dim
        logits = model(x_cropped) # (N, L, E)
        logits = logits[:, -1, :] # (N, E)
        probs = torch.nn.functional.softmax(logits, dim=-1) # (N, E)
        y_pred = torch.multinomial(probs, num_samples=1) # (N, 1)
        x = torch.cat((x, y_pred), dim=1) # (N, L) + (N, 1) = (N, L + 1)

        if print_batch_num is not None:
            next_token = decode_fn(y_pred[print_batch_num].cpu().numpy())
            print(next_token, end='')
    return x