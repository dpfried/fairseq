import os
import sys
import json
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel

from typing import List

from tokenizers import ByteLevelBPETokenizer

#root_dir="/checkpoint/dpf/models/cm-1.3B-ourtok-lr8e-4/"
root_dir="/checkpoint/dpf/models/cm-6B-armen/cm-6B-ourtok/"
ours = TransformerLanguageModel.from_pretrained(root_dir, "best.pt", bpe="gpt2_pretokenization_newlines_only", gpt2_encoder_json=f"{root_dir}/vocab.json", gpt2_vocab_bpe=f"{root_dir}/merges.txt").cuda().eval()
is_cm = True

EOSS = "<eoss>"

def make_sentinel(i):
    return f"<sentinel:{i}>"

tokenizer = ByteLevelBPETokenizer.from_file(
    os.path.join(root_dir, "vocab.json"),
    os.path.join(root_dir, "merges.txt"),
    pretokenizer_split_newlines_only=True
)

if is_cm:
    special_tokens = []
    for i in range(256):
        special_tokens.append(make_sentinel(i))
    special_tokens.append(EOSS)
    tokenizer.add_special_tokens(special_tokens)

# set the max generation length
ours.cfg.generation['max_len_b'] = 500

TOKENIZER_OFFSET = 4

EOSS_ID = tokenizer.token_to_id(EOSS) + TOKENIZER_OFFSET

def encode(s):
    return torch.tensor(tokenizer.encode(s).ids) + TOKENIZER_OFFSET

def decode(token_ids):
    token_ids = torch.tensor(token_ids)
    return tokenizer.decode((token_ids - TOKENIZER_OFFSET).tolist(), skip_special_tokens=False)

def sentinel_id(i):
    return tokenizer.token_to_id(make_sentinel(i)) + TOKENIZER_OFFSET

def complete(s, model=ours, **kwargs):
    """ complete the prefix s autoregressively """
    with torch.no_grad():
        #encoded = model.encode(s)
        encoded = torch.tensor(tokenizer.encode(s).ids) + 4
        #encoded = encoded.cuda()
        completion = model.generate([encoded], **kwargs)[0][0]['tokens']
        completion = (completion - 4)[:-1]
        return tokenizer.decode(completion.cpu().tolist(), skip_special_tokens=False)
        #return model.decode(completion)

def infill(parts: List[str], model=ours, verbose=False, **kwargs):
    # Force the model to fill in code in between each string in parts
    # see code_to_docstring and docstring_to_code for example usages
    assert isinstance(parts, list)
    infills = []
    if len(parts) == 1:
        return complete(parts[0])[len(parts[0]):]

    ids = []

    # encode parts separated by sentinel
    for sentinel_ix, part in enumerate(parts):
        part_tokens = encode(part)
        ids.extend(part_tokens.tolist())
        if sentinel_ix < len(parts) - 1:
            ids.append(sentinel_id(sentinel_ix))

    infills = []

    complete = []

    # autoregressively fill in
    for sentinel_ix, part in enumerate(parts[:-1]):
        ids.append(sentinel_id(sentinel_ix))
        if verbose:
            print(part, end="")
            print(f"<sentinel:{sentinel_ix}>", end="")
        with torch.no_grad():
            completion = model.generate([torch.tensor(ids)], **kwargs)[0][0]['tokens'].tolist()
            if completion[-1] == 2:
                completion = completion[:-1]


        completion = completion[len(ids):]

        if EOSS_ID in completion:
            completion = completion[:completion.index(EOSS_ID)+1]
        else:
            if not verbose:
                print(f"warning: {EOSS} not found", file=sys.stderr)
            completion = completion + [EOSS_ID]

        ids.extend(completion)

        decoded = decode(completion[:-1])
        complete.append(part)
        complete.append(decoded)
        infills.append(decoded)

    complete.append(parts[-1])

    if verbose:
        print(parts[-1])
        print("-"*20)
        print(''.join((complete)))
    return {
        'complete': complete,
        'infills': infills,
        'ids': ids,
        'raw': decode(ids)
    }

def code_to_docstring(**kwargs):
    header = '''def count_words(filename):
    "'''

    body = '''"
    counts = Counter()
    with open(filename) as file:
        for line in file:
            words = line.split(' ')
            counts.update(words)
    return counts\n<|/ file |>'''
    return infill([header, body], **kwargs)

def docstring_to_code(**kwargs):
    return infill(['def ', '    """Count the number of occurrences of each word in the file."""\n', '<|/ file |>'], **kwargs)

if __name__ == "__main__":
    _ = code_to_docstring(verbose=True, sampling=True, sampling_topp=0.6, temperature=0.6)

