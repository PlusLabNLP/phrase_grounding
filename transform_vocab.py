from transformers import AutoTokenizer
import torch
lines=open('object_vocab.txt').readlines()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vs=tokenizer.vocab_size
sv = torch.zeros(len(lines), vs)
for i, l in enumerate(lines):
    v=(tokenizer(l)['input_ids'][1:-1])
    for vv in v:
        sv[i][vv] = 1

torch.save(sv, 'object_vocab_bertbaseuncased.pt')
