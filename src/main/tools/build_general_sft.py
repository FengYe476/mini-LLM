import argparse
import json
import random

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description = 'convert a HuggingFace instruction dataset into the SFT jsonl format')
    parser.add_argument('--repo', default = 'HuggingFaceH4/ultrachat_200k')
    parser.add_argument('--split', default = 'train_sft')
    parser.add_argument('--field', default = 'messages')
    parser.add_argument('--max-samples', type = int, default = 100_000)
    parser.add_argument('--max-tokens', type = int, default = 1024)
    parser.add_argument('--tok', default = 'data/tok.json')
    parser.add_argument('--out', default = 'data/sft_general.jsonl')
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tokenizer import Tokenizer
    from datasets import load_dataset

    tokenizer = Tokenizer.load(args.tok)
    ds = load_dataset(args.repo, split = args.split, streaming = True)

    kept, skipped_long, skipped_shape = [], 0, 0
    for row in ds:
        messages = row.get(args.field)
        if not isinstance(messages, list) or len(messages) < 2:
            skipped_shape += 1
            continue
        clean = [{'role': m.get('role'), 'content': m.get('content')} for m in messages
                 if m.get('role') in ('system', 'user', 'assistant')]
        if not any(m['role'] == 'assistant' and m['content'] for m in clean):
            skipped_shape += 1
            continue
        try:
            ids, mask = tokenizer.render_conversation(clean, 10 ** 9)
        except ValueError:
            skipped_shape += 1
            continue
        if len(ids) > args.max_tokens:
            skipped_long += 1
            continue
        kept.append(clean)
        if len(kept) >= args.max_samples:
            break

    random.Random(args.seed).shuffle(kept)
    print(f'[general sft]: kept {len(kept)}, dropped {skipped_long} over {args.max_tokens} tokens, {skipped_shape} malformed')

    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    temp = out.with_name(out.name + '.tmp')
    with open(temp, 'w', encoding = 'utf-8') as f:
        for c in kept:
            f.write(json.dumps({'messages': c}, ensure_ascii = False) + '\n')
    temp.replace(out)
    print(f'[general sft]: wrote {out} ({out.stat().st_size / 1e6:.0f} MB)')
    return


if __name__ == '__main__':
    main()
