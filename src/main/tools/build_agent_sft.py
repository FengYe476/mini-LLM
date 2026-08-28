import argparse
import json
import random
import tarfile

from pathlib import Path

SHORT_SYSTEM = 'you are yuki, a coding agent. you work by calling tools and reading their output.'


def iter_trajectories(archives: list[Path]) -> tuple[list, int, int]:
    kept, seen, failed = [], 0, 0
    for archive in archives:
        with tarfile.open(archive, 'r:gz') as tar:
            sessions, rewards = {}, {}
            for member in tar:
                if member.name.endswith('agent/yuki_session.json'):
                    sessions[member.name.rsplit('/agent/', 1)[0]] = member
                elif member.name.endswith('verifier/reward.txt'):
                    rewards[member.name.rsplit('/verifier/', 1)[0]] = member
            for trial, member in sessions.items():
                seen += 1
                reward_member = rewards.get(trial)
                if reward_member is None:
                    failed += 1
                    continue
                reward = tar.extractfile(reward_member).read().decode(errors = 'replace').strip()
                if reward != '1':
                    failed += 1
                    continue
                raw = tar.extractfile(member).read().decode('utf-8', errors = 'replace')
                try:
                    kept.append(json.loads(raw))
                except json.JSONDecodeError:
                    failed += 1
    return kept, seen, failed


def task_prefix(messages: list[dict]) -> list[dict]:
    user = next((m for m in messages if m.get('role') == 'user'), None)
    prefix = [{'role': 'system', 'content': SHORT_SYSTEM}]
    if user is not None:
        prefix.append({'role': 'user', 'content': user.get('content') or ''})
    return prefix


def windows(messages: list[dict], tokenizer, block_size: int, history: int) -> list[list[dict]]:
    prefix = task_prefix(messages)
    body = [m for m in messages if m.get('role') != 'system']
    if body and body[0].get('role') == 'user':
        body = body[1:]

    out = []
    for i, m in enumerate(body):
        if m.get('role') != 'assistant':
            continue
        sample = prefix + body[max(0, i - history):i] + [m]
        ids, mask = tokenizer.render_conversation(sample, block_size)
        if 1 not in mask:
            continue
        out.append(sample)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description = 'turn agent benchmark trajectories into windowed SFT conversations')
    parser.add_argument('--archives', nargs = '+', required = True)
    parser.add_argument('--tok', default = 'data/tok.json')
    parser.add_argument('--out', default = 'data/sft_agent.jsonl')
    parser.add_argument('--block-size', type = int, default = 1024)
    parser.add_argument('--history', type = int, default = 6)
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tokenizer import Tokenizer

    tokenizer = Tokenizer.load(args.tok)
    archives = [Path(a) for a in args.archives]
    missing = [a for a in archives if not a.exists()]
    if missing:
        raise FileNotFoundError(f'[agent sft]: these archives do not exist: {missing}')

    trajectories, seen, dropped = iter_trajectories(archives)
    print(f'[agent sft]: {seen} trajectories seen, {len(trajectories)} succeeded (reward=1), {dropped} dropped')

    samples = []
    for messages in trajectories:
        samples.extend(windows(messages, tokenizer, args.block_size, args.history))
    random.Random(args.seed).shuffle(samples)
    print(f'[agent sft]: {len(samples)} windowed samples at history={args.history}')

    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    temp = out.with_name(out.name + '.tmp')
    with open(temp, 'w', encoding = 'utf-8') as f:
        for s in samples:
            f.write(json.dumps({'messages': s}, ensure_ascii = False) + '\n')
    temp.replace(out)

    total = supervised = 0
    for s in samples[:500]:
        ids, mask = tokenizer.render_conversation(s, args.block_size)
        total += len(ids)
        supervised += sum(mask)
    print(f'[agent sft]: wrote {out} ({out.stat().st_size / 1e6:.0f} MB)')
    print(f'[agent sft]: mean {total / min(500, len(samples)):.0f} tokens per sample, {supervised / max(1, total) * 100:.1f}% supervised')
    return


if __name__ == '__main__':
    main()
