#!/bin/bash
set -o pipefail
cd /workspace/mini-LLM/src/main
export HF_TOKEN=$(cat /workspace/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
LOG=/workspace/pipeline.log
POD_ID=$(cat /workspace/.pod_id)

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { say "FAILED at: $*"; echo "PIPELINE_FAILED $*" >> /workspace/STATUS; exit 1; }

say "=== stage 1: fetch shards from HF ==="
if [ ! -f /dev/shm/shards/meta.json ]; then
  python3 - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="ye476/mini-llm-shards", repo_type="dataset",
                      local_dir="/dev/shm/shards", token=os.environ["HF_TOKEN"],
                      max_workers=16)
print("downloaded to", p)
PY
fi
T=$(ls /dev/shm/shards/train_*.bin 2>/dev/null | wc -l)
V=$(ls /dev/shm/shards/val_*.bin 2>/dev/null | wc -l)
say "shards: train=$T val=$V"
[ "$T" = "116" ] && [ "$V" = "1" ] || fail "shard count wrong (train=$T val=$V)"
rm -f data/shards && ln -sfn /dev/shm/shards data/shards

say "=== stage 2: pretrain ==="
python3 -u pretrain.py 2>&1 | tee -a /workspace/train.log | tail -2
[ -f data/pretrain_checkpoint.pt ] || fail "no pretrain checkpoint"
say "pretrain done: $(tail -1 /workspace/train.log)"

say "=== stage 3: export base ==="
python3 export_model.py --ckpt data/pretrain_checkpoint.pt --out /workspace/mini-llm-base.pt 2>&1 | tee -a "$LOG"
[ -f /workspace/mini-llm-base.pt ] || fail "export failed"

say "=== stage 4: build SFT data ==="
python3 tools/build_general_sft.py --max-samples 100000 --out data/sft_general.jsonl 2>&1 | tee -a "$LOG"
[ -f data/sft_agent.jsonl ] || fail "agent sft data missing"
cat data/sft_general.jsonl data/sft_agent.jsonl | shuf > data/sft_mixed.jsonl
say "mixed SFT: $(wc -l < data/sft_mixed.jsonl) conversations"

say "=== stage 5: SFT ==="
python3 -u train.py --data data/sft_mixed.jsonl --base data/pretrain_checkpoint.pt --out data/sft_checkpoint.pt 2>&1 | tee -a /workspace/sft.log | tail -2
[ -f data/sft_checkpoint.pt ] || fail "no sft checkpoint"
python3 export_model.py --ckpt data/sft_checkpoint.pt --out /workspace/mini-llm-sft.pt 2>&1 | tee -a "$LOG"

say "=== stage 6: upload weights to HF ==="
python3 - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import HfApi, create_repo
tok = os.environ["HF_TOKEN"]
api = HfApi(token=tok)
repo = "ye476/mini-llm-132m"
create_repo(repo, private=True, exist_ok=True, token=tok)
for f in ("mini-llm-base.pt", "mini-llm-sft.pt"):
    api.upload_file(path_or_fileobj=f"/workspace/{f}", path_in_repo=f,
                    repo_id=repo, token=tok)
    print("uploaded", f)
for f in ("/workspace/train.log", "/workspace/sft.log"):
    api.upload_file(path_or_fileobj=f, path_in_repo=f"logs/{os.path.basename(f)}",
                    repo_id=repo, token=tok)
print("UPLOAD_OK")
PY
grep -q UPLOAD_OK "$LOG" || fail "weight upload to HF failed"

say "=== stage 7: done, stopping pod ==="
echo "PIPELINE_OK" >> /workspace/STATUS
sync
sleep 120
curl -s -m 30 -H "Content-Type: application/json" \
  "https://api.runpod.io/graphql?api_key=$(cat /workspace/.rpkey)" \
  -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" \
  >> "$LOG" 2>&1
