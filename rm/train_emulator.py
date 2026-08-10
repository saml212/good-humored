#!/usr/bin/env python3
"""Emulator v1 trainer: DeBERTa-v3-base + scalar head regressing the
within-contest percentile of NYCC captions.

Pre-registered eval: mean within-contest Spearman on the 77 held-out
val contests (same metric as the reaction-logprob chain; its
zero-training result was 0.122). Best checkpoint by val Spearman is
saved; every eval is appended to the log so the curve is inspectable.

Usage (box, one GPU):
  CUDA_VISIBLE_DEVICES=0 python3 -m rm.train_emulator \
      --data-dir /data/good-humored/emulator-v1 \
      --out-dir /data/good-humored/runs/emulator_v1 \
      --epochs 2 --batch-size 256
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

# deberta-v3 NaN'd on the FIRST optimizer step in fp32 AND bf16 on the
# box (very new transformers; its rewritten DebertaV2 path is the
# suspect -- step-1 loss was sane both times). roberta-base is the
# version-robust default; --model allows a deberta retry in a pinned venv.
MODEL_NAME = "roberta-base"
MAX_LEN = 128


class CaptionDataset(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r["context"], r["text"], r["target"], r["contest"]


def collate(tokenizer):
    def fn(batch):
        ctx, txt, tgt, contest = zip(*batch)
        enc = tokenizer(list(ctx), list(txt), truncation=True,
                        max_length=MAX_LEN, padding=True,
                        return_tensors="pt")
        return enc, torch.tensor(tgt, dtype=torch.float32), contest
    return fn


def spearman(a, b):
    """Spearman rho via rank correlation (average ranks for ties)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return cov / (sa * sb) if sa * sb else 0.0


@torch.no_grad()
def evaluate(model, loader, device, log):
    model.eval()
    by_contest = {}
    for enc, tgt, contests in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        pred = model(**enc).logits.squeeze(-1).float().cpu().tolist()
        for p, t, c in zip(pred, tgt.tolist(), contests):
            by_contest.setdefault(c, ([], []))
            by_contest[c][0].append(p)
            by_contest[c][1].append(t)
    rhos = [spearman(p, t) for p, t in by_contest.values()]
    mean_rho = sum(rhos) / len(rhos)
    frac_pos = sum(1 for r in rhos if r > 0) / len(rhos)
    log("EVAL contests=%d mean_within_contest_spearman=%.4f frac_positive=%.3f"
        % (len(rhos), mean_rho, frac_pos))
    model.train()
    return mean_rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--evals-per-epoch", type=int, default=2)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers; 0 (main-process) is the "
                         "default after a fork-after-CUDA hang froze the "
                         "first full run mid-loop at step ~400 for 2h+")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                    help="deberta-v3 NaN'd immediately under bf16 autocast "
                         "(disentangled-attention half-precision fragility); "
                         "fp32 is the safe default for a 184M model on H100")
    args = ap.parse_args()

    torch.manual_seed(0)
    out = Path(args.out_dir)
    if (out / "best").exists():
        raise SystemExit("out-dir already has a best/ checkpoint -- "
                         "use a fresh out-dir (rerun hygiene, audit #5)")
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(__file__, out / "train_emulator.archived.py")
    log_f = open(out / "train.log", "a")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, get_linear_schedule_with_warmup)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression").to(device)

    train_ds = CaptionDataset(Path(args.data_dir) / "nycc_train.jsonl")
    val_ds = CaptionDataset(Path(args.data_dir) / "nycc_val.jsonl")
    log("train=%d val=%d" % (len(train_ds), len(val_ds)))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate(tokenizer),
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False,
                        collate_fn=collate(tokenizer),
                        num_workers=args.workers)

    steps_total = len(train_dl) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.03 * steps_total),
                                            steps_total)
    use_autocast = args.precision == "bf16"
    eval_every = max(1, len(train_dl) // args.evals_per_epoch)
    best = -1.0
    best_step = 0
    step = 0
    log("steps_total=%d eval_every=%d" % (steps_total, eval_every))
    for epoch in range(args.epochs):
        for enc, tgt, _ in train_dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            tgt = tgt.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_autocast):
                pred = model(**enc).logits.squeeze(-1)
                loss = torch.nn.functional.mse_loss(pred.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step <= 3 or step % 100 == 0:
                log("step %d/%d loss=%.4f lr=%.2e"
                    % (step, steps_total, loss.item(), sched.get_last_lr()[0]))
                if loss.item() != loss.item():  # NaN: fail fast and loud
                    log("FATAL: loss is NaN at step %d -- aborting" % step)
                    raise SystemExit(1)
            if step % eval_every == 0:
                rho = evaluate(model, val_dl, device, log)
                if rho > best:
                    best, best_step = rho, step
                    model.save_pretrained(out / "best")
                    tokenizer.save_pretrained(out / "best")
                    log("new best %.4f -> saved" % best)
    rho = evaluate(model, val_dl, device, log)
    if rho > best:
        best, best_step = rho, step
        model.save_pretrained(out / "best")
        tokenizer.save_pretrained(out / "best")
    log("DONE best_mean_within_contest_spearman=%.4f (step %d)"
        % (best, best_step))
    (out / "RESULT.json").write_text(json.dumps(
        {"best_mean_within_contest_spearman": round(best, 4),
         "best_step": best_step,
         "train_rows": len(train_ds), "val_rows": len(val_ds),
         "model": args.model, "epochs": args.epochs,
         "batch_size": args.batch_size, "lr": args.lr}))


if __name__ == "__main__":
    main()
