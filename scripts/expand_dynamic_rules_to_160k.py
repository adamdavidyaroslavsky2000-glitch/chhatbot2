#!/usr/bin/env python3
import json, random, argparse, hashlib, glob
random.seed(1337)
GOOD_PHRASES = ["(Ensure relevance.)","(Avoid repetition.)","(List assumptions.)","(Provide one metric.)","(Stay concise.)","(Cite uncertainty when needed.)","(Prefer concrete verbs.)"]
def mutate(rec):
    new = json.loads(json.dumps(rec))
    c = new.get("controls", {})
    for k in ("temperature","top_p","frequency_penalty","presence_penalty"):
        if k in c: c[k] = float(max(0.3, min(1.2, c[k] + random.uniform(-0.08, 0.08))))
    if "max_tokens" in c: c["max_tokens"] = int(max(80, min(260, c["max_tokens"] + random.randint(-10, 10))))
    txt = new.get("input","")
    if random.random() < 0.25:
        txt = txt.replace("Tone: concise", "Tone: neutral").replace("Tone: friendly", "Tone: formal").replace("Tone: formal", "Tone: friendly")
        new["input"] = txt
    out = new.get("output","").strip()
    new["output"] = out + "\n" + random.choice(GOOD_PHRASES)
    dr = new.get("dynamic_rules", [])
    random.shuffle(dr)
    for r in dr[: random.randint(1,3)]:
        p = r.get("params", {})
        for k, v in list(p.items()):
            if isinstance(v,(int,float)):
                p[k] = float(max(0.1, min(1.5, v + random.uniform(-0.05, 0.05))))
    return new
def expand_file(path, target):
    with open(path, "r", encoding="utf-8") as f:
        seeds = [json.loads(ln) for ln in f]
    out, seen = [], set()
    def add(o):
        s = json.dumps(o, ensure_ascii=False, sort_keys=True)
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        if h not in seen: seen.add(h); out.append(o)
    for s in seeds: add(s)
    while len(out) < target: add(mutate(random.choice(seeds)))
    with open(path, "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Expanded {path} -> {len(out)} lines")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default="rule_data/dynamic_rules_*.jsonl")
    ap.add_argument("--target", type=int, default=160000)
    args = ap.parse_args()
    for fp in sorted(glob.glob(args.glob)): expand_file(fp, args.target)
if __name__ == "__main__": main()
