import argparse, json, torch, math, string, unicodedata
from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM
PRINTABLE = set(string.printable) | {'\n', ' '}
STOP_STRINGS = ['\nSystem:', '\nSource:', '\nUser:']
def is_printable_english(s: str) -> bool:
    import unicodedata
    if not s: return False
    for ch in s:
        if ch == '\uFFFD' or unicodedata.category(ch).startswith('C'): return False
        if ch not in PRINTABLE: return False
    return True
def build_bad_token_ids(tok):
    bad = []
    for tid in range(tok.VOCAB_SIZE):
        piece = tok.decode([tid])
        if not is_printable_english(piece): bad.append(tid)
    return set(bad)
def block_repeated_trigrams(x_ids: torch.Tensor, logits: torch.Tensor):
    if x_ids.size(1) < 3: return
    hist = x_ids[0].tolist(); tail = tuple(hist[-2:])
    for i in range(len(hist)-2):
        tri = tuple(hist[i:i+3])
        if tri[:-1] == tail: logits[0, tri[-1]] = -1e9
@torch.no_grad()
def sample_dynamic_rules(model, tok, prompt, fixed_rules, dynamic_rules, controls, topic, fmt, device='cpu'):
    temperature = float(controls.get('temperature', 0.7)); top_p = float(controls.get('top_p', 0.9))
    freq_pen = float(controls.get('frequency_penalty', 0.8)); pres_pen = float(controls.get('presence_penalty', 0.5))
    max_tokens = int(controls.get('max_tokens', 200))
    x = torch.tensor(tok.encode(prompt, add_bos=True, add_eos=False), dtype=torch.long, device=device).unsqueeze(0)
    bad_ids = build_bad_token_ids(tok); topic_ids = set(tok.encode(topic, add_bos=False, add_eos=False)); freq = {}
    for step in range(max_tokens):
        if x.size(1) > model.max_seq_len: x = x[:, -model.max_seq_len:]
        logits = model(x)[:, -1, :]
        for tid, cnt in freq.items(): logits[0, tid] -= freq_pen * cnt
        for tid in set(x[0].tolist()): logits[0, tid] -= pres_pen
        for tt in topic_ids: logits[0, tt] += 0.7
        if bad_ids: logits[0, list(bad_ids)] = -1e9
        for r in dynamic_rules:
            name = r.get('name',''); params = r.get('params',{}); w = float(params.get('weight', 0.3))
            if name == 'penalize_meta_leak':
                for sym in ['System:', 'Source:', '<', '>', '{%']:
                    for b in tok.encode(sym, add_bos=False, add_eos=False): logits[0, b] -= 1.2 * w
            elif name == 'boost_domain_terms':
                for tt in topic_ids: logits[0, tt] += 0.5 * w
            elif name == 'json_guard' and fmt == 'json':
                for sym in ['{','}','"',':',',']:
                    for b in tok.encode(sym, add_bos=False, add_eos=False): logits[0, b] += 0.3 * w
            elif name == 'bullets_guard' and fmt == 'bullets':
                for b in tok.encode('- ', add_bos=False, add_eos=False): logits[0, b] += 0.2 * w
            elif name == 'numbers_guard' and fmt == 'numbered':
                for sym in ['1','2','3','4',')']:
                    for b in tok.encode(sym, add_bos=False, add_eos=False): logits[0, b] += 0.1 * w
            elif name == 'term_dedup':
                for tid, cnt in sorted(freq.items(), key=lambda kv: -kv[1])[:10]:
                    logits[0, tid] -= 0.15 * w * cnt
            elif name == 'caveat_inserter':
                if step > max_tokens*0.7:
                    for sym in ['However', 'Assumption', 'Limit', 'Caveat']:
                        for b in tok.encode(sym, add_bos=False, add_eos=False): logits[0, b] += 0.1 * w
        prog = step / max(1, max_tokens - 1)
        dyn_temp = max(0.5, min(0.9, temperature * (1.0 - 0.2 * prog)))
        logits = logits / max(1e-6, dyn_temp)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(sorted_probs, num_samples=1)
        token = sorted_idx.gather(-1, idx_next)
        tid = token.item(); x = torch.cat([x, token], dim=1); freq[tid] = freq.get(tid, 0) + 1
        text = tok.decode(x.squeeze(0).tolist())
        if any(s in text for s in STOP_STRINGS) and step > 60:
            pos = min([text.find(s) for s in STOP_STRINGS if s in text]); return text[:pos].rstrip()
    return tok.decode(x.squeeze(0).tolist()).rstrip()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--spm_model', type=str, default='tokenizer/spm.model')
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--topic', type=str, required=True)
    ap.add_argument('--format', type=str, default='paragraph')
    ap.add_argument('--fixed_json', type=str, default='[]')
    ap.add_argument('--dynamic_json', type=str, default='[]')
    ap.add_argument('--controls_json', type=str, default='{}')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()
    tok = SPMTokenizer(args.spm_model)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg = ckpt['config']
    model = TinyTransformerLM(**cfg); model.load_state_dict(ckpt['model_state']); model.to(args.device).eval()
    fixed_rules = json.loads(args.fixed_json); dynamic_rules = json.loads(args.dynamic_json); controls = json.loads(args.controls_json)
    txt = sample_dynamic_rules(model, tok, args.prompt, fixed_rules, dynamic_rules, controls, topic=args.topic, fmt=args.format, device=args.device)
    print(txt)
if __name__ == '__main__': main()
