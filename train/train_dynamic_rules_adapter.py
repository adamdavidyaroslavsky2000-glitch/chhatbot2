import os, json, argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizer.spm_tokenizer import SPMTokenizer
from model.transformer import TinyTransformerLM
from glob import glob
class DynamicRuleDataset(Dataset):
    def __init__(self, glob_pat, tokenizer: SPMTokenizer, max_seq_len: int = 1024, limit_per_file: int = None):
        self.tok = tokenizer; self.max_seq = max_seq_len; self.samples = []
        for p in sorted(glob(glob_pat)):
            with open(p, 'r', encoding='utf-8') as f:
                for i, ln in enumerate(f):
                    if limit_per_file and i >= limit_per_file: break
                    obj = json.loads(ln); inp = obj.get('input',''); out = obj.get('output','')
                    fixed = obj.get('fixed_rules',[]); dyn = obj.get('dynamic_rules',[]); controls = obj.get('controls',{})
                    header = ('Guidance:\n' + 'Fixed: ' + ', '.join(r.get('name','') for r in fixed) + '\n' + 'Dynamic: ' + ', '.join(r.get('name','') for r in dyn) + '\n' + 'Controls: ' + json.dumps(controls) + '\n')
                    prompt = header + '\n' + inp + '\n\nAssistant:'; target = out
                    ids = self.tok.encode(prompt + ' ' + target, add_bos=True, add_eos=True)
                    if len(ids) > self.max_seq: ids = ids[:self.max_seq]
                    self.samples.append(ids)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids = self.samples[i]
        x = torch.tensor(ids[:-1], dtype=torch.long); y = torch.tensor(ids[1:], dtype=torch.long)
        pad = self.max_seq - len(ids) + 1
        if pad > 0:
            x = torch.cat([x, torch.full((pad,), self.tok.PAD, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad,), self.tok.PAD, dtype=torch.long)])
        return x[:self.max_seq-1], y[:self.max_seq-1]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_in', type=str, required=True)
    ap.add_argument('--ckpt_out', type=str, default='checkpoints/dynamic_rules_adapter.pt')
    ap.add_argument('--spm_model', type=str, default='tokenizer/spm.model')
    ap.add_argument('--rules_glob', type=str, default='rule_data/dynamic_rules_*.jsonl')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--device', type=str, default='mps')
    ap.add_argument('--limit_per_file', type=int, default=None)
    args = ap.parse_args()
    tok = SPMTokenizer(args.spm_model)
    base = torch.load(args.ckpt_in, map_location='cpu')
    cfg = base['config']; cfg['vocab_size'] = tok.VOCAB_SIZE
    model = TinyTransformerLM(**cfg); model.load_state_dict(base['model_state'])
    import torch.backends.mps as mps_backend
    device = torch.device(args.device if (hasattr(mps_backend,'mps') and mps_backend.is_available()) or args.device!='mps' else 'cpu')
    model.to(device); model.train()
    ds = DynamicRuleDataset(args.rules_glob, tokenizer=tok, max_seq_len=cfg['max_seq_len'], limit_per_file=args.limit_per_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9,0.95), weight_decay=0.05)
    for step, (x,y) in enumerate(dl):
        if step >= args.steps: break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type!='cpu' else torch.float32, enabled=(device.type!='cpu')):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=tok.PAD)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0: print(f'[dynamic-rules] step {step}/{args.steps} loss {loss.item():.3f}')
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'config': cfg, 'spm_model': args.spm_model}, args.ckpt_out)
    print(f'✅ Saved dynamic-rules adapter checkpoint to {args.ckpt_out}')
if __name__ == '__main__': main()
