"""Recompute the main results of the controlled experiments from the saved outputs.

Usage:  python verify_results.py <folder with the result files>
1. Checks every file against the SHA-256 fingerprints in MANIFEST.md (detects any later edit).
2. Recomputes accuracy, McNemar tests and token totals from the raw per-question outputs,
   independently of the notebook that produced the tables.
"""
import hashlib, json, os, re, sys
from math import comb

folder = sys.argv[1] if len(sys.argv) > 1 else '.'
here = os.path.dirname(os.path.abspath(__file__))

# 1. fingerprints
manifest = open(os.path.join(here, 'MANIFEST.md'), encoding='utf-8').read()
expected = dict(re.findall(r'\| `([^`]+)` \| \d+ \| [^|]* \| `([0-9a-f]{64})` \|', manifest))
print('File fingerprints:')
for name, h in expected.items():
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        print(f'  missing   {name}'); continue
    ok = hashlib.sha256(open(p, 'rb').read()).hexdigest() == h
    print(f'  {"OK      " if ok else "CHANGED "} {name}')

# 2. recompute from the raw outputs
def cap3(ga):
    for s in ga[:3]:
        if s['ans'] is not None:
            return s['ans']
        if s['round'] == 3 or s is ga[-1]:
            return (s.get('forced') or {}).get('ans')

def one_round(ga):
    s = ga[0]
    return s['ans'] if s['ans'] is not None else (s.get('forced') or {}).get('ans')

def mcnemar(a, b):
    g = sum(1 for x, y in zip(a, b) if x and not y); l = sum(1 for x, y in zip(a, b) if y and not x)
    n, k = g + l, min(g, l)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return g, l, p

for ds in ('primary', 'medmcqa'):
    R = [json.loads(l) for l in open(os.path.join(folder, f'r3_gpt-4.1-mini_{ds}.jsonl'))]
    key = [r['key'] for r in R]
    C = {'No-RAG': [r['norag']['ans'] for r in R], 'Single-pass RAG': [r['sp']['ans'] for r in R],
         'One round': [one_round(r['ga']) for r in R], 'RAG-SEQ (cap 3)': [cap3(r['ga']) for r in R],
         'Decoupled': [r['dec']['ans'] for r in R]}
    ok = {c: [a == k for a, k in zip(v, key)] for c, v in C.items()}
    print(f'\n{ds}: {len(R)} questions; models that answered: {sorted({r["norag"]["served"] for r in R})}')
    for c, v in ok.items():
        print(f'  {c:16s} accuracy {sum(v) / len(v):.3f}   unparsed {sum(a is None for a in C[c])}')
    for a, b in (('RAG-SEQ (cap 3)', 'Single-pass RAG'), ('RAG-SEQ (cap 3)', 'No-RAG'), ('Single-pass RAG', 'No-RAG'),
                 ('One round', 'Single-pass RAG'), ('RAG-SEQ (cap 3)', 'One round'), ('RAG-SEQ (cap 3)', 'Decoupled')):
        g, l, p = mcnemar(ok[a], ok[b])
        print(f'  {a} vs {b}: {g} gained, {l} lost, exact McNemar p = {p:.4f}')

tot = {}
for l in open(os.path.join(folder, 'api_cache.jsonl')):
    r = json.loads(l)['res']; t = tot.setdefault(r['served'], [0, 0, 0, 0, 0.0])
    t[0] += 1; t[1] += r['tin']; t[2] += r['cached']; t[3] += r['tout']; t[4] += r['cost']
print('\nAPI usage recorded in api_cache.jsonl (compare with the OpenAI usage page for 27 Oct 2026):')
for m, (n, tin, cached, tout, usd) in tot.items():
    print(f'  {m}: {n} requests, {tin:,} input tokens ({cached:,} cached), {tout:,} output tokens, US${usd:.2f}')

# 3. runs A and B (29 October 2026), if their files are present
pa = os.path.join(folder, 'r3A_forced_answers_medmcqa.jsonl')
if os.path.exists(pa):
    A = {}
    for l in open(pa):
        x = json.loads(l); A[(x['id'], x['condition'])] = x
    print(f'\nRun A: {len(A)} follow-ups; original request matched in {sum(x["original_call_matched"] for x in A.values())}; '
          f'still unanswered {sum(x["ans"] is None for x in A.values())}')
    R = [json.loads(l) for l in open(os.path.join(folder, 'r3_gpt-4.1-mini_medmcqa.jsonl'))]
    key = [r['key'] for r in R]
    def fu(r, c):
        a = r[c]['ans']
        return a if a is not None or (r['id'], c) not in A else A[(r['id'], c)]['ans']
    C = {'No-RAG': [fu(r, 'norag') for r in R], 'Single-pass RAG': [fu(r, 'sp') for r in R],
         'One round': [one_round(r['ga']) for r in R], 'RAG-SEQ (cap 3)': [cap3(r['ga']) for r in R],
         'Decoupled': [fu(r, 'dec') for r in R]}
    ok = {c: [a == k for a, k in zip(v, key)] for c, v in C.items()}
    print('  MedMCQA with the follow-up answers:')
    for c, v in ok.items():
        print(f'  {c:16s} accuracy {sum(v) / len(v):.3f}   unanswered {sum(a is None for a in C[c])}')
    for a, b in (('RAG-SEQ (cap 3)', 'Single-pass RAG'), ('RAG-SEQ (cap 3)', 'No-RAG'), ('Single-pass RAG', 'No-RAG'),
                 ('One round', 'Single-pass RAG'), ('RAG-SEQ (cap 3)', 'Decoupled')):
        g, l, p = mcnemar(ok[a], ok[b])
        print(f'  {a} vs {b}: {g} gained, {l} lost, exact McNemar p = {p:.4f}')
for ds in ('primary', 'medmcqa'):
    pb = os.path.join(folder, f'r3B_faithfulness_{ds}.jsonl')
    if not os.path.exists(pb):
        continue
    B = [json.loads(l) for l in open(pb)]
    w = [b for b in B if not b['correct']]; c = [b for b in B if b['correct']]
    print(f'\nRun B, {ds}: {len(B)} justifications, context matched {sum(b["context_matched"] for b in B)}; '
          f'supported {sum(b["supported"] for b in B)}; correct answers supported {sum(b["supported"] for b in c)}/{len(c)}; '
          f'wrong answers supported {sum(b["supported"] for b in w)}/{len(w)}')
pc = os.path.join(folder, 'api_cache_runs_A_B.jsonl')
if os.path.exists(pc):
    tot = {}
    for l in open(pc):
        r = json.loads(l)['res']; t = tot.setdefault(r['served'], [0, 0, 0, 0, 0.0])
        t[0] += 1; t[1] += r['tin']; t[2] += r['cached']; t[3] += r['tout']; t[4] += r['cost']
    print('\nAPI usage of runs A and B (compare with the OpenAI usage page for 29 Oct 2026):')
    for m, (n, tin, cached, tout, usd) in tot.items():
        print(f'  {m}: {n} requests, {tin:,} input tokens ({cached:,} cached), {tout:,} output tokens, US${usd:.2f}')

# 4. MedMCQA questions that refer to an image the dataset does not supply (sensitivity analysis)
import csv
qf = os.path.join(folder, 'medmcqa_set.csv')
if os.path.exists(qf) and os.path.exists(pa):
    Qm = {r['id']: r for r in csv.DictReader(open(qf, encoding='utf-8'))}
    pat = re.compile(r"(shown|given|seen|depicted|illustrated)\s+(below|above|here)|\b(is|are|have been|has been|as)\s+shown\b|\bthis segment\b", re.I)
    img = {r['id'] for r in R if pat.search(Qm[r['id']]['question']) or any('<img' in Qm[r['id']][L].lower() for L in 'ABCD')}
    print(f'\nImage-dependent MedMCQA questions: {len(img)} ({", ".join(sorted(i[:8] for i in img))})')
    keep = [r['id'] not in img for r in R]
    orig_sp = [r['sp']['ans'] == r['key'] for r in R]
    for name, a, b in (('original scoring', [cap3(r['ga']) == r['key'] for r in R], orig_sp),
                       ('with follow-up', ok['RAG-SEQ (cap 3)'], ok['Single-pass RAG'])):
        g, l, p = mcnemar([x for x, k in zip(a, keep) if k], [y for y, k in zip(b, keep) if k])
        print(f'  RAG-SEQ vs single-pass RAG without them, {name}: {g} gained, {l} lost, p = {p:.4f}')
