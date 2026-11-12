"""Phase 4a: mark morpheme boundaries in the 1B-char reference training mix.

Segmenter: CAMeL Tools MLE disambiguators (context-free, so each word *type* is segmented once).
  - MSA lines (arb) and Gulf/Moroccan lines (ars, ary): calima-msa-r13
  - Egyptian lines (arz): calima-egy-r13
Two granularities:
  clitic   : d3seg (MSA db for all lines; the egy db has no d3seg) - splits proclitics و ف ب ل ك س ال and pronominal enclitics; surface-exact
  morpheme : bwtok          - also splits inflectional affixes (ي+كتب+ون, مدرس+ة); Egyptian db handles
             dialect morphology (ما+ب+ي+عرف+ش). bwtok normalizes spelling, so its boundaries are mapped
             back onto the surface word by character alignment (difflib); poor alignments are dropped.
Only word types seen >= 2 times are segmented (singletons stay whole).
Output: data/train_text/mix_{clitic,morpheme}.txt with U+E000 between morphemes inside words,
        data/morph/segmentations_{scheme}.pkl  {group: {word: (boundary offsets)}}
"""
import os, pickle, re, sys, time
from collections import Counter
from difflib import SequenceMatcher
from multiprocessing import Pool

B = os.environ["B"]
TXT, OUT = f"{B}/data/train_text", f"{B}/data/morph"
MARK = ""
WORD = re.compile("[ء-غف-يٱ-ۓ]+")
# block order and sizes written by train_reference.build_mix_text (see logs/train_data_mix.log)
BLOCKS = [("arb_Arab", 500_012_282), ("ary_Arab", 166_673_067), ("arz_Arab", 166_694_648), ("ars_Arab", 166_666_826)]
GROUP = {"arb_Arab": "msa", "ary_Arab": "msa", "ars_Arab": "msa", "arz_Arab": "egy"}
DB = {"msa": "calima-msa-r13", "egy": "calima-egy-r13"}


def line_groups(lines):
    out, b, used = [], 0, 0
    for ln in lines:
        out.append(GROUP[BLOCKS[b][0]])
        used += len(ln) - 1
        if used >= BLOCKS[b][1] and b < len(BLOCKS) - 1:
            b, used = b + 1, 0
    return out


def count_chunk(args):
    lines, groups = args
    c = {"msa": Counter(), "egy": Counter()}
    for ln, g in zip(lines, groups):
        c[g].update(WORD.findall(ln))
    return c


def boundaries(pieces, surface):
    """Boundary offsets in `surface` for a list of segment strings (spelling may differ)."""
    segs = [p.strip("+") for p in pieces if p.strip("+")]
    if len(segs) < 2:
        return ()
    ps = "".join(segs)
    cuts = []
    n = 0
    for s in segs[:-1]:
        n += len(s)
        cuts.append(n)
    if ps == surface:
        return tuple(cuts)
    sm = SequenceMatcher(None, ps, surface, autojunk=False)
    if sm.ratio() < 0.75:
        return ()
    m = [None] * (len(ps) + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(i1, i2 + 1):
            if m[k] is None:
                m[k] = j1 + (k - i1 if tag == "equal" else round((k - i1) * (j2 - j1) / max(i2 - i1, 1)))
    out = sorted({min(max(m[c], 0), len(surface)) for c in cuts})
    return tuple(x for x in out if 0 < x < len(surface))


_tk = {}


def seg_init(group):
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer
    mle = MLEDisambiguator.pretrained(DB[group])
    msa = mle if group == "msa" else MLEDisambiguator.pretrained(DB["msa"])  # egy db has no d3seg
    _tk["clitic"] = MorphologicalTokenizer(msa, scheme="d3seg", split=False, diac=False)
    _tk["morpheme"] = MorphologicalTokenizer(mle, scheme="bwtok", split=False, diac=False)


def seg_chunk(words):
    res = {"clitic": {}, "morpheme": {}}
    for scheme in res:
        out = _tk[scheme].tokenize(words)
        for w, o in zip(words, out):
            b = boundaries(o.split("_"), w)
            if b:
                res[scheme][w] = b
    return res


def mark_chunk(args):
    lines, groups, scheme = args
    seg = SEGS[scheme]
    def rep(m, g):
        w = m.group(0)
        b = seg[g].get(w)
        if not b:
            return w
        parts, prev = [], 0
        for x in b:
            parts.append(w[prev:x]); prev = x
        parts.append(w[prev:])
        return MARK.join(parts)
    return "".join(WORD.sub(lambda m: rep(m, g), ln) for ln, g in zip(lines, groups))


SEGS = {}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    lines = open(f"{TXT}/mix.txt").readlines()
    assert not any(MARK in ln for ln in lines[:100000]), "marker already present in text"
    groups = line_groups(lines)
    print(f"{len(lines):,} lines; groups: {Counter(groups)}  ({time.time() - t0:.0f}s)", flush=True)
    n = 96
    step = len(lines) // n + 1
    chunks = [(lines[i:i + step], groups[i:i + step]) for i in range(0, len(lines), step)]
    with Pool(n) as p:
        parts = p.map(count_chunk, chunks)
    counts = {g: Counter() for g in DB}
    for c in parts:
        for g in DB:
            counts[g].update(c[g])
    for g in DB:
        print(f"{g}: {len(counts[g]):,} types, {sum(counts[g].values()):,} tokens, "
              f"{sum(1 for v in counts[g].values() if v >= 2):,} types with freq>=2  ({time.time() - t0:.0f}s)", flush=True)

    segs = {"clitic": {}, "morpheme": {}}
    for g in DB:
        types = [w for w, c in counts[g].items() if c >= 2]
        wchunks = [types[i:i + 2000] for i in range(0, len(types), 2000)]
        with Pool(64, initializer=seg_init, initargs=(g,)) as p:
            for k, r in enumerate(p.imap_unordered(seg_chunk, wchunks, chunksize=4)):
                for s in segs:
                    segs[s].setdefault(g, {}).update(r[s])
                if k % 500 == 0:
                    print(f"  {g}: {k:,}/{len(wchunks):,} chunks ({time.time() - t0:.0f}s)", flush=True)
        tok_cov = {s: sum(counts[g][w] for w in segs[s].get(g, {})) / sum(counts[g].values()) for s in segs}
        print(f"{g}: segmented word types clitic={len(segs['clitic'].get(g, {})):,} morpheme={len(segs['morpheme'].get(g, {})):,}; "
              f"share of running words with >=1 boundary: {tok_cov}  ({time.time() - t0:.0f}s)", flush=True)
    for s in segs:
        pickle.dump(segs[s], open(f"{OUT}/segmentations_{s}.pkl", "wb"))
    SEGS.update(segs)

    for s in segs:
        with Pool(n) as p:
            texts = p.map(mark_chunk, [(l, g, s) for l, g in chunks])
        with open(f"{TXT}/mix_{s}.txt", "w") as f:
            for t in texts:
                f.write(t)
        print(f"wrote mix_{s}.txt  ({time.time() - t0:.0f}s)", flush=True)
