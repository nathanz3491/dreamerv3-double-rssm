"""Turn a tools/death_eval.py log into a docs/eval_results.json run entry.

Reads the per-episode lines and the summary that death_eval prints, and
writes the fields make_comparison_report.py uses: achievements ± standard
error over the worlds, lifespan, normalized return, cause of death, the
unlock rate of every achievement, and the world hash (which must match
across compared runs).

  python tools/eval_entry.py logs/v2_taste.log --id v2_taste \
      --label "v2-taste: rules out, taste in" --short v2-taste
"""

import argparse
import json
import pathlib
import re

import numpy as np

CAUSES = {
    'food ran out first': 'food',
    'drink ran out first': 'drink',
    'energy ran out first': 'energy',
    'no necessity hit zero -> mob damage or fall/lava': 'mob / lava',
}


def parse(text):
  ach = [int(m) for m in re.findall(r'^\s+ep\s+\d+/\d+ .* (\d+) ach ', text,
                                    re.M)]
  if not ach:
    raise ValueError('no episode lines; is this a death_eval log?')
  life = re.search(r'^lifespan\s+mean\s+([\d.]+)', text, re.M)
  ret = re.search(r'^return\s+raw [\d.]+ = ([\d.]+)% normalised', text, re.M)
  world = re.search(r'^worlds .* hash ([0-9a-f]+)', text, re.M)
  rates = {k: float(v) for k, v in re.findall(
      r'^  ([A-Z_]+)\s+([\d.]+)  #*$', text, re.M)}
  never = re.search(r'-- never unlocked \(\d+\): (.*)$', text, re.M)
  if never:
    rates.update({k.strip(): 0.0 for k in never.group(1).split(',')})
  block = text.split('cause of death', 1)[-1].split('reference:', 1)[0]
  causes = {CAUSES[c.strip()]: int(n) for n, c in re.findall(
      r'^\s+(\d+)\s+(.+)$', block, re.M) if c.strip() in CAUSES}
  a = np.array(ach, float)
  return dict(
      episodes=len(a), achievements=round(float(a.mean()), 2),
      se=round(float(a.std(ddof=1) / np.sqrt(len(a))), 2),
      lifespan=round(float(life.group(1)), 1), ret=float(ret.group(1)),
      causes=causes, rates=rates, wake_up=rates.get('WAKE_UP', 0.0),
      world=world.group(1))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('log')
  ap.add_argument('--id', required=True)
  ap.add_argument('--label', required=True)
  ap.add_argument('--short', required=True)
  ap.add_argument('--fixed', action='store_true',
                  help='trained on the 2026-10-11 fixed code (bold row)')
  args = ap.parse_args()
  entry = dict(id=args.id, label=args.label, short=args.short, final=True)
  if args.fixed:
    entry['fixed'] = True
  entry.update(parse(pathlib.Path(args.log).read_text(encoding='utf-8')))
  print(json.dumps(entry, indent=1))


if __name__ == '__main__':
  main()
