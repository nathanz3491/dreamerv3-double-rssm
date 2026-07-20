"""Go/no-go probe for the compositional reward-cause head (Phase 1).

This is the experiment the whole reward-cause direction hinges on. It loads a
trained rewcause checkpoint, runs greedy eval episodes on Craftax, and -- at
every step where the *held-out* achievement actually unlocked -- reads the
head's predicted feature vector phi. If compositional transfer worked, the head
recovers the held-out achievement's defining features (e.g. verb=MAKE,
tool=PICKAXE, material=IRON, tier>=3 for iron-pickaxe) purely by extrapolating
from the wood/stone tiers it *did* train on. rewcause_eval scores that.

Invoke via main.py (reuses the full config -> agent/env/checkpoint pipeline):

  python dreamerv3/main.py --configs craftax size50m \\
      --script rewcause_probe \\
      --rewcause True --rewcause_holdout MAKE_IRON_PICKAXE \\
      --run.from_checkpoint <logdir>/ckpt/latest

Note on reachability: the head is only *blinded* to the held-out achievement
(its loss is masked there); the environment still yields it whenever the policy
gets there. So the probe can only score states the eval policy actually reaches.
If the agent never mines iron, hold out an achievement it *does* reach
(e.g. MAKE_STONE_PICKAXE) -- that is the identical tier-extrapolation test, just
executable with a weaker agent.
"""

from functools import partial as bind

import elements
import embodied
import numpy as np

# How long to probe. Stop as soon as we have enough held-out target states, or
# when the episode budget is exhausted (the agent may simply never reach it).
PROBE_MAX_EPISODES = 300
PROBE_MIN_STATES = 50
STEPS_PER_CHUNK = 100


def rewcause_probe(make_agent, make_env, args, holdout):
  assert args.from_checkpoint, (
      'rewcause_probe needs a trained checkpoint: '
      'pass --run.from_checkpoint <path>')

  # Imported here (not at module load) so dreamerv3 is on sys.path via main.py.
  from dreamerv3 import craftax_features as cf
  from dreamerv3 import rewcause_eval as rce

  # Resolve the held-out target (first token of rewcause_holdout: name or index).
  target_name = None
  for tok in str(holdout).split(','):
    tok = tok.strip()
    if tok:
      target_name = cf.ACHIEVEMENT_NAMES[int(tok)] if tok.isdigit() else tok
      break
  assert target_name, (
      'rewcause_probe needs a held-out achievement. Re-run the probe with '
      '--rewcause True --rewcause_holdout MAKE_IRON_PICKAXE (the same holdout '
      'the checkpoint was trained with).')
  assert target_name in cf.ACHIEVEMENT_NAMES, (
      f'Unknown achievement {target_name!r}; expected one of '
      f'{cf.ACHIEVEMENT_NAMES}')
  target_idx = cf.ACHIEVEMENT_NAMES.index(target_name)

  agent = make_agent()

  cp = elements.Checkpoint()
  cp.agent = agent
  cp.load(args.from_checkpoint, keys=['agent'])

  collected = []
  episodes = [0]

  def collect(tran, worker):
    # `ach` is the multi-hot of achievements newly unlocked this step; when the
    # held-out target fires, grab the head's per-feature prediction for it.
    if 'rewcause_prob' not in tran:
      return
    if float(tran['ach'][target_idx]) > 0.5:
      collected.append(np.asarray(tran['rewcause_prob'], np.float32))
    if bool(tran['is_last']):
      episodes[0] += 1

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=(not args.debug))
  driver.on_step(collect)

  print(f'Probing reward-cause head for held-out target: {target_name}')
  print(f'Checkpoint: {args.from_checkpoint}')
  # 'probe' mode makes the agent emit out['rewcause_prob']; see Agent.policy.
  policy = lambda *a: agent.policy(*a, mode='probe')
  driver.reset(agent.init_policy)
  try:
    while episodes[0] < PROBE_MAX_EPISODES and len(collected) < PROBE_MIN_STATES:
      driver(policy, steps=STEPS_PER_CHUNK)
  finally:
    driver.close()

  n = len(collected)
  header = (
      f'\n=== reward-cause probe: {target_name} ===\n'
      f'held-out states collected : {n}\n'
      f'episodes run              : {episodes[0]}\n')
  if n == 0:
    report = header + (
        'RESULT: inconclusive -- the eval policy never unlocked '
        f'{target_name}, so there are no held-out states to score.\n'
        'Fixes: (1) hold out an achievement the agent actually reaches, e.g. '
        'MAKE_STONE_PICKAXE (same tier-extrapolation test); (2) raise '
        'PROBE_MAX_EPISODES; (3) probe a stronger checkpoint.')
  else:
    probs = np.stack(collected, 0)                       # [N, PHI_DIM]
    metrics = rce.score_predictions(probs, target_name=target_name)
    report = header + rce.format_report(metrics)

  print(report)
  outfile = elements.Path(args.logdir) / 'rewcause_probe.txt'
  outfile.write(report, mode='w')
  print(f'\nWrote probe report: {outfile}')
  return report
