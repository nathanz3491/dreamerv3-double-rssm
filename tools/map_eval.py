"""Score RSSM-2's map belief against the true map, split seen / unseen / prior.

The point is that these are three different questions and the single BCE the
training run logs blends them into one uninterpretable number:

  seen    cells the agent observed        -- did it REMEMBER
  unseen  cells it never observed         -- did it INFER
  prior   the same cells, scored by a fake model that ignores its input and
          emits the dataset-average per plane -- the "learned nothing" floor

Only `unseen` vs `prior` carries a claim. If they match, the model knows nothing
about terrain it has not seen, which is what a procedurally generated world
implies is knowable. If unseen beats prior, a spatial prior has genuinely
emerged from the agent's own experience -- which is the result worth reporting,
and it is only meaningful on a model trained WITHOUT privileged supervision
(env.craftax.map_privileged False), where unseen cells are a real held-out set.

Ground truth is used here and only here. Measuring against it was never the
problem; learning from it was.

  python tools/map_eval.py --logdir ~/logdir/run --episodes 10 \
      --configs craftax size50m --env.craftax.mapmodel True \
      --agent.mapmodel.enabled True --agent.mapmodel.to_actor True \
      --jax.platform cpu
"""

import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, '.')


def build(argv, logdir):
  import elements
  import ruamel.yaml as yaml
  from dreamerv3 import main as m
  cfg_path = pathlib.Path(m.__file__).parent / 'configs.yaml'
  cfgs = yaml.YAML(typ='safe').load(cfg_path.read_text(encoding='utf-8'))
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(cfgs['defaults'])
  for name in parsed.configs:
    config = config.update(cfgs[name])
  return elements.Flags(config).parse(other + [f'--logdir={logdir}'])


def bce(pred, target, eps=1e-6):
  """Elementwise binary cross-entropy, same form the training loss uses."""
  p = np.clip(pred, eps, 1.0 - eps)
  return -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument('--logdir', required=True)
  ap.add_argument('--episodes', type=int, default=10)
  ap.add_argument('--max-steps', type=int, default=600)
  known_args, rest = ap.parse_known_args()

  config = build(rest, known_args.logdir)
  import jax
  from dreamerv3 import main as dv3main
  from dreamerv3 import craftax_map as M

  agent = dv3main.make_agent(config)
  env = dv3main.make_env(config, 0)
  ckdir = pathlib.Path(known_args.logdir) / 'ckpt'
  target_ck = ckdir / (ckdir / 'latest').read_text().strip()
  agent.load(pickle.loads((target_ck / 'agent.pkl').read_bytes()))

  preds, truths, masks = [], [], []
  for ep in range(known_args.episodes):
    obs = env.step({'action': np.zeros((), np.int32),
                    'reset': np.ones((), bool)})
    carry = agent.init_policy(batch_size=1)
    known = None
    step = 0
    while step < known_args.max_steps:
      batched = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith('log/')}
      carry, act, out = agent.policy(carry, batched, mode='probe')
      if 'map_pred' not in out:
        raise SystemExit(
            'no map_pred in the probe output -- run with '
            '--agent.mapmodel.enabled True --agent.mapmodel.to_actor True')
      with jax.transfer_guard('allow'):
        state = env._state
        known = M.update_known(known, state)
        truths.append(M.coarse_map(state))          # ground truth, eval only
        masks.append(M.known_fraction(known) > 0)
        preds.append(np.asarray(out['map_pred'])[0])
      obs = env.step({'action': np.int32(np.asarray(act['action'])[0]),
                      'reset': np.zeros((), bool)})
      step += 1
      if bool(obs['is_last']):
        break

  pred = np.stack(preds).astype(np.float64)         # (N, 12, 12, P)
  true = np.stack(truths).astype(np.float64)[..., :pred.shape[-1]]
  seen = np.stack(masks)[..., None]                 # (N, 12, 12, 1)
  # P_SEEN is the model's own visitation recency, not a claim about terrain, and
  # coarse_map leaves it at zero here. Scoring it would compare the model against
  # a target this script never filled in. Terrain planes only.
  keep = [i for i in range(pred.shape[-1]) if i != M.P_SEEN]
  pred, true = pred[..., keep], true[..., keep]
  names = [M.PLANE_NAMES[i] for i in keep]

  # The floor: ignore the input entirely and emit each plane's mean. Fitted on
  # the very data it is scored on, which makes it a GENEROUS floor -- anything
  # the model cannot beat here, it has no business claiming to predict.
  prior = np.broadcast_to(true.mean((0, 1, 2)), true.shape)
  model_l, prior_l = bce(pred, true), bce(prior, true)
  seen_b = np.broadcast_to(seen, true.shape)
  n_seen, n_unseen = seen_b.sum(), (~seen_b).sum()

  def split(loss, plane=None):
    l = loss if plane is None else loss[..., plane:plane + 1]
    m = seen_b if plane is None else seen_b[..., plane:plane + 1]
    return (l[m].mean() if m.any() else float('nan'),
            l[~m].mean() if (~m).any() else float('nan'))

  def gain(floor, got):
    """Share of the floor's loss the model removes. Blank on a dead plane: one
    that is zero everywhere has a floor of ~0, and the ratio then reports an
    arbitrarily large number about nothing."""
    return (floor - got) / floor * 100 if floor > 1e-4 else float('nan')

  def pct(x):
    return f'{x:+8.1f}%' if np.isfinite(x) else f'{"--":>9}'

  print()
  print(f'{known_args.episodes} episodes, {len(preds)} steps, '
        f'{len(names)} terrain planes')
  print(f'cells observed: {100 * n_seen / (n_seen + n_unseen):.1f}%')
  print()

  ms, mu = split(model_l)
  ps, pu = split(prior_l)
  print('                      seen      unseen')
  print(f'  model             {ms:7.4f}   {mu:7.4f}')
  print(f'  prior floor       {ps:7.4f}   {pu:7.4f}')
  print()
  print(f'  seen   cells: model beats the prior floor by {pct(gain(ps, ms))}')
  print(f'  unseen cells: model beats the prior floor by {pct(gain(pu, mu))}')
  print()
  print('  unseen ~0%  => it knows nothing about terrain it never saw, which is')
  print('                 all a fresh-map-per-episode world allows.')
  print('  unseen >0%  => a spatial prior emerged. Only meaningful on a model')
  print('                 trained WITHOUT privileged supervision.')
  print('  unseen <0%  => confidently WRONG: worse than saying nothing, and the')
  print('                 actor is consuming it.')
  print()

  print(f'{"plane":<13}{"seen":>8}{"pri_s":>8}{"unseen":>8}{"pri_u":>8}'
        f'{"gain_u":>9}{"mean_p":>8}{"mean_t":>8}{"r_seen":>8}')
  for i, name in enumerate(names):
    s_i, u_i = split(model_l, i)
    ps_i, pu_i = split(prior_l, i)
    # Calibration and correlation on observed cells. A model that tracks the map
    # but is badly scaled shows high r with mismatched means; one not tracking it
    # at all shows r near zero, which would mean this script reads the wrong
    # tensor rather than the model being bad.
    m = seen_b[..., i]
    pv, tv = pred[..., i][m], true[..., i][m]
    r = (float(np.corrcoef(pv, tv)[0, 1])
         if pv.size > 1 and pv.std() > 1e-9 and tv.std() > 1e-9 else float('nan'))
    print(f'{name:<13}{s_i:8.4f}{ps_i:8.4f}{u_i:8.4f}{pu_i:8.4f}'
          f'{pct(gain(pu_i, u_i))}{pv.mean():8.3f}{tv.mean():8.3f}{r:8.3f}')


if __name__ == '__main__':
  main()
