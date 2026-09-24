"""Draw the current model -- DreamerV3 + RSSM-2 map model + tech-tree potential.

Writes docs/architecture.svg, and docs/architecture.pdf when a Chromium browser
(Edge or Chrome) is available to print it. Coordinates are laid out by hand;
the point of the figure is that every box sits where the code puts it:

  * the potential lives in the ENV WRAPPER and reaches the agent only as reward;
  * RSSM-2 reads sg(feat) and its own actions, never coordinates or the map;
  * the map crop feeds pi and V only -- never the reward/continue heads.

  python tools/make_architecture.py
"""

import html
import pathlib
import re
import shutil
import subprocess

W, H = 1900, 1330
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"
INK, DIM, ARROW, TARGET = '#212529', '#495057', '#343a40', '#868e96'

FAMILY = {
    #          region fill   stroke     box fill
    'env':    ('#f1f3f5', '#868e96', '#ffffff'),
    'phi':    ('#fff4e0', '#e8a33d', '#fff4e0'),
    'wm':     ('#eef4ff', '#4c7bd9', '#ffffff'),
    'map':    ('#ecf8f1', '#2f9e6a', '#ffffff'),
    'ac':     ('#f4f0ff', '#7b5cd6', '#ffffff'),
    'replay': ('#f8f9fa', '#495057', '#f8f9fa'),
    'note':   ('#f8f9fa', '#adb5bd', '#f8f9fa'),
    'prop':   ('#fafafa', '#adb5bd', '#fafafa'),
}

out = []


def rich(text):
  """Escape, then turn x_{t} into a real subscript."""
  text = html.escape(text, quote=False)
  return re.sub(r'_\{([^}]*)\}',
                r'<tspan baseline-shift="sub" font-size="75%">\1</tspan>', text)


def label(x, y, text, size=12, weight='normal', color=INK, anchor='start',
          style='normal', rot=0):
  turn = f' transform="rotate({rot} {x} {y})"' if rot else ''
  out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
             f'font-style="{style}" fill="{color}" text-anchor="{anchor}"{turn}>'
             f'{rich(text)}</text>')


def region(x, y, w, h, fam, title):
  fill, stroke, _ = FAMILY[fam]
  out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
  label(x + 16, y + 24, title, 15, 'bold', stroke)


def box(x, y, w, h, fam, title, lines=(), dashed=False, fill=None,
        italic=False):
  _, stroke, bfill = FAMILY[fam]
  dash = ' stroke-dasharray="6 4"' if dashed else ''
  out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
             f'fill="{fill or bfill}" stroke="{stroke}" stroke-width="1.4"{dash}/>')
  label(x + 12, y + 20, title, 13, 'bold', INK,
        style='italic' if italic else 'normal')
  for i, line in enumerate(lines):
    label(x + 12, y + 38 + 16 * i, line, 12, color=DIM)


def arrow(points, dashed=False, text=None, at=None, size=10.5, anchor='middle',
          rot=0):
  color = TARGET if dashed else ARROW
  dash = ' stroke-dasharray="5 4"' if dashed else ''
  head = 'headT' if dashed else 'head'
  pts = ' '.join(f'{x},{y}' for x, y in points)
  out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
             f'stroke-width="1.5"{dash} marker-end="url(#{head})"/>')
  if text:
    tx, ty = at
    label(tx, ty, text, size, color=color, anchor=anchor, rot=rot)


def sg(x, y):
  out.append(f'<rect x="{x - 13}" y="{y - 9}" width="26" height="18" rx="9" '
             f'fill="#e03131"/>')
  label(x, y + 4, 'sg', 11, 'bold', '#ffffff', 'middle')


def build():
  out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="{FONT}">')
  out.append('<defs>'
             f'<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" '
             f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker>'
             f'<marker id="headT" viewBox="0 0 10 10" refX="9" refY="5" '
             f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M0,0 L10,5 L0,10 z" fill="{TARGET}"/></marker></defs>')
  out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

  label(20, 34, 'Double-RSSM DreamerV3 + tech-tree potential  —  '
        'Craftax-Symbolic, size50m', 20, 'bold')
  label(20, 56, 'branch fix/spine-recipe-costs · observation-only map targets '
        '+ hindsight · imag_shift False · map_privileged False', 12.5, color=DIM)

  # ---------------------------------------------------------------- env
  region(20, 100, 430, 1000, 'env', 'Environment & wrapper')
  box(40, 140, 390, 92, 'env', 'Craftax-Symbolic-v1  (true state s)', [
      '48×48 map × 9 levels · mobs · light map',
      '43 actions · 67 achievements weighted 1/3/5/8 (Σ 226)',
      'player always spawns at map centre (24, 24)'])
  label(40, 262, 'embodied/envs/craftax.py — computed from s every step',
        11.5, 'bold', DIM)
  arrow([(390, 232), (390, 285)], text='s', at=(402, 264), anchor='start')
  box(40, 285, 390, 78, 'env', "obs['vector']  8268  — the only agent input", [
      '9×11×83 local view (blocks, items, mobs, light)',
      '+ 51 stats: inventory, vitals, xp, mana, spells, facing'])
  box(40, 378, 390, 62, 'prop', "obs['valid']  43  — PROPOSED", [
      '43 precondition flags, derived from obs contents'],
      dashed=True, italic=True)
  box(40, 455, 390, 172, 'phi', 'Tech-tree potential Φ — craftax_potential.py', [
      'Φ(s) = Σ_{spine} w_{a} · progress_{a}(s) + capability(s)',
      'progress = 1 if unlocked, else 0.25 × prereqs met',
      '13-rung spine: PLACE_TABLE (wood ≥ 2) → pickaxes',
      '   → COLLECT_STONE / COAL / IRON → COLLECT_DIAMOND',
      'capability: pickaxe 2.0, sword 1.2 per tier held',
      "F = γ·Φ(s′) − Φ(s)      γ = 0.997, scale 4",
      'unlocked rungs keep full weight: winning never lowers Φ'])
  arrow([(235, 627), (235, 648)], text='F', at=(246, 642), anchor='start')
  box(40, 648, 390, 78, 'phi', 'reward r  (what replay stores)', [
      'r = Σ new achievements × {1,3,5,8} + 0.1·Δhealth',
      '      + F      ← potential shaping'], fill='#fffaf0')
  box(40, 741, 390, 46, 'env', 'is_first · is_last · is_terminal', [])
  label(52, 778, 'timeout is not terminal — value bootstraps through it', 12,
        color=DIM)
  box(40, 802, 390, 150, 'map', 'RSSM-2 targets — training only, never inputs', [
      'map12 12×12×16  mosaic of own lit 9×11 windows',
      'mapknown 12×12  share of cell observed → loss weight',
      'mapseen 12×12  visit recency (decay 0.99)',
      'mappos 0..143  coarse cell (fixed spawn ⇒ dead-reckonable)',
      'unseen cells: no gradient; scored only by map_eval'],
      fill='#f6fcf8')
  box(40, 967, 390, 62, 'note', 'ablation switch', [
      'env.craftax.map_privileged True → full-map targets (old runs)'])
  for y, dashed in ((324, False), (409, True), (687, False), (764, False),
                    (877, False)):
    arrow([(430, y), (470, y)], dashed=dashed)

  # ------------------------------------------------------------- replay
  out.append('<rect x="470" y="100" width="100" height="1000" rx="12" '
             'fill="#f8f9fa" stroke="#495057" stroke-width="1.5"/>')
  for i, line in enumerate(['Replay', '', '4e5 steps', '', 'batch', '16 × 64',
                            '', 'train ratio', '512', '', '1.1M env', 'steps']):
    label(520, 130 + 20 * i, line, 15 if i == 0 else 12,
          'bold' if i == 0 else 'normal', INK if i == 0 else DIM, 'middle')

  # -------------------------------------------------------- world model
  region(600, 100, 520, 560, 'wm', 'World model — DreamerV3 RSSM-1')
  arrow([(570, 179), (620, 179)], text='obs, a, r', at=(595, 172), size=9.5)
  box(620, 140, 230, 78, 'wm', 'Encoder', [
      'symlog(vector [⊕ valid])',
      '→ MLP 3 × 512 → tokens e_{t}'])
  box(870, 150, 230, 58, 'wm', 'previous action a_{t−1}', ['one-hot, 43'])
  arrow([(735, 218), (735, 240)])
  arrow([(985, 208), (985, 240)])
  box(620, 240, 480, 150, 'wm', 'RSSM-1  (fast: every env step)', [
      'deter h_{t} 4096, block GRU (8 blocks):   h_{t} = f(h_{t−1}, z_{t−1}, a_{t−1})',
      'posterior  z_{t} ~ q(z | h_{t}, e_{t})     ← training and acting',
      'prior        z_{t} ~ p(z | h_{t})           ← imagination',
      'stoch z: 32 × 32 categorical, 1% unimix',
      'h = deterministic memory carried step to step;',
      'z = stochastic code for what this frame adds'])
  arrow([(860, 390), (860, 410)])
  box(620, 410, 480, 40, 'wm', 'feat = [h, z]   →   5120      (feat2tensor)',
      fill='#dce7ff')
  for x in (695, 860, 1025):
    arrow([(x, 450), (x, 470)])
  box(620, 470, 150, 78, 'wm', 'Decoder', ['MLP → vector', '[⊕ valid]'])
  box(785, 470, 150, 78, 'wm', 'Reward head', ['symexp two-hot', '255 bins'])
  box(950, 470, 150, 78, 'wm', 'Continue head', ['binary', '1 − terminal'])
  for x in (695, 860, 1025):
    arrow([(x, 548), (x, 565)], dashed=True)
  box(620, 565, 480, 78, 'wm', 'world-model loss', [
      'L_{dec} + L_{rew} (r includes F) + L_{con} + KL_{dyn} + KL_{rep}  (free nats 1)',
      'map kept OUT of rew/con — a map error must not corrupt imagined reward'],
      dashed=True)
  arrow([(570, 604), (620, 604)], dashed=True, text='r, term', at=(595, 597),
        size=9.5)

  # --------------------------------------------------------------- RSSM-2
  region(600, 680, 520, 420, 'map', 'RSSM-2 — slow spatial belief (mapmodel.py)')
  arrow([(620, 430), (608, 430), (608, 752), (620, 752)])
  sg(608, 670)
  box(620, 720, 480, 78, 'map', 'aggregate one tick = 8 env steps', [
      'mean sg(feat) 5120  ⊕  Σ action deltas (dy, dx)  ⊕  tick',
      'position from its OWN actions — no coordinates given'])
  arrow([(570, 785), (620, 785)], text='a', at=(595, 778), size=9.5)
  arrow([(860, 798), (860, 815)])
  box(620, 815, 480, 62, 'map', 'GRU   deter_{2} 1024   (hidden 512 × 2 layers)', [
      'steps once per tick — 8× slower than RSSM-1'])
  arrow([(860, 877), (860, 895)])
  box(620, 895, 480, 62, 'map', 'map decoder', [
      'map logits 12×12×16          position logits 144'])
  arrow([(860, 957), (860, 975)], dashed=True)
  box(620, 975, 480, 110, 'map', 'RSSM-2 loss', [
      'BCE weighted by mapknown — unseen cells weigh 0',
      'hindsight: each tick graded vs end-of-episode mosaic',
      'normalised by observed mass × 144 cells',
      '+ position cross-entropy vs mappos'], dashed=True)
  arrow([(570, 1030), (620, 1030)], dashed=True, text='targets', at=(595, 1023),
        size=9.5)

  # ---------------------------------------------------------- actor-critic
  region(1140, 100, 700, 560, 'ac', 'Actor–critic — trained in imagination')
  arrow([(1100, 430), (1130, 430), (1130, 171), (1160, 171)],
        text='posteriors', at=(1125, 300), rot=-90)
  box(1160, 140, 320, 62, 'ac', 'start states', [
      'every replay posterior (h, z): 16 × 64'])
  arrow([(1320, 202), (1320, 222)])
  box(1160, 222, 320, 110, 'ac', 'imagine H = 15 steps (RSSM-1 prior)', [
      'a_{t} ~ π(· | ainp_{t})',
      'h_{t+1} = f(h_{t}, z_{t}, a_{t});   z_{t+1} ~ p(z | h_{t+1})',
      'reward & continue heads read feat only'])
  box(1160, 352, 320, 94, 'phi', 'Φ inside the dream', [
      'the reward head learned r including F,',
      'so every imagined reward carries the ramp.',
      'The agent never reads Φ as an input.'])
  box(1500, 140, 320, 78, 'ac', 'ainp = actor2tensor(feat, mapfeat)', [
      'feat 5120 ⊕ mapfeat 1296 = 6416',
      'feeds π and V only — never rew / con'], fill='#ebe4ff')
  arrow([(1480, 240), (1490, 240), (1490, 190), (1500, 190)])
  arrow([(1660, 218), (1660, 238)])
  box(1500, 238, 320, 94, 'ac', 'Actor π — 43-way categorical', [
      'L = −log π(a) · sg(adv) − 3e-4 · H(π)',
      'adv = λ-return − V, normalised',
      'imagined rollouts sample from π'])
  arrow([(1500, 285), (1480, 285)], text='a', at=(1490, 279), size=9.5)
  arrow([(1480, 316), (1490, 316), (1490, 397), (1500, 397)])
  box(1500, 350, 320, 94, 'ac', 'Critic V + slow EMA target', [
      'λ-returns: λ = 0.95, horizon 333 (γ ≈ 0.997)',
      'reads the same ainp',
      'also fit on replay returns'])
  box(1160, 466, 660, 108, 'ac', 'acting in the real env — policy(), every step', [
      'obs → encoder → RSSM-1 posterior → feat',
      'feat → RSSM-2 window sum; every 8th step: GRU tick → new deter_{2} → mapfeat',
      'ainp → π → sample a_{t} → env        (no imagination while acting)'],
      fill='#f9f7ff')
  arrow([(1820, 520), (1872, 520), (1872, 80), (330, 80), (330, 140)],
        text='acting: a_{t} ~ π goes to the real env, one step at a time',
        at=(1100, 73), size=11.5)

  # ------------------------------------------------------ map feature
  region(1140, 680, 700, 420, 'map', 'Actor-side map feature — mapfeat (agent.py)')
  arrow([(1100, 926), (1130, 926), (1130, 714), (1660, 714), (1660, 732)],
        text='map & position logits', at=(1125, 820), rot=-90)
  box(1500, 732, 320, 62, 'map', 'σ(map logits)', ['12×12×16 belief'])
  arrow([(1660, 794), (1660, 810)])
  box(1500, 810, 320, 78, 'map', 'crop_egocentric', [
      '9×9×16 window centred on',
      'argmax(position logits)'])
  arrow([(1660, 888), (1660, 906)])
  box(1500, 906, 320, 78, 'map', 'flatten 1296 → sg → × gate', [
      'gate: learned scalar, initialised 0',
      '— opens only if the map pays'])
  arrow([(1820, 945), (1850, 945), (1850, 179), (1820, 179)],
        text='mapfeat 1296', at=(1862, 600), size=10, rot=-90)
  sg(1850, 860)
  box(1160, 732, 320, 126, 'note', 'in imagination (imag_shift False)', [
      'RSSM-2 does not step while dreaming:',
      'the crop is FROZEN at the imagination',
      'start and shared by all 15 steps.',
      'Sliding it made rollout and loss disagree',
      'and broke the policy gradient.'])
  box(1160, 876, 320, 108, 'note', 'gradient boundaries', [
      'sg(feat) in: map loss never reaches RSSM-1',
      'sg(crop) out: policy loss never reaches',
      '   RSSM-2 — only the gate learns from it'])

  # ------------------------------------------------------------- legend
  out.append('<rect x="20" y="1120" width="1860" height="190" rx="12" '
             'fill="#ffffff" stroke="#dee2e6" stroke-width="1.5"/>')
  label(40, 1146, 'Legend', 14, 'bold')
  arrow([(40, 1172), (100, 1172)])
  label(112, 1176, 'data / forward pass')
  arrow([(40, 1200), (100, 1200)], dashed=True)
  label(112, 1204, 'training target → loss')
  sg(70, 1228)
  label(112, 1232, 'stop-gradient')
  out.append('<rect x="44" y="1250" width="52" height="22" rx="5" '
             'fill="#fafafa" stroke="#adb5bd" stroke-dasharray="6 4"/>')
  label(112, 1266, 'proposed, not built')

  label(360, 1146, 'Colour', 14, 'bold')
  for i, (fam, text) in enumerate([
      ('env', 'environment & wrapper'), ('phi', 'potential Φ / reward'),
      ('wm', 'RSSM-1 world model (DreamerV3)'), ('map', 'RSSM-2 map model'),
      ('ac', 'actor–critic')]):
    fill, stroke, _ = FAMILY[fam]
    y = 1162 + 26 * i
    out.append(f'<rect x="360" y="{y}" width="30" height="16" rx="4" '
               f'fill="{fill}" stroke="{stroke}"/>')
    label(400, y + 13, text)

  label(700, 1146, 'What trains what', 14, 'bold')
  for i, line in enumerate([
      'RSSM-1, encoder, decoder, reward, continue  ←  world-model loss on replay',
      'RSSM-2  ←  map BCE + position CE only (sg on its input)',
      'π, V  ←  imagined λ-returns; the gate is the only map parameter they move',
      'Φ  ←  nothing: a fixed function of state, reaching the agent only via r']):
    label(700, 1172 + 24 * i, line, 12.5, color=DIM)

  label(1290, 1146, "Where 'valid' would go (proposed)", 14, 'bold')
  for i, line in enumerate([
      'a new obs key, concatenated with vector at the encoder',
      '→ RSSM-1 latent; the decoder must reconstruct it, so every',
      'imagined step carries it — no extra predictor needed.',
      'Not via actor2tensor: validity changes step to step, and a',
      'frozen, map-style input would be stale inside the dream.']):
    label(1290, 1172 + 24 * i, line, 12.5, color=DIM)

  out.append('</svg>')
  return '\n'.join(out)


def browser():
  for p in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Google\Chrome\Application\chrome.exe'):
    if pathlib.Path(p).exists():
      return p
  return shutil.which('chromium') or shutil.which('google-chrome')


def main():
  docs = pathlib.Path(__file__).resolve().parent.parent / 'docs'
  svg = docs / 'architecture.svg'
  svg.write_text(build(), encoding='utf-8')
  print('wrote', svg)

  exe = browser()
  if not exe:
    print('no Chromium browser found; skipping PDF')
    return
  # Print through a wrapper page sized to the drawing, so the PDF is one page
  # at the figure's own aspect ratio instead of a letter page with margins.
  wrap = docs / '_architecture_print.html'
  wrap.write_text(
      f'<html><head><style>@page {{ size: {W}px {H}px; margin: 0 }}'
      f'html, body {{ margin: 0 }}</style></head><body>'
      f'<img src="architecture.svg" width="{W}" height="{H}"></body></html>',
      encoding='utf-8')
  pdf = docs / 'architecture.pdf'
  subprocess.run([exe, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                  f'--print-to-pdf={pdf}', wrap.as_uri()],
                 check=True, capture_output=True)
  wrap.unlink()
  print('wrote', pdf)


if __name__ == '__main__':
  main()
