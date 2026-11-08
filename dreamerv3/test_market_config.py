"""The market agent's config entries: everything is off by default.

Pure yaml, no JAX.

Run: python -m pytest dreamerv3/test_market_config.py
"""

import pathlib

import pytest

yaml = pytest.importorskip('yaml')


def _defaults():
  path = pathlib.Path(__file__).with_name('configs.yaml')
  return yaml.safe_load(path.read_text(encoding='utf-8'))['defaults']


def test_the_market_is_off_by_default():
  cfg = _defaults()['agent']
  assert cfg['market']['enabled'] is False and cfg['market']['shift'] is False
  assert cfg['market']['price0'] > 0
  assert 'qhead' in cfg['loss_scales']


def test_the_quota_is_off_by_default():
  assert _defaults()['env']['craftax']['quota']['enabled'] is False


def test_market_constants_are_sane():
  cfg, full = _defaults()['agent'], _defaults()
  mk = cfg['market']
  assert mk['window'] < full['batch_length']   # report batches clamp it
  assert mk['sigma'] > 0 and mk['bound'] > 0 and mk['dead'] >= 0
  assert mk['eps_s'] > 0 and mk['eps_d'] > 0 and mk['k'] >= 1
  assert cfg['valid']['mask'] is False        # the mask counts as cheating
  assert 'mkt' in cfg['loss_scales']
