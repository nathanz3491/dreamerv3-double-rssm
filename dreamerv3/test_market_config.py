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
