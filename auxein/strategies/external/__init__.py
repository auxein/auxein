"""Strategies that wrap an external algorithm (design doc §3.4): their purpose is to prove that such algorithms fit the contract."""

from auxein.strategies.external.pycma import PycmaStrategy

__all__ = ["PycmaStrategy"]
