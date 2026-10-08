"""Genome storage without pickle: raw bytes for arrays, JSON for everything else."""

import json
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np

from auxein.backend import Array, Backend, HostArray, is_array
from auxein.core import Batch

_HOST = Backend()

G = TypeVar("G")


class GenomeEncodingError(TypeError):
    """A genome cannot be stored yet."""


@dataclass(frozen=True)
class EncodedGenome:
    kind: str  # "array" or "json"
    data: bytes
    dtype: str | None = None
    shape: str | None = None  # JSON list


def encode_array(host: HostArray) -> EncodedGenome:
    """An array genome (already on the host) as raw C-order bytes plus its dtype and shape."""
    contiguous = np.ascontiguousarray(host)
    return EncodedGenome("array", contiguous.tobytes(), str(contiguous.dtype), json.dumps(list(contiguous.shape)))


def encode_genome(genome: object) -> EncodedGenome:
    """A genome as bytes: an array of any supported backend as raw bytes, anything else as JSON if it can be."""
    if is_array(genome):
        return encode_array(_HOST.to_numpy(genome))
    try:
        return EncodedGenome("json", json.dumps(genome).encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise GenomeEncodingError(
            f"a genome of type {type(genome).__name__} cannot be recorded: it is neither an array nor JSON-serialisable ({error}). "
            "Structured genomes get proper storage in a later step; until then use array or JSON-serialisable genomes, "
            "or run without run_dir."
        ) from error


def decode_genome(kind: str, data: bytes, dtype: str | None, shape: str | None) -> Array | object:
    """The inverse of `encode_genome`: arrays come back as read-only numpy arrays, JSON as plain Python values."""
    if kind == "array":
        assert dtype is not None and shape is not None
        return np.ndarray(shape=tuple(cast("list[int]", json.loads(shape))), dtype=np.dtype(dtype), buffer=data)
    if kind == "json":
        return json.loads(data.decode("utf-8"))
    raise ValueError(f"unknown genome kind {kind!r}")


def encode_batch(batch: Batch[G]) -> list[EncodedGenome]:
    """The genomes of a batch as bytes, in ask order. An array-backed batch makes one device-to-host copy for all of it.

    The recorder stores these; replay compares them with the recorded ones, which is what "byte-identical genome" means.
    """
    genomes = batch.as_array()
    if genomes is not None:
        host = _HOST.to_numpy(genomes)
        return [encode_array(host[i]) for i in range(host.shape[0])]
    return [encode_genome(c.genome) for c in batch.candidates]
