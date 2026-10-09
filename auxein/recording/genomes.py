"""Genome storage without pickle: raw bytes for arrays, canonical JSON for structured genomes, plain JSON otherwise."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import numpy as np

from auxein.backend import Array, Backend, HostArray, is_array
from auxein.core import Batch
from auxein.spaces.codec import CanonicalEncodingError, GenomeCodec, canonical_json

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


def encode_genome(genome: object, codec: GenomeCodec[Any] | None = None) -> EncodedGenome:
    """A genome as bytes.

    An array of any supported backend is raw bytes. With a codec (a structured space, design doc §4.1) the genome is the
    **canonical JSON** of its encoding, so equal genomes give identical bytes. Without one, a JSON-serialisable value is JSON.
    """
    if is_array(genome):
        return encode_array(_HOST.to_numpy(genome))
    if codec is not None:
        try:
            return EncodedGenome("json", canonical_json(codec.encode(genome)))
        except CanonicalEncodingError as error:
            raise GenomeEncodingError(
                f"the codec encoded a genome of type {type(genome).__name__} to something that is not canonical JSON: {error}"
            ) from error
    try:
        return EncodedGenome("json", json.dumps(genome).encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise GenomeEncodingError(
            f"a genome of type {type(genome).__name__} cannot be recorded: it is neither an array nor JSON-serialisable ({error}). "
            "Give its search space a codec (design doc §4.1), use array or JSON-serialisable genomes, or run without run_dir."
        ) from error


def decode_genome(kind: str, data: bytes, dtype: str | None, shape: str | None, codec: GenomeCodec[Any] | None = None) -> Array | object:
    """The inverse of `encode_genome`: arrays come back as read-only numpy arrays; JSON as plain Python values, or, when the
    space's codec is known, as the genomes it decodes them to."""
    if kind == "array":
        assert dtype is not None and shape is not None
        return np.ndarray(shape=tuple(cast("list[int]", json.loads(shape))), dtype=np.dtype(dtype), buffer=data)
    if kind == "json":
        value = json.loads(data.decode("utf-8"))
        return value if codec is None else codec.decode(value)
    raise ValueError(f"unknown genome kind {kind!r}")


def genome_hash(data: bytes) -> str:
    """The key of an encoded genome in the genome store (design doc §10.3): the SHA-256 of its bytes, in hex."""
    return hashlib.sha256(data).hexdigest()


def encode_batch(batch: Batch[G], codec: GenomeCodec[Any] | None = None) -> list[EncodedGenome]:
    """The genomes of a batch as bytes, in ask order. An array-backed batch makes one device-to-host copy for all of it.

    The recorder stores these; replay compares them with the recorded ones, which is what "byte-identical genome" means.
    """
    genomes = batch.as_array()
    if genomes is not None:
        host = _HOST.to_numpy(genomes)
        return [encode_array(host[i]) for i in range(host.shape[0])]
    return [encode_genome(c.genome, codec) for c in batch.candidates]
