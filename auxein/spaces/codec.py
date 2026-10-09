"""Codecs: how a structured genome becomes bytes, and back (design doc §4.1, §4.5).

A structured search space provides a codec: genome to a JSON-serialisable value, and back. The **canonical encoding** is the
JSON of that value with sorted keys, no insignificant whitespace, only string keys and no non-finite numbers, so that equal
genomes always give identical bytes. That is what recording, replay's byte-identical check, the genome store's hashes and
checkpoints rely on.
"""

import json
from typing import Protocol, TypeVar, cast

G = TypeVar("G")

JsonValue = object
"""A value `json` can encode: None, bool, int, float (finite), str, and lists and dicts (with str keys) of those."""


class CanonicalEncodingError(TypeError):
    """A genome encodes to something that is not canonical JSON: a type JSON does not have, a non-finite number, a non-string key."""


def _check_keys(value: JsonValue) -> None:
    """Dict keys must be strings: `json` would turn `1` into `"1"`, and two different genomes would encode alike."""
    if isinstance(value, dict):
        for key, item in cast("dict[object, object]", value).items():
            if not isinstance(key, str):
                raise CanonicalEncodingError(f"a genome must encode to a JSON value, but a dict has the non-string key {key!r}")
            _check_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in cast("list[object]", value):
            _check_keys(item)


def canonical_json(value: JsonValue) -> bytes:
    """The canonical bytes of a JSON-serialisable value: UTF-8, keys sorted, separators `,` and `:` with no whitespace.

    Values keep their JSON type: `1` and `1.0` are different encodings, as are `0.0` and `-0.0`, and floats are written by
    Python's shortest round-trip `repr`, so a float survives the round trip exactly. NaN and infinities are refused (JSON has
    no such numbers, and NaN is not equal to itself), as are non-string dict keys and anything else `json` cannot encode (a
    numpy `float32` or `int64`, bytes, a set): the error says what was wrong. A numpy `float64` is a Python float and encodes
    as one. A genome that is a tuple encodes as a list, so a codec must decode it back to a tuple.
    """
    _check_keys(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CanonicalEncodingError(
            f"a genome must encode to a JSON value (None, bool, int, finite float, str, lists and dicts with str keys of those): {error}"
        ) from error


class GenomeCodec(Protocol[G]):
    """Converts genomes of a structured space to JSON-serialisable values and back. `decode(encode(g)) == g` for every genome."""

    def encode(self, genome: G) -> JsonValue: ...

    def decode(self, value: JsonValue) -> G: ...


class StructuredSpace(Protocol[G]):
    """A search space whose genomes are not arrays: besides sampling and membership, it has a `codec` and a `describe()`."""

    @property
    def codec(self) -> GenomeCodec[G]: ...

    def describe(self) -> dict[str, object]: ...


def codec_of(space: object) -> "GenomeCodec[object] | None":
    """The codec of a space, or None for spaces that have none (arrays, plain JSON-serialisable genomes)."""
    codec = getattr(space, "codec", None)
    return codec if codec is not None and hasattr(codec, "encode") and hasattr(codec, "decode") else None
