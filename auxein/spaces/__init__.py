"""Search spaces (design doc §4.5): the genomes a problem admits."""

from auxein.spaces.box import Box
from auxein.spaces.codec import CanonicalEncodingError, GenomeCodec, StructuredSpace, canonical_json, codec_of
from auxein.spaces.sequence import SequenceCodec, SequenceSpace
from auxein.spaces.space import Space

__all__ = [
    "Box",
    "CanonicalEncodingError",
    "GenomeCodec",
    "SequenceCodec",
    "SequenceSpace",
    "Space",
    "StructuredSpace",
    "canonical_json",
    "codec_of",
]
