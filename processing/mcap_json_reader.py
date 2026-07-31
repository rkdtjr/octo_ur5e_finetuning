"""Minimal, dependency-light MCAP reader for JSON stored in ROS 2 std_msgs/String.

The collector writes ``/octo_collector/robot_state`` as a serialized
``std_msgs/msg/String`` inside an MCAP file.  On the robot workstation the same
file can be read with ``rosbag2_py``.  This module also provides a portable
fallback so offline processing does not require a full ROS installation.

The fallback intentionally supports only the subset used by this project:
MCAP channel/message/chunk records, CDR-encoded ``std_msgs/String``, and
uncompressed, zstd, or lz4 chunks.  It is not a general ROS deserializer.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import ctypes.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Iterator

_MAGIC = b"\x89MCAP0\r\n"


class McapReadError(RuntimeError):
    """Raised when the project-specific MCAP fallback cannot decode a file."""


@dataclass(frozen=True)
class _Channel:
    topic: str
    message_encoding: str


def _u16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def _u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _u64(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", data, offset)[0], offset + 8


def _string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = _u32(data, offset)
    end = offset + length
    if end > len(data):
        raise McapReadError("truncated MCAP string")
    return data[offset:end].decode("utf-8"), end


def _records(data: bytes) -> Iterator[tuple[int, bytes]]:
    offset = 0
    while offset < len(data):
        if offset + 9 > len(data):
            raise McapReadError("truncated MCAP record header")
        opcode = data[offset]
        length = struct.unpack_from("<Q", data, offset + 1)[0]
        start = offset + 9
        end = start + length
        if end > len(data):
            raise McapReadError("truncated MCAP record body")
        yield opcode, data[start:end]
        offset = end


def _decompress_zstd_ctypes(payload: bytes, uncompressed_size: int) -> bytes:
    library_name = ctypes.util.find_library("zstd")
    if not library_name:
        raise McapReadError("zstd chunk found but libzstd is unavailable")
    library = ctypes.CDLL(library_name)
    library.ZSTD_decompress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    library.ZSTD_decompress.restype = ctypes.c_size_t
    library.ZSTD_isError.argtypes = [ctypes.c_size_t]
    library.ZSTD_isError.restype = ctypes.c_uint
    library.ZSTD_getErrorName.argtypes = [ctypes.c_size_t]
    library.ZSTD_getErrorName.restype = ctypes.c_char_p

    destination = ctypes.create_string_buffer(uncompressed_size)
    source = ctypes.create_string_buffer(payload)
    result = library.ZSTD_decompress(
        destination,
        uncompressed_size,
        source,
        len(payload),
    )
    if library.ZSTD_isError(result):
        name = library.ZSTD_getErrorName(result).decode("utf-8", errors="replace")
        raise McapReadError(f"zstd decompression failed: {name}")
    return destination.raw[:result]


def _decompress(compression: str, payload: bytes, uncompressed_size: int) -> bytes:
    if compression == "":
        return payload
    if compression == "zstd":
        try:
            import zstandard  # type: ignore

            return zstandard.ZstdDecompressor().decompress(
                payload,
                max_output_size=uncompressed_size,
            )
        except ImportError:
            pass
        try:
            return _decompress_zstd_ctypes(payload, uncompressed_size)
        except McapReadError:
            executable = shutil.which("zstd")
            if executable is None:
                raise
            completed = subprocess.run(
                [executable, "--quiet", "--decompress", "--stdout"],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise McapReadError(
                    "zstd decompression failed: "
                    + completed.stderr.decode("utf-8", errors="replace")
                )
            return completed.stdout
    if compression == "lz4":
        try:
            import lz4.frame  # type: ignore
        except ImportError as exc:
            raise McapReadError("lz4 chunk found but python-lz4 is unavailable") from exc
        return lz4.frame.decompress(payload)
    raise McapReadError(f"unsupported MCAP compression: {compression!r}")


def _decode_cdr_string(data: bytes) -> str:
    if len(data) < 8:
        raise McapReadError("truncated CDR String")
    # ROS 2 SerializedPayload encapsulation. 0x0001/0x0003 are little-endian.
    little_endian = data[:2] in (b"\x00\x01", b"\x00\x03")
    endian = "<" if little_endian else ">"
    length = struct.unpack_from(endian + "I", data, 4)[0]
    if length == 0 or 8 + length > len(data):
        raise McapReadError("invalid CDR String length")
    payload = data[8 : 8 + length]
    if payload[-1] == 0:
        payload = payload[:-1]
    return payload.decode("utf-8")


def _read_portable(path: Path, topic: str) -> list[dict]:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != _MAGIC or raw[-8:] != _MAGIC:
        raise McapReadError(f"not a complete MCAP file: {path}")

    channels: dict[int, _Channel] = {}
    result: list[dict] = []

    def consume(record_bytes: bytes) -> None:
        for opcode, body in _records(record_bytes):
            if opcode == 0x04:  # Channel
                offset = 0
                channel_id, offset = _u16(body, offset)
                _, offset = _u16(body, offset)  # schema id
                channel_topic, offset = _string(body, offset)
                encoding, offset = _string(body, offset)
                channels[channel_id] = _Channel(channel_topic, encoding)
            elif opcode == 0x05:  # Message
                offset = 0
                channel_id, offset = _u16(body, offset)
                _, offset = _u32(body, offset)  # sequence
                _, offset = _u64(body, offset)  # log time
                _, offset = _u64(body, offset)  # publish time
                channel = channels.get(channel_id)
                if channel is None or channel.topic != topic:
                    continue
                if channel.message_encoding != "cdr":
                    raise McapReadError(
                        f"topic {topic!r} uses unsupported encoding "
                        f"{channel.message_encoding!r}"
                    )
                result.append(json.loads(_decode_cdr_string(body[offset:])))
            elif opcode == 0x06:  # Chunk
                offset = 0
                _, offset = _u64(body, offset)  # message start
                _, offset = _u64(body, offset)  # message end
                uncompressed_size, offset = _u64(body, offset)
                _, offset = _u32(body, offset)  # uncompressed CRC
                compression, offset = _string(body, offset)
                compressed_size, offset = _u64(body, offset)
                end = offset + compressed_size
                if end > len(body):
                    raise McapReadError("truncated MCAP chunk payload")
                consume(
                    _decompress(
                        compression,
                        body[offset:end],
                        int(uncompressed_size),
                    )
                )

    # Exclude both magic sequences. Summary records may duplicate channels; this
    # is harmless because identical channel IDs must have identical definitions.
    consume(raw[8:-8])
    return result


def _read_with_rosbag2(path: Path, topic: str) -> list[dict]:
    import rosbag2_py  # type: ignore
    from rclpy.serialization import deserialize_message  # type: ignore
    from std_msgs.msg import String  # type: ignore

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    result: list[dict] = []
    while reader.has_next():
        current_topic, data, _ = reader.read_next()
        if current_topic == topic:
            result.append(json.loads(deserialize_message(data, String).data))
    return result


def read_json_string_topic(
    path: str | Path,
    topic: str = "/octo_collector/robot_state",
) -> list[dict]:
    """Read JSON objects published as ROS 2 ``std_msgs/String`` from MCAP.

    Uses the official ROS reader when available, otherwise the portable MCAP
    subset reader above.  An empty topic is treated as an error rather than as a
    valid episode, preventing silent all-invalid synchronization output.
    """

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        states = _read_with_rosbag2(resolved, topic)
    except (ImportError, ModuleNotFoundError):
        states = _read_portable(resolved, topic)
    if not states:
        raise McapReadError(f"topic {topic!r} has no messages in {resolved}")
    return states
