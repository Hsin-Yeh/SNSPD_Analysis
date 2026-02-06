#!/usr/bin/env python3
"""
Inject a histogram from a source .phu file into a target .phu file by
overwriting a chosen curve in the target and re-labeling its block ID.

Default use case:
- Take block 73 from the 0nW file
- Insert it as block 0 into the 73mV file
- Write to a separate output copy for verification
"""

import argparse
import struct
from pathlib import Path


TAG_TYPES_8B = {
    0xFFFF0001,  # tyEmpty8
    0x00000008,  # tyBool8
    0x10000008,  # tyInt8
    0x11000008,  # tyBitSet64
    0x12000008,  # tyColor8
    0x20000008,  # tyFloat8
    0x21000008,  # tyTDateTime
}


def read_phu_header_with_offsets(filepath):
    """Read PHU header and capture offsets for tag values (for in-place edits)."""
    header = {}
    offsets = {}

    with open(filepath, 'rb') as f:
        # Magic + version
        f.read(8)
        f.read(8)

        while True:
            tag_ident = f.read(32).decode('ascii', errors='ignore').rstrip('\0')
            if not tag_ident:
                break

            tag_idx = struct.unpack('<i', f.read(4))[0]
            tag_type = struct.unpack('<i', f.read(4))[0]

            value_offset = f.tell()

            if tag_type in TAG_TYPES_8B:
                tag_value = struct.unpack('<q', f.read(8))[0]
            elif tag_type == 0x2001FFFF:  # tyFloat8Array
                array_size = struct.unpack('<q', f.read(8))[0]
                tag_value = struct.unpack(f'<{array_size}d', f.read(8 * array_size))
            elif tag_type == 0x4001FFFF:  # tyAnsiString
                string_size = struct.unpack('<q', f.read(8))[0]
                tag_value = f.read(string_size).decode('ascii', errors='ignore').rstrip('\0')
            elif tag_type == 0x4002FFFF:  # tyWideString
                string_size = struct.unpack('<q', f.read(8))[0]
                tag_value = f.read(string_size * 2).decode('utf-16-le', errors='ignore').rstrip('\0')
            elif tag_type == 0xFFFFFFFF:  # tyBinaryBlob
                blob_size = struct.unpack('<q', f.read(8))[0]
                tag_value = f"<binary data: {blob_size} bytes>"
                f.read(blob_size)
            else:
                tag_value = f"<unknown type 0x{tag_type:08X}>"
                f.read(8)

            if tag_idx == -1:
                header[tag_ident] = tag_value
            else:
                header.setdefault(tag_ident, {})[tag_idx] = tag_value

            offsets[(tag_ident, tag_idx)] = (tag_type, value_offset)

            if tag_ident == 'Header_End':
                break

    return header, offsets


def get_curve_index_for_block(header, block_id):
    curve_index = header.get('HistResDscr_CurveIndex', {})
    for idx, bid in curve_index.items():
        if bid == block_id:
            return idx
    return None


def read_histogram(filepath, header, curve_index):
    data_offsets = header.get('HistResDscr_DataOffset', {})
    bins = header.get('HistResDscr_HistogramBins', {})

    if curve_index not in data_offsets or curve_index not in bins:
        raise ValueError(f"Missing data offset or bins for curve index {curve_index}")

    offset = data_offsets[curve_index]
    num_bins = bins[curve_index]

    with open(filepath, 'rb') as f:
        f.seek(offset)
        data = struct.unpack(f'<{num_bins}I', f.read(4 * num_bins))

    return list(data), offset, num_bins


def write_histogram(filepath, offset, data):
    packed = struct.pack(f'<{len(data)}I', *data)
    with open(filepath, 'r+b') as f:
        f.seek(offset)
        f.write(packed)


def update_curve_index(filepath, offsets, curve_index, new_block_id):
    key = ('HistResDscr_CurveIndex', curve_index)
    if key not in offsets:
        raise ValueError(f"No header tag found for curve index {curve_index}")

    tag_type, value_offset = offsets[key]
    if tag_type != 0x10000008:  # tyInt8
        raise ValueError(f"Unexpected tag type for curve index: 0x{tag_type:08X}")

    with open(filepath, 'r+b') as f:
        f.seek(value_offset)
        f.write(struct.pack('<q', int(new_block_id)))


def main():
    parser = argparse.ArgumentParser(description="Inject a block histogram into a target PHU file.")
    parser.add_argument('--source', default='/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_0nW_20260205_0518.phu')
    parser.add_argument('--target', default='/Users/ya/SNSPD_rawdata/SMSPD_3/TCSPC/SMSPD_3_2-7_500kHz_73mV_20260205_1213.phu')
    parser.add_argument('--source-block', type=int, default=73, help='Block ID in source file to copy')
    parser.add_argument('--target-block', type=int, default=0, help='Block ID to set in target file')
    parser.add_argument('--target-curve-index', type=int, default=None, help='Curve index to overwrite in target file')
    parser.add_argument('--output', default=None, help='Output PHU file path')
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    output_path = Path(args.output) if args.output else target_path.with_name(
        f"{target_path.stem}_with_block0_from_{args.source_block}.phu"
    )

    # Read headers
    src_header, _ = read_phu_header_with_offsets(source_path)
    tgt_header, tgt_offsets = read_phu_header_with_offsets(target_path)

    # Locate source curve for block ID
    src_curve_idx = get_curve_index_for_block(src_header, args.source_block)
    if src_curve_idx is None:
        raise ValueError(f"Block {args.source_block} not found in source file")

    # Choose target curve index to overwrite
    if args.target_curve_index is not None:
        tgt_curve_idx = args.target_curve_index
    else:
        tgt_curve_idx = max(tgt_header.get('HistResDscr_CurveIndex', {}).keys())

    # Read source histogram
    src_hist, _, src_bins = read_histogram(source_path, src_header, src_curve_idx)

    # Read target histogram metadata
    _, tgt_offset, tgt_bins = read_histogram(target_path, tgt_header, tgt_curve_idx)

    if src_bins != tgt_bins:
        raise ValueError(
            f"Bin count mismatch: source {src_bins} vs target {tgt_bins}. "
            "Cannot overwrite safely."
        )

    # Copy target to output
    output_path.write_bytes(target_path.read_bytes())

    # Overwrite histogram and update curve index mapping
    write_histogram(output_path, tgt_offset, src_hist)
    update_curve_index(output_path, tgt_offsets, tgt_curve_idx, args.target_block)

    original_block = tgt_header.get('HistResDscr_CurveIndex', {}).get(tgt_curve_idx, None)
    print("=== Injection Summary ===")
    print(f"Source file: {source_path}")
    print(f"Target file: {target_path}")
    print(f"Output file: {output_path}")
    print(f"Source block ID: {args.source_block} (curve index {src_curve_idx})")
    print(f"Target curve index overwritten: {tgt_curve_idx} (was block ID {original_block})")
    print(f"Target block ID set to: {args.target_block}")
    print("Note: This replaces one curve in the target file with the source histogram.")


if __name__ == '__main__':
    main()