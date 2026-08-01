#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ncnn "Convolution 1d input compatibility path" layers to InnerProduct + Reshape.

Background
----------
Since ncnn commit f6f734f4 (#6751, 2026-06), Convolution layers that receive a
flattened (dims==1) input emit:
    "Convolution 1d input compatibility path is deprecated and will be removed,
     please replace this layer with InnerProduct"
and the int8 path of that compatibility branch can crash (access violation) on
dims==1 inputs.  The recommended replacement is InnerProduct (+ a Reshape to keep
the original output shape), e.g.

    Convolution  name 1 1 in out 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384 9=1
becomes
    InnerProduct name 1 1 in tmp 0=64 1=1 2=16384 9=1
    Reshape      name_shape 1 1 tmp out 0=1 1=1 2=64

This script finds every 1x1 Convolution whose input is produced by
  * a global Pooling layer (4=1),
  * a Reduction layer, or
  * another converted Convolution (direct chain),
and rewrites the .param accordingly.  For fp32 models the .bin is copied
unchanged (the 1x1 conv weight layout [out][in] equals the InnerProduct layout).

For int8 models the .bin must be rewritten: a Convolution with int8_scale_term>100
stores an extra per-layer top_blob_int8_scales float that InnerProduct does not
read (and InnerProduct always outputs fp32).  The script therefore changes the
param's int8_scale_term from 102/101 to 2/1 and drops that 4-byte scale from the
bin, keeping every later layer byte-aligned.

Usage
-----
    python fix_conv1d.py <in.param> <in.bin> <out.param> <out.bin>
"""

import struct
import sys


def align4(n):
    return (n + 3) & ~3


def parse_param(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0].strip() == "7767517", "not an ncnn param file"
    n_layer, n_blob = map(int, lines[1].split())
    layers = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        layers.append(parts)
    assert len(layers) == n_layer, f"layer count mismatch {len(layers)} != {n_layer}"
    return n_layer, n_blob, lines, layers


def param_dict(parts):
    d = {}
    for tok in parts[5:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[int(k)] = v
    return d


def find_targets(layers):
    """Return a boolean mask of Convolution layers to convert."""
    producer = {}
    for i, parts in enumerate(layers):
        num_in = int(parts[2])
        num_out = int(parts[3])
        for o in parts[4 + num_in:4 + num_in + num_out]:
            producer[o] = i

    is_target = [False] * len(layers)
    changed = True
    while changed:
        changed = False
        for i, parts in enumerate(layers):
            if is_target[i]:
                continue
            if parts[0] != "Convolution":
                continue
            d = param_dict(parts)
            kw = d.get(1)
            kh = d.get(11, kw)
            if kw != "1" or kh != "1":
                continue  # not a 1x1 convolution
            in_blob = parts[4]
            prod = producer.get(in_blob)
            if prod is None:
                continue
            ptyp = layers[prod][0]
            reason = None
            if ptyp == "Pooling" and param_dict(layers[prod]).get(4) == "1":
                reason = "GAP"          # global average pooling -> flatten
            elif ptyp == "Reduction":
                reason = "RED"          # mean-reduced feature vector
            elif is_target[prod]:
                reason = "CHAIN"        # follows another converted conv
            if reason:
                is_target[i] = True
                changed = True
    return is_target


class BinReader:
    """Sequential reader over an ncnn .bin that mirrors ModelBinFromDataReader."""

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        assert self.pos + n <= len(self.data), (
            f"bin read past EOF at {self.pos}+{n} of {len(self.data)}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def load_type0(self, w):
        """Return (tag, payload_bytes) consumed for mb.load(w, 0)."""
        start = self.pos
        tag = struct.unpack("<I", self.read(4))[0]
        if tag == 0x01306B47:       # fp16
            n = align4(2 * w)
        elif tag == 0x000D4B38:     # int8
            n = align4(w)
        elif tag == 0x0002C056:     # float with scaling
            n = 4 * w
        else:
            f0 = tag & 0xFF
            f1 = (tag >> 8) & 0xFF
            f2 = (tag >> 16) & 0xFF
            f3 = (tag >> 24) & 0xFF
            if f0 + f1 + f2 + f3 != 0:
                n = 256 * 4 + align4(w)   # quantized lookup table
            else:
                n = 4 * w                 # raw fp32
        payload = self.read(n)
        return tag, payload

    def load_type1(self, w):
        """Return payload bytes consumed for mb.load(w, 1) (bias/scales)."""
        return self.read(4 * w)


def copy_bin(src, dst, layers, targets):
    """Walk the bin in layer order, copying data; drop int8 top scales of targets."""
    rd = BinReader(src)
    out = bytearray()

    def emit(payload):
        out.extend(payload)

    for i, parts in enumerate(layers):
        typ = parts[0]
        d = param_dict(parts)

        def consume_weight():
            tag, payload = rd.load_type0(int(d[6]))
            emit(struct.pack("<I", tag) + payload)

        def consume_weight_ip():
            tag, payload = rd.load_type0(int(d[2]))
            emit(struct.pack("<I", tag) + payload)

        def consume_bias(w):
            emit(rd.load_type1(w))

        if typ == "Convolution":
            consume_weight()
            if d.get(5, "0") != "0":
                consume_bias(int(d[0]))
            i8 = int(d.get(8, "0"))
            if i8 != 0:
                consume_bias(int(d[0]))     # weight_data_int8_scales
                consume_bias(1)             # bottom_blob_int8_scales
            if i8 > 100:
                _ = rd.load_type1(1)        # top_blob_int8_scales
                if not targets[i]:
                    emit(_)
        elif typ == "ConvolutionDepthWise":
            consume_weight()
            if d.get(5, "0") != "0":
                consume_bias(int(d[0]))
            i8 = int(d.get(8, "0"))
            group = int(d.get(7, d[0]))
            if i8 in (1, 101):
                consume_bias(group)
                consume_bias(1)
            elif i8 in (2, 102):
                consume_bias(1)
                consume_bias(1)
            if i8 > 100:
                _ = rd.load_type1(1)
                if not targets[i]:
                    emit(_)
        elif typ == "Deconvolution":
            consume_weight()
            if d.get(5, "0") != "0":
                consume_bias(int(d[0]))
        elif typ == "InnerProduct":
            consume_weight_ip()
            if d.get(1, "0") != "0":
                consume_bias(int(d[0]))
            i8 = int(d.get(8, "0"))
            if i8 != 0:
                consume_bias(int(d[0]))
                consume_bias(1)
        else:
            # no weights
            pass

    # remaining bytes: file may be padded; ensure nothing meaningful is left
    leftover = len(src) - rd.pos
    if leftover != 0:
        # tolerate trailing padding but flag unexpected leftovers
        print(f"  note: {leftover} trailing bytes unconsumed", file=sys.stderr)
        out.extend(src[rd.pos:])
    dst.write(bytes(out))


def rewrite_param(in_lines, layers, targets):
    """Produce new param lines with InnerProduct+Reshape replacements.

    Unchanged layers keep their original text (including column alignment);
    new InnerProduct/Reshape lines are aligned to the same column widths.
    """
    n_new_blobs = sum(1 for t in targets if t)
    max_blob = -1
    for parts in layers:
        num_in = int(parts[2])
        num_out = int(parts[3])
        for b in parts[4:4 + num_in + num_out]:
            if b.isdigit():
                max_blob = max(max_blob, int(b))
    next_blob = max_blob + 1

    # original text of each layer line (in the same order as layers)
    orig_text = [l for l in in_lines[2:] if l.strip()]

    # column widths used by the original file for type/name (derived from text)
    w_type = 1
    w_name = 1
    for line in orig_text:
        t = line.split()
        if len(t) < 3:
            continue
        s_name = line.index(t[1])
        s_num = line.index(t[2])
        w_type = max(w_type, s_name)
        w_name = max(w_name, s_num - s_name)

    def fmt(parts):
        head = parts[0].ljust(w_type) + parts[1].ljust(w_name)
        return head + " " + " ".join(parts[2:])

    out_lines = [in_lines[0], in_lines[1]]
    n_layer, n_blob = map(int, in_lines[1].split())
    n_layer_new = n_layer + n_new_blobs
    n_blob_new = n_blob + n_new_blobs
    out_lines[1] = f"{n_layer_new} {n_blob_new}"

    for i, parts in enumerate(layers):
        if not targets[i]:
            out_lines.append(orig_text[i])
            continue
        d = param_dict(parts)
        no = d[0]                      # num_output
        bt = d.get(5, "0")             # bias_term
        ws = d[6]                      # weight_data_size
        i8 = d.get(8, None)
        at = d.get(9, None)
        ap = d.get(10, None)
        in_blob = parts[4]
        out_blob = parts[4 + 1]
        tmp = str(next_blob)
        next_blob += 1

        ip = ["InnerProduct", parts[1], "1", "1", in_blob, tmp, f"0={no}", f"1={bt}", f"2={ws}"]
        if i8 is not None:
            v = int(i8)
            if v > 100:
                v -= 100
            ip.append(f"8={v}")
        if at is not None:
            ip.append(f"9={at}")
        if ap is not None:
            ip.append(f"10={ap}")
        rs = ["Reshape", parts[1] + "_shape", "1", "1", tmp, out_blob, "0=1", "1=1", f"2={no}"]
        out_lines.append(fmt(ip))
        out_lines.append(fmt(rs))
    return out_lines


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(2)
    in_param, in_bin, out_param, out_bin = sys.argv[1:]

    n_layer, n_blob, in_lines, layers = parse_param(in_param)
    targets = find_targets(layers)
    n_t = sum(1 for t in targets if t)
    print(f"{in_param}: {n_t} convolution(s) -> InnerProduct+Reshape")

    new_lines = rewrite_param(in_lines, layers, targets)
    with open(out_param, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(new_lines) + "\n")

    with open(in_bin, "rb") as f:
        src = f.read()
    with open(out_bin, "wb") as f:
        copy_bin(src, f, layers, targets)


if __name__ == "__main__":
    main()
