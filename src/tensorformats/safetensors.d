
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.safetensors;
import std.algorithm;
import std.exception;
import std.json;
import std.range;
import tensorformats.storage;
import tensorformats.tensorreader;
import tensorformats.types;
import tensorformats.utils;

private ulong jsonToUlong(JSONValue v) @safe
{
    if (v.type == JSONType.integer)
    {
        auto i = v.integer;
        enforce!TensorReaderException(i >= 0);
        return i;
    }
    else if (v.type == JSONType.uinteger)
    {
        auto i = v.uinteger;
        return i;
    }
    enforce!TensorReaderException(false);
    assert(false);
}

private ValueType convertSafetensorsType(string name) @safe
{
    switch (name)
    {
        case "F32": return ValueType.float_;
        case "F64": return ValueType.double_;
        case "F16": return ValueType.half;
        case "BF16": return ValueType.bfloat16;
        case "U8": return ValueType.uint8;
        case "U16": return ValueType.uint16;
        case "U32": return ValueType.uint32;
        case "U64": return ValueType.uint64;
        case "I8": return ValueType.int8;
        case "I16": return ValueType.int16;
        case "I32": return ValueType.int32;
        case "I64": return ValueType.int64;
        case "F8_E5M2": return ValueType.f8_e5m2;
        case "F8_E4M3": return ValueType.f8_e4m3;
        case "BOOL": return ValueType.bool_;
        default: return ValueType.unknown;
    }
}

private TensorInfo[] readTensorInfos(StorageReader reader, ref ulong offset) @safe
{
    auto lengthBuffer = reader.read(ulong.sizeof, ReadFlags.temporary);
    ulong headerLength = lengthBuffer.fromLittleEndian!ulong;
    enforce!TensorReaderException(headerLength <= 100 * 1024 * 1024);
    auto headerBuffer = reader.read(cast(size_t) headerLength, ReadFlags.temporary);
    auto header = parseJSON(cast(const(char)[]) headerBuffer);
    offset = 8 + headerLength;

    TensorInfo[] tensors;
    foreach (string key, value; header.objectNoRef)
    {
        if (key == "__metadata__")
            continue;
        TensorInfo tensor;
        tensor.name = key;
        tensor.offsetStart = value["data_offsets"].arrayNoRef[0].jsonToUlong;
        ulong offsetEnd = value["data_offsets"].arrayNoRef[1].jsonToUlong;
        enforce!TensorReaderException(offsetEnd >= tensor.offsetStart);
        tensor.sizeBytes = offsetEnd - tensor.offsetStart;
        tensor.type = convertSafetensorsType(value["dtype"].str);
        tensor.shape = value["shape"].arrayNoRef.map!(x => x.jsonToUlong).array;
        ulong[] stride = new ulong[tensor.shape.length];
        if (tensor.shape.length)
        {
            stride[$ - 1] = 1;
            foreach_reverse (i; 0 .. tensor.shape.length - 1)
                stride[i] = stride[i + 1] * tensor.shape[i + 1];
        }
        tensor.stride = stride;
        tensors ~= tensor;
    }
    tensors.sort!((a, b) => a.offsetStart < b.offsetStart);
    ulong lastEnd = 0;
    foreach (ref tensor; tensors)
    {
        enforce!TensorReaderException(tensor.offsetStart == lastEnd);
        lastEnd = tensor.offsetStart + tensor.sizeBytes;
    }

    return tensors;
}

/**
Tensor reader for safetensors file format

A specification of the file format can be found here:
- https://github.com/huggingface/safetensors
*/
class SafetensorsReader : AbstractTensorReader!StorageReader
{
    this(StorageReader reader) @safe
    {
        super(reader);
        tensors = readTensorInfos(reader, offset);
        size = 0;
        foreach (tensor; tensors)
            if (tensor.offsetStart + tensor.sizeBytes > size)
                size = tensor.offsetStart + tensor.sizeBytes;
    }

    private ulong offset;
    private ulong size;
    private TensorInfo[] tensors;
    private int state;

    override bool readNextBuffer() @safe
    {
        if (state == 0)
        {
            state++;
            setRegion(offset, size);
            return true;
        }
        tensors = [];
        setRegion(ulong.max, 0);
        return false;
    }

    override const(TensorInfo)[] tensorsInBuffer() @safe
    {
        if (state)
        {
            return tensors;
        }
        return [];
    }
}
