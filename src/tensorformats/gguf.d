
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.gguf;
import std.algorithm;
import std.exception;
import std.range;
import tensorformats.storage;
import tensorformats.tensorreader;
import tensorformats.types;
import tensorformats.utils;

private struct TypeUDA(T)
{
    alias Type = T;
    size_t expectedSize;
}
private template TypeUDAFor(string name)
{
    enum TypeUDAFor = __traits(getAttributes, __traits(getMember, ValueType, name))[0];
}

// See ggml_type in https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
private enum GgmlType: uint
{
    @(ValueType.float_) F32     = 0,
    @(ValueType.half) F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    @(ValueType.int8) I8      = 24,
    @(ValueType.int16) I16     = 25,
    @(ValueType.int32) I32     = 26,
    @(ValueType.int64) I64     = 27,
    @(ValueType.double_) F64     = 28,
    IQ1_M   = 29
};

private ValueType convertGgufType(GgmlType type) @safe
{
    switch (type)
    {
        static foreach (name; __traits(allMembers, GgmlType))
        {
            static if (__traits(getAttributes, __traits(getMember, GgmlType, name)).length)
            {
                case __traits(getMember, GgmlType, name):
                    return __traits(getAttributes, __traits(getMember, GgmlType, name))[0];
            }
        }
    default:
        return ValueType.unknown;
    }
}

// See gguf_metadata_value_type in https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
private enum GgufMetadataValueType: uint
{
    // The value is a 8-bit unsigned integer.
    @(ValueType.uint8) UINT8 = 0,
    // The value is a 8-bit signed integer.
    @(ValueType.int8) INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    @(ValueType.uint16) UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    @(ValueType.int16) INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    @(ValueType.uint32) UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    @(ValueType.int32) INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    @(ValueType.float_) FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    @(ValueType.bool_) BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    @(ValueType.uint64) UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    @(ValueType.int64) INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    @(ValueType.double_) FLOAT64 = 12,
};

private enum GGUF_DEFAULT_ALIGNMENT = 32;

/**
Tensor reader for GGUF file format

A specification of the file format can be found here:
- https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
*/
class GGUFReader : AbstractTensorReader!StorageReader
{
    private T readLE(T)()
    {
        auto data = reader.read(T.sizeof, ReadFlags.temporary);
        return fromLittleEndian!T(data);
    }

    this(StorageReader reader) @safe
    {
        super(reader);

        auto magic = reader.read(4, ReadFlags.temporary);
        enforce!TensorReaderException(magic == ['G', 'G', 'U', 'F']);
        auto version_ = readLE!uint;
        enforce!TensorReaderException(version_ == 3);

        auto count = readLE!ulong;
        auto metadataKVCount = readLE!ulong;

        tensors.reserve(cast(size_t) count);

        foreach (i; 0 .. metadataKVCount)
            readMetadataKV();

        foreach (i; 0 .. count)
            readTensorInfo();

        offset = reader.currentPos;
        offset += (currentAlignment - (reader.currentPos % currentAlignment)) % currentAlignment;

        size = 0;
        foreach (tensor; tensors)
            if (tensor.offsetStart + tensor.sizeBytes > size)
                size = tensor.offsetStart + tensor.sizeBytes;
    }

    private const(char)[] readString() @safe
    {
        auto length = readLE!ulong;
        enforce!TensorReaderException(length <= size_t.max);
        auto data = reader.read(cast(size_t) length, ReadFlags.none);
        return cast(const(char)[]) data;
    }
    private void readType(GgufMetadataValueType type) @safe
    {
        s: switch (type)
        {
            static foreach (name; __traits(allMembers, GgufMetadataValueType))
            {
                static if (__traits(getAttributes, __traits(getMember, GgufMetadataValueType, name)).length)
                {
                    case __traits(getMember, GgufMetadataValueType, name):
                    {
                        enum valueType = __traits(getAttributes, __traits(getMember, GgufMetadataValueType, name))[0];
                        enum size = valueTypeSizeof(valueType);
                        auto value = reader.read(size, ReadFlags.temporary);
                        break s;
                    }
                }
            }
        case GgufMetadataValueType.STRING:
        {
            auto value = readString();
            break;
        }
        case GgufMetadataValueType.ARRAY:
        {
            auto elemType = readLE!uint;
            auto len = readLE!ulong;
            foreach (i; 0 .. len)
                readType(cast(GgufMetadataValueType) elemType);
            break;
        }
        default:
            enforce!TensorReaderException(false);
        }
    }
    private void readMetadataKV() @safe
    {
        auto key = readString();
        auto type = readLE!uint;
        if (key == "general.alignment" && type == GgufMetadataValueType.UINT32)
        {
            auto value = reader.read(4, ReadFlags.temporary | ReadFlags.peek).fromLittleEndian!uint;
            enforce!TensorReaderException(value && value % 8 == 0);
            currentAlignment = value;
        }
        readType(cast(GgufMetadataValueType) type);
    }

    private void readTensorInfo() @safe
    {
        auto name = readString();
        auto dim = readLE!uint;
        ulong[] shape = new ulong[dim];
        foreach (i; 0 .. dim)
            shape[dim - 1 - i] = readLE!ulong;
        auto type = readLE!uint;
        auto offset = readLE!ulong;

        TensorInfo tensor;
        tensor.name = name.idup;
        tensor.offsetStart = offset;
        tensor.type = convertGgufType(cast(GgmlType) type);
        tensor.sizeBytes = valueTypeSizeof(tensor.type);
        foreach (i; 0 .. dim)
            tensor.sizeBytes *= shape[i];
        ulong[] stride = new ulong[dim];
        if (shape.length)
        {
            stride[$ - 1] = 1;
            foreach_reverse (i; 0 .. dim - 1)
                stride[i] = stride[i + 1] * shape[i + 1];
        }
        tensor.shape = shape;
        tensor.stride = stride;
        tensors ~= tensor;
    }

    private ulong offset;
    private ulong size;
    private TensorInfo[] tensors;
    private int state;
    private ulong currentAlignment = GGUF_DEFAULT_ALIGNMENT;

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
