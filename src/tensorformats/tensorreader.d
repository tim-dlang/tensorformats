
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.tensorreader;
import std.algorithm;
import std.array;
import std.conv;
import tensorformats.storage;
import tensorformats.types;

/**
Exception thrown on tensor reading errors
*/
class TensorReaderException : Exception
{
    this(string msg, string file = __FILE__, size_t line = __LINE__) pure nothrow @safe
    {
        super(msg, file, line);
    }
}

/**
Metadata about a tensor without actual data
*/
struct TensorInfo
{
    /**
    Name of tensor
    */
    string name;

    /**
    Offset of start of tensor data from buffer start
    */
    ulong offsetStart;

    /**
    Size of tensor data in bytes
    */
    ulong sizeBytes;

    /**
    Type of values in the the tensor, which also determines element size
    */
    ValueType type;

    /**
    Number of elements for every dimension of the tensor
    */
    const(ulong)[] shape;

    /**
    Offset of next element for every dimension of the tensor
    */
    const(ulong)[] stride;
}

/**
Generate a string representation of a tensor

Big tensor are abbreviated.

Params:
    app = Appender for generating the string
    tensorInfo = Metadata about the tensor
    data = Buffer with data for the tensor
    extraIndent = Number of extra space after every newline
*/
void tensorToString(ref Appender!string app, TensorInfo tensorInfo, const(ubyte)[] data, size_t extraIndent = 0)
{
    assert(tensorInfo.shape.length == tensorInfo.stride.length);
    const dim = tensorInfo.shape.length;
    const elemSize = valueTypeSizeof(tensorInfo.type);
    if (dim == 0)
    {
        app.put("single value = ");
        app.put(valueToString(tensorInfo.type, data[0 .. elemSize]));
        return;
    }
    app.put("[");
    for (ulong y = 0; y < tensorInfo.shape[0]; y++)
    {
        if (y >= 5 && y < tensorInfo.shape[0] - 5)
        {
            if (dim > 1)
            {
                app.put(",\n ");
                foreach (_; 0 .. extraIndent)
                    app.put(" ");
                app.put("...");
            }
            else
                app.put(", ...");
            y = tensorInfo.shape[0] - 5;
        }
        const rowOffset = y * tensorInfo.stride[0] * elemSize;
        if (y)
        {
            app.put(dim > 1 ? ",\n " : ", ");
            if (dim > 1)
                foreach (_; 0 .. extraIndent)
                    app.put(" ");
        }
        if (dim > 1)
        {
            TensorInfo next = tensorInfo;
            next.shape = next.shape[1 .. $];
            next.stride = next.stride[1 .. $];
            next.offsetStart += rowOffset;
            next.sizeBytes -= rowOffset;
            tensorToString(app, next, data[cast(size_t) rowOffset .. $], extraIndent + 1);
        }
        else
        {
            app.put(valueToString(tensorInfo.type, data[cast(size_t) rowOffset .. cast(size_t) (rowOffset + elemSize)]));
        }
    }
    app.put("]");
}

/**
Convert coordinates in a tensor into an index for the buffer

Params:
    coords = Coordinates of element
    shape = Shape of tensor
    stride = Stride of tensor
*/
ulong coordsToIndex(const(ulong)[] coords, const(ulong)[] shape, const(ulong)[] stride) @safe
in(shape.length == coords.length)
in(stride.length == 0 || stride.length == coords.length)
{
    ulong r;
    if (stride.length == 0)
    {
        ulong stride2 = 1;
        foreach_reverse (i; 0 .. coords.length)
        {
            r += coords[i] * stride2;
            stride2 *= shape[i];
        }
    }
    else
    {
        foreach (i; 0 .. coords.length)
            r += coords[i] * stride[i];
    }
    return r;
}

/**
Get one element from a tensor and convert it to a compile time type

Params:
    tensorInfo = Metadata about the tensor
    data = Buffer with data for the tensor
    coords = Coordinates of element
*/
T tensorGetElem(T)(TensorInfo tensorInfo, const(ubyte)[] data, const(ulong)[] coords)
in(tensorInfo.shape.length == tensorInfo.stride.length)
in(tensorInfo.shape.length == coords.length)
{
    auto index = coordsToIndex(coords, tensorInfo.shape, tensorInfo.stride);
    auto size = valueTypeSizeof(tensorInfo.type);
    return valueToType!T(tensorInfo.type, data[cast(size_t) (size * index) .. cast(size_t) (size * (index + 1))]);
}

/**
Interface for reading tensors from a file/stream

The data is split into buffers, where every buffer can contain multiple
tensors. Method `readNextBuffer` has to be called before reading any
tensors. It returns `true` until no more buffers can be read.
The size of the current buffer can be received with method `bufferSize`.
Metadata about all tensors in the current buffer is available with
method `tensorsInBuffer`. The tensors can use overlapping data for
some file formats. Data can be read using methods from base interface
`StorageReader`.
*/
interface TensorReader : StorageReader
{
    /**
    Start reading the next buffer and return if it exists
    */
    bool readNextBuffer() @safe;

    /**
    Get metadata about all tensors in the current buffer
    */
    const(TensorInfo)[] tensorsInBuffer() @safe;

    /**
    Get the size of the current buffer
    */
    ulong bufferSize() @safe;

    /**
    Get metadata for all tensors in remaining buffers

    This method can be used when the data is not needed. For some file
    formats it can be more efficient, because only part the file has
    to be read.

    The metadata will be without offsets, because it can contain
    tensors from different buffers.
    */
    const(TensorInfo)[] readAllTensorInfos() @safe;
}

/**
Base class with common implementation for tensor readers
*/
class AbstractTensorReader(NextReader : StorageReader) : TensorReader
{
    this(NextReader reader) scope @safe
    {
        this.reader = reader;
    }

    protected NextReader reader;
    protected ulong regionOffset = ulong.max;
    protected ulong regionSize;

    protected void setRegion(ulong offset, ulong size) @safe
    {
        if (offset != ulong.max)
            reader.seekTo(offset);
        regionOffset = offset;
        regionSize = size;
    }

    ulong currentPos() @safe
    {
        assert(regionOffset != ulong.max);
        auto r = reader.currentPos;
        assert(r >= regionOffset);
        assert(r <= regionOffset + regionSize, text(r, " ", regionOffset, " ", regionSize));
        return r - regionOffset;
    }

    ulong currentPosOrig() @safe
    {
        return reader.currentPos;
    }

    const(ubyte)[] read(size_t length, ReadFlags flags) @safe
    {
        assert(regionOffset != ulong.max);
        auto pos = currentPos;
        if (length > regionSize - pos)
        {
            length = cast(size_t) (regionSize - pos);
            if (length == 0)
            {
                if (!(flags & ReadFlags.allowEmpty))
                    throw new TensorReaderException("unexpected end");
            }
            else
            {
                if (!(flags & ReadFlags.allowPartial))
                    throw new TensorReaderException("unexpected end");
            }
        }
        return reader.read(length, flags);
    }

    bool canSeekBack(bool allowDetect = true) @safe
    {
        return reader.canSeekBack(allowDetect);
    }

    void seekTo(ulong pos) @safe
    {
        assert(regionOffset != ulong.max);
        assert(pos <= regionSize);
        reader.seekTo(regionOffset + pos);
    }

    void seekFromBack(ulong pos) @safe
    {
        assert(regionOffset != ulong.max);
        assert(pos <= regionSize);
        reader.seekTo(regionOffset + regionSize - pos);
    }

    abstract bool readNextBuffer() @safe;
    abstract const(TensorInfo)[] tensorsInBuffer() @safe;

    ulong bufferSize() @safe
    {
        return regionSize;
    }

    const(TensorInfo)[] readAllTensorInfos() @safe
    {
        TensorInfo[] r;
        while (readNextBuffer())
        {
            foreach (TensorInfo tensor; tensorsInBuffer)
            {
                tensor.offsetStart = ulong.max;
                r ~= tensor;
            }
        }
        return r;
    }
}

/**
Detect tensor file format and return tensor reader

Params:
    reader = Reader for input file/stream
    smallBuffers = Prefer small buffers instead of big buffers with multiple tensors
*/
TensorReader readTensors(StorageReader reader, bool smallBuffers = true) @safe
{
    import tensorformats.buffersplitter;
    import tensorformats.gguf;
    import tensorformats.pytorch;
    import tensorformats.safetensors;
    import tensorformats.zip;

    const ubyte[] data = reader.read(12, ReadFlags.temporary | ReadFlags.peek);
    static immutable ubyte[] ggufPrefix = ['G', 'G', 'U', 'F'];
    static immutable ubyte[] zipPrefix = ['P', 'K', 0x03, 0x04];
    TensorReader r;
    if (data.startsWith(ggufPrefix))
    {
        r = new GGUFReader(reader);
    }
    else if (data.startsWith(zipPrefix))
    {
        ZipReader zipReader = new ZipReader(reader);
        r = new PytorchReader(zipReader);
    }
    else if (data.length >= 9 && data[8] == '{')
    {
        r = new SafetensorsReader(reader);
    }
    else
        throw new TensorReaderException("Undetected file format");

    if (smallBuffers)
        r = new BufferSplitter(r);
    return r;
}
