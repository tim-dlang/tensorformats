
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.buffersplitter;
import std.algorithm;
import tensorformats.storage;
import tensorformats.tensorreader;

private struct BufferInfo
{
    ulong offset;
    ulong size;
    TensorInfo[] tensors;
}

/**
Split every buffer into multiple buffers for different tensors.
Overlapping tensors can still be in the same buffer.
*/
class BufferSplitter : AbstractTensorReader!TensorReader
{
    this(TensorReader reader) @safe
    {
        super(reader);
    }

    BufferInfo[] buffers;
    size_t currentBuffer;

    override bool readNextBuffer() @safe
    {
        if (currentBuffer + 1 < buffers.length)
        {
            currentBuffer++;
            setRegion(buffers[currentBuffer].offset, buffers[currentBuffer].size);
            return true;
        }
        if (reader.readNextBuffer())
        {
            TensorInfo[] tensors = reader.tensorsInBuffer.dup;
            tensors.sort!((a, b) => a.offsetStart < b.offsetStart);
            buffers = [];

            for (size_t i = 0; i < tensors.length; )
            {
                size_t overlapping = 1;
                ulong maxEnd = tensors[i].offsetStart + tensors[i].sizeBytes;
                while (i + overlapping < tensors.length)
                {
                    if (tensors[i + overlapping].offsetStart >= maxEnd)
                        break;
                    ulong end = tensors[i + overlapping].offsetStart + tensors[i + overlapping].sizeBytes;
                    if (end > maxEnd)
                        maxEnd = end;
                    overlapping++;
                }
                BufferInfo buffer;
                buffer.tensors = tensors[i .. i + overlapping];
                buffer.offset = tensors[i].offsetStart;
                buffer.size = 0;
                foreach (ref tensor; buffer.tensors)
                {
                    tensor.offsetStart -= buffer.offset;
                    if (tensor.offsetStart + tensor.sizeBytes > buffer.size)
                        buffer.size = tensor.offsetStart + tensor.sizeBytes;
                }
                this.buffers ~= buffer;
                i += overlapping;
            }
            if (tensors.length == 0)
                buffers ~= BufferInfo(0, 0, []);
            currentBuffer = 0;
            setRegion(buffers[currentBuffer].offset, buffers[currentBuffer].size);
            return true;
        }
        setRegion(ulong.max, 0);
        return false;
    }

    override const(TensorInfo)[] tensorsInBuffer() @safe
    {
        return buffers[currentBuffer].tensors;
    }

    override ulong bufferSize() @safe
    {
        return regionSize;
    }

    override const(TensorInfo)[] readAllTensorInfos() @safe
    {
        TensorInfo[] r;
        while (++currentBuffer < buffers.length)
        {
            foreach (TensorInfo tensor; buffers[currentBuffer].tensors)
            {
                tensor.offsetStart = ulong.max;
                r ~= tensor;
            }
        }
        r ~= reader.readAllTensorInfos();
        setRegion(ulong.max, 0);
        return r;
    }
}
