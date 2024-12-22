import std.algorithm;
import std.array;
import std.conv;
import std.stdio;
import tensorformats.storage;
import tensorformats.tensorreader;
import tensorformats.types;

int main(string[] args)
{
    string filename;
    bool useMmap;
    bool smallBuffers = true;
    bool noData;
    for (size_t i = 1; i < args.length; i++)
    {
        string arg = args[i];
        if (arg.startsWith("-"))
        {
            if (arg == "--mmap")
                useMmap = true;
            else if (arg == "--small-buffers")
                smallBuffers = true;
            else if (arg == "--big-buffers")
                smallBuffers = false;
            else if (arg == "--no-data")
                noData = true;
            else
            {
                writeln("Unknown arg ", arg);
                return 1;
            }
        }
        else
        {
            if (filename.length)
            {
                writeln("Multiple files not supported");
                return 1;
            }
            filename = arg;
        }
    }
    if (filename.length == 0)
    {
        writeln("Missing filename");
        return 1;
    }
    StorageReader storage;
    if (filename.endsWith(".gz"))
        storage = new GzipStorage(filename);
    else if (useMmap)
        storage = new MmapStorage(filename);
    else
        storage = new FileStorage(filename);
    TensorReader reader = readTensors(storage, smallBuffers);

    if (noData)
    {
        foreach (tensor; reader.readAllTensorInfos)
        {
            writef("%s %s shape=%s stride=%s\n",
                tensor.name, tensor.type,
                tensor.shape.map!(x => text(x)).join("x"),
                tensor.stride);
        }
    }
    else
    {
        size_t bufferNr;
        while (reader.readNextBuffer())
        {
            ulong start = reader.currentPosOrig;
            auto dataBuffer = reader.read(reader.bufferSize(), ReadFlags.none);
            foreach (tensor; reader.tensorsInBuffer)
            {
                auto data2 = dataBuffer[tensor.offsetStart .. $][0 .. tensor.sizeBytes];
                writef("0x%08x 0x%08x buffer=%u %s %s shape=%s stride=%s\n",
                    start + tensor.offsetStart, tensor.offsetStart,
                    bufferNr,
                    tensor.name, tensor.type,
                    tensor.shape.map!(x => text(x)).join("x"),
                    tensor.stride);
                Appender!string app;
                app.put("  ");
                tensorToString(app, tensor, data2, 2);
                writeln(app.data);
            }
            bufferNr++;
        }
    }
    return 0;
}
