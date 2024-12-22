
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.pytorch;
import std.algorithm;
import std.conv;
import std.exception;
import tensorformats.pickle;
import tensorformats.storage;
import tensorformats.tensorreader;
import tensorformats.types;
import tensorformats.zip;

private struct PtStorage
{
    bool fileSeen;
    ulong size;
    TensorInfo[] tensors;
}

private ValueType convertPytorchType1(const(char)[] name) @safe
{
    switch (name)
    {
        case "torch.FloatStorage": return ValueType.float_;
        case "torch.DoubleStorage": return ValueType.double_;
        case "torch.HalfStorage": return ValueType.half;
        case "torch.BFloat16Storage": return ValueType.bfloat16;
        case "torch.ByteStorage": return ValueType.uint8;
        case "torch.CharStorage": return ValueType.int8;
        case "torch.ShortStorage": return ValueType.int16;
        case "torch.IntStorage": return ValueType.int32;
        case "torch.LongStorage": return ValueType.int64;
        case "torch.BoolStorage": return ValueType.bool_;
        case "torch.ComplexDoubleStorage": return ValueType.cdouble_;
        case "torch.ComplexFloatStorage": return ValueType.cfloat_;
        case "torch.storage.UntypedStorage": return ValueType.unknown;
        default: return ValueType.unknown;
    }
}

private ValueType convertPytorchType2(const(char)[] name) @safe
{
    switch (name)
    {
        case "torch.float":
        case "torch.float32": return ValueType.float_;
        case "torch.double":
        case "torch.float64": return ValueType.double_;
        case "torch.half":
        case "torch.float16": return ValueType.half;
        case "torch.bfloat16": return ValueType.bfloat16;
        case "torch.uint8": return ValueType.uint8;
        case "torch.int8": return ValueType.int8;
        case "torch.short":
        case "torch.int16": return ValueType.int16;
        case "torch.int":
        case "torch.int32": return ValueType.int32;
        case "torch.long":
        case "torch.int64": return ValueType.int64;
        case "torch.bool": return ValueType.bool_;
        case "torch.cdouble":
        case "torch.complex128": return ValueType.cdouble_;
        case "torch.cfloat":
        case "torch.complex64": return ValueType.cfloat_;
        case "torch.chalf":
        case "torch.complex32": return ValueType.chalf;
        case "torch.float8_e5m2": return ValueType.f8_e5m2;
        case "torch.float8_e4m3fn": return ValueType.f8_e4m3;
        default: return ValueType.unknown;
    }
}

private void findTensors(ref PtStorage*[string] ptStorages, string name, Item* item) @safe
{
    if (item is null)
        return;
    if (item.type == ItemType.dict
        || (item.type == ItemType.reduce
            && item.childs[0].type == ItemType.global
            && item.childs[0].data == "collections.OrderedDict"))
    {
        foreach (c; item.dictChilds)
        {
            enforce!TensorReaderException(c[0].type == ItemType.str);
            string name2 = name;
            if (name2.length)
                name2 ~= ".";
            name2 ~= c[0].data;
            findTensors(ptStorages, name2, c[1]);
        }
    }
    else if (item.type.among(ItemType.list, ItemType.tuple))
    {
        foreach (i, c; item.childs)
        {
            string name2 = name;
            if (name2.length)
                name2 ~= ".";
            name2 ~= text(i);
            findTensors(ptStorages, name2, c);
        }
    }
    else if (item.type == ItemType.reduce)
    {
        enforce!TensorReaderException(item.childs[0].type == ItemType.global);
        enforce!TensorReaderException(item.childs[0].data.among("torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor_v3"));
        enforce!TensorReaderException(item.childs[1].type == ItemType.tuple);
        enforce!TensorReaderException(item.childs[1].childs[0].type == ItemType.persid);
        enforce!TensorReaderException(item.childs[1].childs[0].childs[0].type == ItemType.tuple);
        auto storageTuple = item.childs[1].childs[0].childs[0].childs;
        enforce!TensorReaderException(storageTuple[0].type == ItemType.str);
        enforce!TensorReaderException(storageTuple[0].data == "storage");
        enforce!TensorReaderException(storageTuple[1].type == ItemType.global);
        enforce!TensorReaderException(storageTuple[2].type == ItemType.str);
        enforce!TensorReaderException(storageTuple[4].type == ItemType.int_);
        auto storageEntry = ptStorages.require(storageTuple[2].toStr.idup, new PtStorage);
        TensorInfo tensor;
        tensor.name = name;
        tensor.type = convertPytorchType1(storageTuple[1].toStr);
        storageEntry.size = (cast(const(char)[]) storageTuple[4].data).to!ulong;
        if (item.childs[0].data == "torch._utils._rebuild_tensor_v2")
            storageEntry.size *= valueTypeSizeof(tensor.type);
        if (item.childs[0].data == "torch._utils._rebuild_tensor_v3")
            tensor.type = convertPytorchType2(item.childs[1].childs[6].toStr);
        tensor.offsetStart = (cast(const(char)[]) item.childs[1].childs[1].data).to!ulong;
        tensor.offsetStart *= valueTypeSizeof(tensor.type);
        enforce!TensorReaderException(item.childs[1].childs[2].type == ItemType.tuple);
        foreach (c; item.childs[1].childs[2].childs)
        {
            enforce!TensorReaderException(c.type == ItemType.int_);
            tensor.shape ~= (cast(const(char)[]) c.data).to!size_t;
        }
        enforce!TensorReaderException(item.childs[1].childs[3].type == ItemType.tuple);
        foreach (c; item.childs[1].childs[3].childs)
        {
            enforce!TensorReaderException(c.type == ItemType.int_);
            tensor.stride ~= (cast(const(char)[]) c.data).to!size_t;
        }
        enforce!TensorReaderException(tensor.shape.length == tensor.stride.length);
        tensor.sizeBytes = 1;
        foreach (i; 0 .. tensor.shape.length)
        {
            enforce!TensorReaderException(tensor.shape[i] > 0);
            tensor.sizeBytes += (tensor.shape[i] - 1) * tensor.stride[i];
        }
        tensor.sizeBytes *= valueTypeSizeof(tensor.type);
        enforce!TensorReaderException(tensor.offsetStart + tensor.sizeBytes <= storageEntry.size);
        storageEntry.tensors ~= tensor;
    }
    else
        throw new TensorReaderException(text("Unsupported pickle type ", item.type));
}

/**
Tensor reader for pytorch file format

Documentation about the file format can be found here:
- https://pytorch.org/docs/stable/notes/serialization.html
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
*/
class PytorchReader : AbstractTensorReader!ZipReader
{
    this(ZipReader reader) @safe
    {
        super(reader);
        if (!reader.currentEntry.valid)
        {
            if (!reader.readNextFile())
                throw new TensorReaderException("Empty ZIP file");
        }
        foreach (i; 0 .. reader.currentEntry.filename.length)
        {
            if (reader.currentEntry.filename[i].among('\\', '/'))
                filePrefix = reader.currentEntry.filename[0 .. i + 1];
        }
        if (reader.currentEntry.filename[filePrefix.length .. $] != "data.pkl")
            throw new TensorReaderException("Unexpected first filename: " ~ reader.currentEntry.filename);
        auto pklData = reader.readAll(ReadFlags.none);
        auto item = parsePickle(pklData);
        findTensors(ptStorages, "", item);
    }
    this(StorageReader reader) @safe
    {
        this(new ZipReader(reader));
    }

    private string filePrefix;
    private PtStorage*[string] ptStorages;
    private PtStorage* currentStorage;
    private bool finished;

    override bool readNextBuffer() @safe
    {
        currentStorage = null;
        setRegion(ulong.max, 0);
        if (finished)
            return false;
        while (reader.readNextFile())
        {
            enforce!TensorReaderException(reader.currentEntry.filename.startsWith(filePrefix));
            string filename2 = reader.currentEntry.filename[filePrefix.length .. $];
            if (filename2.startsWith("data/"))
            {
                auto x = filename2[5 .. $] in ptStorages;
                enforce!TensorReaderException(x);
                enforce!TensorReaderException(!(*x).fileSeen);
                (*x).fileSeen = true;
                currentStorage = *x;
                setRegion(0, currentStorage.size);
                return true;
            }
        }

        foreach (name, s; ptStorages)
        {
            if (!s.fileSeen)
                throw new TensorReaderException("No data for PyTorch storage " ~ name);
        }

        return false;
    }

    override const(TensorInfo)[] tensorsInBuffer() @safe
    {
        return currentStorage ? currentStorage.tensors : [];
    }

    override const(TensorInfo)[] readAllTensorInfos() @safe
    {
        TensorInfo[] r;
        foreach (name, s; ptStorages)
        {
            if (s.fileSeen)
                continue;
            s.fileSeen = true;
            foreach (TensorInfo tensor; s.tensors)
            {
                tensor.offsetStart = ulong.max;
                r ~= tensor;
            }
        }
        currentStorage = null;
        setRegion(ulong.max, 0);
        finished = true;
        return r;
    }
}
