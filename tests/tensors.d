import std.algorithm;
import std.array;
import std.complex;
import std.conv;
import std.meta;
import std.stdio;
import tensorformats.storage;
import tensorformats.tensorreader;
import tensorformats.types;

struct ExpectedTensor(T)
{
    ValueType type;
    const(ulong)[] shape;
    const(T)[] data;
}

auto iterateCoords(const(ulong)[] shape)
{
    struct R
    {
        private const(ulong)[] shape;
        private ulong[] coords;
        bool finished;

        bool empty()
        {
            if (finished)
                return true;
            foreach (i; 0 .. shape.length)
            {
                if (coords[i] >= shape[i])
                    return true;
            }
            return false;
        }
        const(ulong)[] front()
        {
            return coords;
        }
        void popFront()
        {
            foreach_reverse (i; 0 .. shape.length)
            {
                if (coords[i] + 1 < shape[i])
                {
                    coords[i]++;
                    coords[i + 1 .. $] = 0;
                    return;
                }
            }
            finished = true;
        }
    }
    return R(shape, new ulong[shape.length]);
}

unittest
{
    assert(iterateCoords([]).map!(x => x.dup).array == [[]]);
    assert(iterateCoords([0]).map!(x => x.dup).array == []);
    assert(iterateCoords([1]).map!(x => x.dup).array == [[0]]);
    assert(iterateCoords([3]).map!(x => x.dup).array == [[0], [1], [2]]);
    assert(iterateCoords([2, 3]).map!(x => x.dup).array == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]);
}

void checkTensors(T)(TensorReader reader, const ExpectedTensor!T[string] expected)
{
    bool[string] seen;
    string[] unexpectedTensors;

    while (reader.readNextBuffer())
    {
        ulong start = reader.currentPosOrig;
        auto dataBuffer = reader.read(cast(size_t) reader.bufferSize(), ReadFlags.none);
        foreach (tensor; reader.tensorsInBuffer)
        {
            auto data2 = dataBuffer[cast(size_t) tensor.offsetStart .. $][0 .. cast(size_t) tensor.sizeBytes];
            assert (tensor.name !in seen, "Duplicate tensor " ~ tensor.name);
            seen[tensor.name] = true;
            if (tensor.name !in expected)
            {
                unexpectedTensors ~= tensor.name;
                continue;
            }
            auto e = expected[tensor.name];
            assert(tensor.type == e.type, text(tensor.name, " ", tensor.type, " ", e.type));
            assert(tensor.shape == e.shape, text(tensor.name, " ", tensor.shape, " ", e.shape));
            foreach (coords; iterateCoords(tensor.shape))
            {
                auto val1 = e.data[cast(size_t) coordsToIndex(coords, e.shape, [])];
                auto val2 = tensorGetElem!T(tensor, data2, coords);
                assert(val1 is val2, text(tensor.name, " ", coords, " ", val1, " ", val2));
            }
        }
    }

    foreach (name, tensor; expected)
        assert(name in seen, "Missing tensor " ~ name);

    assert(unexpectedTensors.length == 0, text("Unexpected tensors ", unexpectedTensors));
}

void checkTensorInfosOnly(T)(TensorReader reader, const ExpectedTensor!T[string] expected)
{
    bool[string] seen;

    auto tensors = reader.readAllTensorInfos();
    assert(!reader.readNextBuffer());

    foreach (tensor; tensors)
    {
        assert (tensor.name !in seen, "Duplicate tensor " ~ tensor.name);
        seen[tensor.name] = true;
        assert(tensor.name in expected, "Unexpected tensor " ~ tensor.name);
        auto e = expected[tensor.name];
        assert(tensor.type == e.type);
        assert(tensor.shape == e.shape);
    }

    foreach (name, tensor; expected)
        assert(name in seen, "Missing tensor " ~ name);
}

void checkTensorsVariants(T)(string filename, const ExpectedTensor!T[string] expected)
{
    foreach (smallBuffers; [true, false])
    {
        static foreach (Storage; AliasSeq!(MmapStorage, FileStorage, FileStorageTestMinRead))
        {
            {
                StorageReader storage = new Storage(filename);
                TensorReader reader = readTensors(storage, smallBuffers);
                checkTensors(reader, expected);
            }
            {
                StorageReader storage = new Storage(filename);
                TensorReader reader = readTensors(storage, smallBuffers);
                checkTensorInfosOnly(reader, expected);
            }
        }
    }
}

void checkTensorsVariantsGzip(T)(string filename, const ExpectedTensor!T[string] expected)
{
    foreach (smallBuffers; [true, false])
    {
        {
            StorageReader storage = new GzipStorage(filename);
            TensorReader reader = readTensors(storage, smallBuffers);
            checkTensors(reader, expected);
        }
        {
            StorageReader storage = new GzipStorage(filename);
            TensorReader reader = readTensors(storage, smallBuffers);
            checkTensorInfosOnly(reader, expected);
        }
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/ints.safetensors", "tests/data/tensors/ints.pt", "tests/data/tensors/ints.gguf"])
    {
        ExpectedTensor!long[string] expected;
        expected["int64"] = ExpectedTensor!long(ValueType.int64, [6], [1, 0, -1, 64, -9223372036854775808, 9223372036854775807]);
        expected["int32"] = ExpectedTensor!long(ValueType.int32, [6], [1, 0, -1, 32, -2147483648, 2147483647]);
        expected["int16"] = ExpectedTensor!long(ValueType.int16, [6], [1, 0, -1, 16, -32768, 32767]);
        expected["int8"] = ExpectedTensor!long(ValueType.int8, [6], [1, 0, -1, 8, -128, 127]);
        if (!filename.endsWith(".gguf"))
        {
            expected["uint8"] = ExpectedTensor!long(ValueType.uint8, [6], [1, 0, 255, 8, 0, 255]);
            expected["bool"] = ExpectedTensor!long(ValueType.bool_, [3], [1, 0, 1]);
        }

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/floats.safetensors", "tests/data/tensors/floats.pt", "tests/data/tensors/floats.gguf"])
    {
        ExpectedTensor!double[string] expected;
        expected["double"] = ExpectedTensor!double(ValueType.double_, [7], [1, 0, -1, 64, -double.max, double.max, double.epsilon]);
        expected["float"] = ExpectedTensor!double(ValueType.float_, [7], [1, 0, -1, 32, -float.max, float.max, float.epsilon]);
        expected["half"] = ExpectedTensor!double(ValueType.half, [7], [1, 0, -1, 16, -65504, 65504, Half.epsilon.get!double]);
        if (!filename.endsWith(".gguf"))
            expected["bfloat16"] = ExpectedTensor!double(ValueType.bfloat16, [7], [1, 0, -1, 16, -BFloat16.max.get!double, BFloat16.max.get!double, BFloat16.epsilon.get!double]);

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/float8.safetensors", "tests/data/tensors/float8.pt"])
    {
        ExpectedTensor!double[string] expected;
        expected["float8_e4m3fn"] = ExpectedTensor!double(ValueType.f8_e4m3, [7], [1, 0, -1, 8, -448, 448, 0.125]);
        expected["float8_e5m2"] = ExpectedTensor!double(ValueType.f8_e5m2, [7], [1, 0, -1, 8, -57344, 57344, 0.25]);

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/complex.pt"])
    {
        ExpectedTensor!(Complex!double)[string] expected;
        expected["cfloat"] = ExpectedTensor!(Complex!double)(ValueType.cfloat_, [12], [
            Complex!double(1, 0), Complex!double(0, 0), Complex!double(-1, 0), Complex!double(32, 0),
            Complex!double(-float.max, 0), Complex!double(float.max, 0), Complex!double(float.epsilon, 0),
            Complex!double(2, 3),
            Complex!double(float.max, float.max), Complex!double(-float.max, float.max),
            Complex!double(float.max, -float.max), Complex!double(-float.max, -float.max)]);
        expected["cdouble"] = ExpectedTensor!(Complex!double)(ValueType.cdouble_, [12], [
            Complex!double(1, 0), Complex!double(0, 0), Complex!double(-1, 0), Complex!double(64, 0),
            Complex!double(-double.max, 0), Complex!double(double.max, 0), Complex!double(double.epsilon, 0),
            Complex!double(2, 3),
            Complex!double(double.max, double.max), Complex!double(-double.max, double.max),
            Complex!double(double.max, -double.max), Complex!double(-double.max, -double.max)]);
        expected["chalf"] = ExpectedTensor!(Complex!double)(ValueType.chalf, [12], [
            Complex!double(1, 0), Complex!double(0, 0), Complex!double(-1, 0), Complex!double(16, 0),
            Complex!double(-65504, 0), Complex!double(65504, 0), Complex!double(Half.epsilon.get!double, 0),
            Complex!double(2, 3),
            Complex!double(65504, 65504), Complex!double(-65504, 65504),
            Complex!double(65504, -65504), Complex!double(-65504, -65504)]);

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/tensor-single.pt"])
    {
        ExpectedTensor!float[string] expected;
        expected[""] = ExpectedTensor!float(ValueType.float_, [5], [0, 1, 2, 3, 4]);

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/tensor-tree.pt"])
    {
        ExpectedTensor!float[string] expected;
        expected["slice1"] = ExpectedTensor!float(ValueType.float_, [2, 3, 4, 2], [
            0, 1, 10, 11, 20, 21, 30, 31,
            100, 101, 110, 111, 120, 121, 130, 131,
            200, 201, 210, 211, 220, 221, 230, 231,
            1000, 1001, 1010, 1011, 1020, 1021, 1030, 1031,
            1100, 1101, 1110, 1111, 1120, 1121, 1130, 1131,
            1200, 1201, 1210, 1211, 1220, 1221, 1230, 1231]);
        expected["slice2"] = ExpectedTensor!float(ValueType.float_, [2, 3, 4, 2], [
            2, 3, 12, 13, 22, 23, 32, 33,
            102, 103, 112, 113, 122, 123, 132, 133,
            202, 203, 212, 213, 222, 223, 232, 233,
            1002, 1003, 1012, 1013, 1022, 1023, 1032, 1033,
            1102, 1103, 1112, 1113, 1122, 1123, 1132, 1133,
            1202, 1203, 1212, 1213, 1222, 1223, 1232, 1233]);
        expected["stride"] = ExpectedTensor!float(ValueType.float_, [3], [0, 2, 4]);
        expected["transpose"] = ExpectedTensor!float(ValueType.float_, [5, 4], [
            0, 10, 20, 30,
            1, 11, 21, 31,
            2, 12, 22, 32,
            3, 13, 23, 33,
            4, 14, 24, 34]);
        expected["permute1"] = ExpectedTensor!float(ValueType.float_, [2, 2, 3, 2], [
            0, 1000, 100, 1100, 200, 1200,
            10, 1010, 110, 1110, 210, 1210,
            1, 1001, 101, 1101, 201, 1201,
            11, 1011, 111, 1111, 211, 1211]);
        expected["permute2"] = ExpectedTensor!float(ValueType.float_, [2, 3, 2, 2], [
            0, 1000, 1, 1001,
            100, 1100, 101, 1101,
            200, 1200, 201, 1201,
            10, 1010, 11, 1011,
            110, 1110, 111, 1111,
            210, 1210, 211, 1211]);
        expected["array.0"] = ExpectedTensor!float(ValueType.float_, [5], [0, 1, 2, 3, 4]);
        expected["array.1"] = ExpectedTensor!float(ValueType.float_, [5], [10, 11, 12, 13, 14]);
        expected["tuple.0"] = ExpectedTensor!float(ValueType.float_, [5], [100, 101, 102, 103, 104]);
        expected["tuple.1"] = ExpectedTensor!float(ValueType.float_, [5], [110, 111, 112, 113, 114]);
        expected["OrderedDict.a"] = ExpectedTensor!float(ValueType.float_, [4, 5], [
            0, 1, 2, 3, 4,
            10, 11, 12, 13, 14,
            20, 21, 22, 23, 24,
            30, 31, 32, 33, 34]);
        expected["OrderedDict.b"] = ExpectedTensor!float(ValueType.float_, [4, 5], [
            100, 101, 102, 103, 104,
            110, 111, 112, 113, 114,
            120, 121, 122, 123, 124,
            130, 131, 132, 133, 134]);
        expected["nested.0.nested1"] = ExpectedTensor!float(ValueType.float_, [2], [0, 1000]);
        expected["nested.0.nested2"] = ExpectedTensor!float(ValueType.float_, [2], [1, 1001]);
        expected["nested.1.0.0.0.nested3"] = ExpectedTensor!float(ValueType.float_, [2], [2, 1002]);
        expected["nested.1.0.0.0.nested4"] = ExpectedTensor!float(ValueType.float_, [2], [3, 1003]);

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/tensor-dims.safetensors",
        "tests/data/tensors/tensor-dims.pt",
        "tests/data/tensors/tensor-dims.gguf",
        "tests/data/tensors/tensor-dims-align8.gguf",
        "tests/data/tensors/tensor-dims-align96.gguf"])
    {
        ExpectedTensor!float[string] expected;
        expected["dim0"] = ExpectedTensor!float(ValueType.float_, [], [4]);
        expected["dim1"] = ExpectedTensor!float(ValueType.float_, [5], [
            0, 1, 2, 3, 4]);
        expected["dim2"] = ExpectedTensor!float(ValueType.float_, [2, 4], [
            0, 1, 2, 3,
            10, 11, 12, 13]);
        expected["dim3"] = ExpectedTensor!float(ValueType.float_, [3, 2, 3], [
            0, 1, 2,
            10, 11, 12,
            100, 101, 102,
            110, 111, 112,
            200, 201, 202,
            210, 211, 212]);
        expected["dim4"] = ExpectedTensor!float(ValueType.float_, [2, 3, 2, 2], [
            0, 1, 10, 11,
            100, 101, 110, 111,
            200, 201, 210, 211,
            1000, 1001, 1010, 1011,
            1100, 1101, 1110, 1111,
            1200, 1201, 1210, 1211]);

        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/tensors/empty.safetensors", "tests/data/tensors/empty.pt", "tests/data/tensors/empty.gguf"])
    {
        ExpectedTensor!float[string] expected;
        checkTensorsVariants(filename, expected);
    }
}

unittest
{
    foreach (filename; ["tests/data/testmodel1/model.safetensors.gz", "tests/data/testmodel1/pytorch_model.bin.gz", "tests/data/testmodel1/Testmodel1-2.2K-F16.gguf.gz"])
    {
        ExpectedTensor!float[string] expected;
        size_t counter;
        void addExpected(string name, immutable ulong[] shape)
        {
            float[] data;
            counter++;
            if (shape.length == 2)
            {
                data.length = cast(size_t) (shape[0] * shape[1]);
                data[] = 0;
                data[0] = counter + 100;
                data[cast(size_t) (shape[1] - 1)] = counter + 200;
                data[cast(size_t) ((shape[0] - 1) * shape[1])] = counter + 300;
                data[cast(size_t) (shape[0] * shape[1] - 1)] = counter + 400;
            }
            if (shape.length == 1)
            {
                data.length = cast(size_t) shape[0];
                if (name.endsWith(".weight"))
                    data[] = 1;
                else
                    data[] = 0;
                data[0] = counter + 100;
                data[cast(size_t) (shape[0] - 1)] = counter + 200;
            }
            if (filename.endsWith(".gguf.gz"))
            {
                name = name.replace("gpt_neox.layers.", "blk.");
                name = name.replace("gpt_neox.final_layer_norm.", "output_norm.");
                name = name.replace("mlp.dense_h_to_4h.", "ffn_up.");
                name = name.replace("mlp.dense_4h_to_h.", "ffn_down.");
                name = name.replace("attention.query_key_value.", "attn_qkv.");
                name = name.replace("attention.dense.", "attn_output.");
                name = name.replace("embed_out.", "output.");
                name = name.replace("gpt_neox.embed_in.", "token_embd.");
                name = name.replace("post_attention_layernorm.", "ffn_norm.");
                name = name.replace("input_layernorm.", "attn_norm.");
            }
            ValueType expectedType = ValueType.float_;
            if (filename.endsWith("F16.gguf.gz") && !name.endsWith(".bias") && !name.endsWith("_norm.weight"))
                expectedType = ValueType.half;
            expected[name] = ExpectedTensor!float(expectedType, shape, data);
        }
        addExpected("embed_out.weight", [258, 4]);
        addExpected("gpt_neox.embed_in.weight", [258, 4]);
        addExpected("gpt_neox.final_layer_norm.bias", [4]);
        addExpected("gpt_neox.final_layer_norm.weight", [4]);
        addExpected("gpt_neox.layers.0.attention.dense.bias", [4]);
        addExpected("gpt_neox.layers.0.attention.dense.weight", [4, 4]);
        addExpected("gpt_neox.layers.0.attention.query_key_value.bias", [12]);
        addExpected("gpt_neox.layers.0.attention.query_key_value.weight", [12, 4]);
        addExpected("gpt_neox.layers.0.input_layernorm.bias", [4]);
        addExpected("gpt_neox.layers.0.input_layernorm.weight", [4]);
        addExpected("gpt_neox.layers.0.mlp.dense_4h_to_h.bias", [4]);
        addExpected("gpt_neox.layers.0.mlp.dense_4h_to_h.weight", [4, 8]);
        addExpected("gpt_neox.layers.0.mlp.dense_h_to_4h.bias", [8]);
        addExpected("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", [8, 4]);
        addExpected("gpt_neox.layers.0.post_attention_layernorm.bias", [4]);
        addExpected("gpt_neox.layers.0.post_attention_layernorm.weight", [4]);

        checkTensorsVariantsGzip(filename, expected);
    }
}
