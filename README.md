# TensorFormats

TensorFormats is a library for reading different tensor file formats in
the D programming language. The file formats are used for machine
learning models, like large language models.

## Features

* Read tensors from different file formats using the same interface
  * [Safetensors](https://github.com/huggingface/safetensors)
  * [Pytorch](https://pytorch.org/docs/stable/notes/serialization.html)
  * [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
* Mmap can be used for mapping the file into memory
* A file can be read in parts, so less memory is needed

## Limitations

* Only reading and not writing is supported
* No alignment guaranteed
* Additional format specific metadata not available
* Quantised formats are not supported yet

## Usage

The example [dumptensors.d](examples/dumptensors.d) can be used to print
the tensors in a file:

```
dub tensorformats:dumptensors -- tests/data/tensors/tensor-dims.safetensors

0x00000000 0x00000000 buffer=0 dim0 float_ shape= stride=[]
  single value = 4
0x00000004 0x00000000 buffer=1 dim1 float_ shape=5 stride=[1]
  [0, 1, 2, 3, 4]
0x00000018 0x00000000 buffer=2 dim2 float_ shape=2x4 stride=[4, 1]
  [[0, 1, 2, 3],
   [10, 11, 12, 13]]
0x00000038 0x00000000 buffer=3 dim3 float_ shape=3x2x3 stride=[6, 3, 1]
  [[[0, 1, 2],
    [10, 11, 12]],
   [[100, 101, 102],
    [110, 111, 112]],
   [[200, 201, 202],
    [210, 211, 212]]]
0x00000080 0x00000000 buffer=4 dim4 float_ shape=2x3x2x2 stride=[12, 4, 2, 1]
  [[[[0, 1],
     [10, 11]],
    [[100, 101],
     [110, 111]],
    [[200, 201],
     [210, 211]]],
   [[[1000, 1001],
     [1010, 1011]],
    [[1100, 1101],
     [1110, 1111]],
    [[1200, 1201],
     [1210, 1211]]]]
```

Here is a short example how tensors can be read from a file:

```d
import tensorformats.tensorreader, tensorformats.storage;
auto storage = new FileStorage(filename);
TensorReader reader = readTensors(storage);
while (reader.readNextBuffer())
{
    auto dataBuffer = reader.read(reader.bufferSize(), ReadFlags.none);
    foreach (tensor; reader.tensorsInBuffer)
    {
        // Use metadata in `tensor` with data in `dataBuffer`
    }
}
storage.close();
```

The file is split into buffers, where every buffer can contain multiple
tensors. The pytorch format allows overlapping tensors in the same buffer.
The metadata for a tensor has to be used to interpret the data.

The file format is automatically detected by `readTensors`. It is also
possible to instantiate a reader for one particular file format instead.

## License

Boost Software License, Version 1.0. See file [LICENSE_1_0.txt](LICENSE_1_0.txt).
