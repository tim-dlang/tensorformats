import std.algorithm;
import std.file;
import std.stdio;
import tensorformats.pickle;
import tensorformats.storage;
import tensorformats.zip;

void main(string[] args)
{
    string filename = args[1];

    StorageReader storage;
    if (filename.endsWith(".gz"))
        storage = new GzipStorage(filename);
    else
        storage = new FileStorage(filename);

    const ubyte[] data = storage.read(12, ReadFlags.temporary | ReadFlags.peek);
    static immutable ubyte[] zipPrefix = ['P', 'K', 0x03, 0x04];
    if (data.startsWith(zipPrefix))
    {
        ZipReader zipReader = new ZipReader(storage);
        if (!zipReader.readNextFile())
            throw new Exception("Empty ZIP file");
        // Use first file in zip
        storage = zipReader;
    }

    auto item = parsePickle(storage);
    writeln(*item);
}
