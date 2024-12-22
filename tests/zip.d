import std.meta;
import std.stdio;
import tensorformats.storage;
import tensorformats.zip;

unittest
{
    static foreach (Storage; AliasSeq!(MmapStorage, FileStorage, FileStorageTestMinRead))
    {{
        auto reader = new ZipReader(new Storage("tests/data/zip/empty.zip"));
        assert(!reader.readNextFile());
    }}
}

immutable ubyte[239] testData1 = cast(immutable(ubyte)[239]) (
      "line 1: abcdefghijklmnopqrstuvwxyz 0123456789 ()[]{}<>&|!-_\n"
    ~ "line 2: abcdefghijklmnopqrstuvwxyz 0123456789 ()[]{}<>&|!-_\n"
    ~ "line 3: abcdefghijklmnopqrstuvwxyz 0123456789 ()[]{}<>&|!-_\n"
    ~ "line 4: abcdefghijklmnopqrstuvwxyz 0123456789 ()[]{}<>&|!-_");

immutable ubyte[64] testData2 = [
    'P', 'K', 0x07, 0x08, 'P', 'K', 0x01, 0x02,
    'P', 'K', 0x03, 0x04, 'P', 'K', 0x05, 0x06,
    'P', 'K', 0x07, 0x08, 'P', 'K', 0x01, 0x02,
    'P', 'K', 0x03, 0x04, 'P', 'K', 0x05, 0x06,
    'P', 'K', 0x07, 0x08, 'P', 'K', 0x01, 0x02,
    'P', 'K', 0x03, 0x04, 'P', 'K', 0x05, 0x06,
    'P', 'K', 0x07, 0x08, 'P', 'K', 0x01, 0x02,
    'P', 'K', 0x03, 0x04, 'P', 'K', 0x05, 0x06
    ];

void compareContentNoRead(ZipReader reader, const(ubyte)[] expected)
{
}
void compareContentBigRead(ZipReader reader, const(ubyte)[] expected)
{
    while (true)
    {
        auto data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        if (expected.length == 0)
            break;
        expected = expected[data.length .. $];
    }
}
void compareContentExactRead(ZipReader reader, const(ubyte)[] expected)
{
    auto data = reader.read(expected.length, ReadFlags.temporary);
    assert(data == expected);
    data = reader.read(1, ReadFlags.temporary | ReadFlags.allowEmpty);
    assert(data == []);
}
void compareContentPeekRead(ZipReader reader, const(ubyte)[] expected)
{
    while (true)
    {
        auto data = reader.read(1024, ReadFlags.temporary | ReadFlags.peek | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        if (expected.length == 0)
            break;
        expected = expected[data.length .. $];
    }
}
void compareContentExactPeekRead(ZipReader reader, const(ubyte)[] expected)
{
    auto data = reader.read(expected.length, ReadFlags.temporary | ReadFlags.peek);
    assert(data == expected);
    data = reader.read(expected.length, ReadFlags.temporary);
    assert(data == expected);
    data = reader.read(1, ReadFlags.temporary | ReadFlags.allowEmpty);
    assert(data == []);
}
void compareContentByteRead(ZipReader reader, const(ubyte)[] expected)
{
    foreach (i; 0 .. expected.length)
    {
        auto data = reader.read(1, ReadFlags.temporary);
        assert(data == expected[i .. i + 1]);
    }
    auto data = reader.read(1, ReadFlags.temporary | ReadFlags.allowEmpty);
    assert(data == []);
}
void compareContentBytePeekRead(ZipReader reader, const(ubyte)[] expected)
{
    foreach (i; 0 .. expected.length)
    {
        auto data = reader.read(1, ReadFlags.temporary | ReadFlags.peek);
        assert(data == expected[i .. i + 1]);
        data = reader.read(1, ReadFlags.temporary);
        assert(data == expected[i .. i + 1]);
    }
    auto data = reader.read(1, ReadFlags.temporary | ReadFlags.allowEmpty);
    assert(data == []);
}
void compareContentSeek(ZipReader reader, const(ubyte)[] expected)
{
    reader.seekTo(expected.length / 2);
    expected = expected[$ / 2 .. $];
    while (true)
    {
        auto data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        if (expected.length == 0)
            break;
        expected = expected[data.length .. $];
    }
}
void compareContentSeekExactRead(ZipReader reader, const(ubyte)[] expected)
{
    reader.seekTo(expected.length / 2);
    auto data = reader.read(expected.length - expected.length / 2, ReadFlags.temporary);
    assert(data == expected[$ / 2 .. $]);
}
void compareContentNoSeekBack(ZipReader reader, const(ubyte)[] expected)
{
    assert(!reader.canSeekBack());
}
void compareContentSeekBack(ZipReader reader, const(ubyte)[] expected)
{
    const origExpected = expected;
    assert(reader.canSeekBack());
    reader.seekFromBack(origExpected.length / 2);
    expected = origExpected[$ - $ / 2 .. $];
    while (true)
    {
        auto data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        if (expected.length == 0)
            break;
        expected = expected[data.length .. $];
    }
    reader.seekFromBack(origExpected.length);
    expected = origExpected;
    while (true)
    {
        auto data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        if (expected.length == 0)
            break;
        expected = expected[data.length .. $];
    }
    reader.seekFromBack(0);
    auto data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
    assert(data == []);
    reader.seekTo(0);
    expected = origExpected;
    while (true)
    {
        data = reader.read(1024, ReadFlags.temporary | ReadFlags.allowPartial | ReadFlags.allowEmpty);
        assert(data.length || expected.length == 0);
        assert(data.length <= expected.length);
        assert(data == expected[0 .. data.length]);
        if (expected.length == 0)
            break;
        expected = expected[data.length .. $];
    }
}
void compareContentSeekBackExactRead(ZipReader reader, const(ubyte)[] expected)
{
    assert(reader.canSeekBack());
    reader.seekFromBack(expected.length / 2);
    auto data = reader.read(expected.length / 2, ReadFlags.temporary);
    assert(data == expected[$ - $ / 2 .. $]);
    reader.seekFromBack(expected.length);
    data = reader.read(expected.length, ReadFlags.temporary);
    assert(data == expected);
    reader.seekFromBack(0);
    data = reader.read(1, ReadFlags.temporary | ReadFlags.allowEmpty);
    assert(data == []);
    reader.seekTo(0);
    data = reader.read(expected.length, ReadFlags.temporary);
    assert(data == expected);
}

alias comparisonFuncs = AliasSeq!(
    compareContentNoRead,
    compareContentBigRead,
    compareContentExactRead,
    compareContentPeekRead,
    compareContentExactPeekRead,
    compareContentByteRead,
    compareContentBytePeekRead,
    compareContentSeek,
    compareContentSeekExactRead
    );

immutable string[] similarTestFiles = [
    "tests/data/zip/testfile1.zip",
    "tests/data/zip/testfile2.zip",
    "tests/data/zip/testfile3.zip",
    "tests/data/zip/testfile4.zip",
    "tests/data/zip/testfile5.zip",
    "tests/data/zip/testfile6.zip",
    "tests/data/zip/testfile7.zip",
    "tests/data/zip/testfile8.zip",
    "tests/data/zip/testfile9.zip",
    ];

void checkTestFile(alias compareContent)(ZipReader reader)
{
    assert(reader.readNextFile());
    assert(reader.currentEntry.filename == "test1.txt");
    compareContent(reader, testData1);
    assert(reader.readNextFile());
    assert(reader.currentEntry.filename == "fakezip.zip");
    compareContent(reader, testData2);
    assert(reader.readNextFile());
    assert(reader.currentEntry.filename == "empty.txt");
    compareContent(reader, []);
    assert(reader.readNextFile());
    assert(reader.currentEntry.filename == "one.txt");
    compareContent(reader, [1]);
    assert(!reader.readNextFile());
}

class TestNoSeekStorage(Base : StorageReader) : Base
{
    this(string filename) scope
    {
        super(filename);
    }

    override bool canSeekBack(bool allowDetect = true)
    {
        return false;
    }

    override void seekTo(ulong pos)
    {
        assert(pos >= currentPos());
        super.seekTo(pos);
    }

    override void seekFromBack(ulong pos)
    {
        assert(false);
    }
}

unittest
{
    foreach (filename; similarTestFiles)
    static foreach (compare; AliasSeq!(comparisonFuncs, compareContentNoSeekBack))
    static foreach (Storage; AliasSeq!(TestNoSeekStorage!MmapStorage, TestNoSeekStorage!FileStorage, TestNoSeekStorage!FileStorageTestMinRead))
    {{
        scope (failure)
            stderr.writeln("Failure for ", __traits(identifier, compare), " ", Storage.stringof, " ", filename);
        auto zipReader = new ZipReader(new Storage(filename));
        assert(!zipReader.usesCentralDir);
        checkTestFile!compare(zipReader);
    }}
}

unittest
{
    foreach (filename; similarTestFiles)
    static foreach (compare; AliasSeq!(comparisonFuncs, compareContentSeekBack, compareContentSeekBackExactRead))
    static foreach (Storage; AliasSeq!(MmapStorage, FileStorage, FileStorageTestMinRead))
    {{
        scope (failure)
            stderr.writeln("Failure for ", __traits(identifier, compare), " ", __traits(identifier, Storage), " ", filename);
        auto zipReader = new ZipReader(new Storage(filename));
        assert(zipReader.usesCentralDir);
        checkTestFile!compare(zipReader);
    }}
}
