
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.zip;
import std.algorithm;
import std.exception;
import std.format;
import tensorformats.storage;
import tensorformats.utils;

enum ZipFlags : ushort
{
    none = 0,
    encrypted = 0x0001,
    lengthAtEnd = 0x0008,
}

struct LocalFileHeader
{
    align(1):
    ushort versionNeeded;
    ZipFlags generalPurposeFlags;
    ushort compression;
    ushort lastModTime;
    ushort lastModDate;
    uint crc32;
    uint compressedSize;
    uint uncompressedSize;
    ushort filenameLength;
    ushort extraLength;
}
static assert(LocalFileHeader.sizeof == 26);

struct CentralFileHeader
{
    align(1):
    ushort versionMadeBy;
    ushort versionNeeded;
    ZipFlags generalPurposeFlags;
    ushort compression;
    ushort lastModTime;
    ushort lastModDate;
    uint crc32;
    uint compressedSize;
    uint uncompressedSize;
    ushort filenameLength;
    ushort extraLength;
    ushort commentLength;
    ushort diskNr;
    ushort internalAttr;
    uint externalAttr;
    uint localHeaderOffset;
}
static assert(CentralFileHeader.sizeof == 42);

struct EndOfCentralDirectoryRecord
{
    align(1):
    ushort diskNr;
    ushort centralDirDisk;
    ushort numCentralDirRecordsThisDisk;
    ushort numCentralDirRecordsTotal;
    uint centralDirSize;
    uint centralDirOffset;
    ushort commentLength;
}
static assert(EndOfCentralDirectoryRecord.sizeof == 18);

struct EndOfCentralDirectoryRecord64
{
    align(1):
    ulong recordSize;
    ushort versionMadeBy;
    ushort versionNeeded;
    uint diskNr;
    uint centralDirDisk;
    ulong numCentralDirRecordsThisDisk;
    ulong numCentralDirRecordsTotal;
    ulong centralDirSize;
    ulong centralDirOffset;
}
static assert(EndOfCentralDirectoryRecord64.sizeof == 52);

struct EndOfCentralDirectoryLocator
{
    align(1):
    uint centralDir64EndDisk;
    ulong centralDir64EndOffset;
    uint numDisksTotal;
}
static assert(EndOfCentralDirectoryLocator.sizeof == 16);

/**
Metadata about an entry in a zip file.
*/
struct ZipEntry
{
    /// This entry is valid
    bool valid;

    /// The entry uses the ZIP 64 extension
    bool isZip64;

    /// The length of the file is known, see `compressedSize` and `uncompressedSize`
    bool hasLength;

    /// Filename of the file
    string filename;

    /// Flags from the file header
    ZipFlags generalPurposeFlags;

    /// Size of compressed file, only valid if `hasLength` is set
    ulong compressedSize;

    /// Size of uncompressed file, only valid if `hasLength` is set
    ulong uncompressedSize;

    /// Offset of local header in zip file
    ulong localHeaderOffset;
}

/**
Class for reading files from zip archives.

Use method `readNextFile` to start reading a file. It returns `true`
if a next file exists. Metadata about the file can then be queried with
`currentEntry`. The content of files can be read using method from
interface `StorageReader`.

See https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
for a specification of zip files.
*/
class ZipReader : StorageReader
{
    this(StorageReader reader) @safe
    {
        this.reader = reader;

        if (reader.canSeekBack())
        {
            tryReadingCentralDir();
        }
    }

    private StorageReader reader;
    private size_t currentCheckedPrefix;
    private uint currentCrc32;
    private ulong currentPosInFile;
    private ZipEntry currentEntry_;
    private bool usesCentralDir_;
    private ZipEntry[] centralDirEntries;

    /**
    Get metadata about the current zip entry

    Only valid after `readNextFile` returned `true`.
    */
    ref const(ZipEntry) currentEntry() @safe => currentEntry_;

    final bool usesCentralDir() @safe => usesCentralDir_;

    private final void tryReadingCentralDir() @safe
    {
        auto tmp = reader.read(2, ReadFlags.temporary);
        enforce!StorageException(tmp == ['P', 'K']);

        reader.seekFromBack(EndOfCentralDirectoryRecord.sizeof + 4);

        auto expectedEnd = reader.currentPos;

        tmp = reader.read(4, ReadFlags.temporary);
        enforce!StorageException(tmp == ['P', 'K', 0x05, 0x06]);

        auto eocdBuf = reader.read(EndOfCentralDirectoryRecord.sizeof, ReadFlags.temporary);
        auto eocd = fromLittleEndian!EndOfCentralDirectoryRecord(eocdBuf);

        ulong centralDirOffset = eocd.centralDirOffset;
        ulong centralDirSize = eocd.centralDirSize;

        if (expectedEnd >= EndOfCentralDirectoryLocator.sizeof - 4)
        {
            reader.seekTo(expectedEnd - EndOfCentralDirectoryLocator.sizeof - 4);

            tmp = reader.read(4, ReadFlags.temporary);
            if (tmp == ['P', 'K', 0x06, 0x07])
            {
                auto locatorBuf = reader.read(EndOfCentralDirectoryLocator.sizeof, ReadFlags.temporary);
                auto locator = fromLittleEndian!EndOfCentralDirectoryLocator(locatorBuf);

                expectedEnd = locator.centralDir64EndOffset;

                reader.seekTo(locator.centralDir64EndOffset);

                tmp = reader.read(4, ReadFlags.temporary);
                enforce!StorageException(tmp == ['P', 'K', 0x06, 0x06]);

                auto eocd64Buf = reader.read(EndOfCentralDirectoryRecord64.sizeof, ReadFlags.temporary);
                auto eocd64 = fromLittleEndian!EndOfCentralDirectoryRecord64(eocd64Buf);

                if (centralDirOffset != uint.max)
                    enforce!StorageException(centralDirOffset == eocd64.centralDirOffset);
                if (centralDirSize != uint.max)
                    enforce!StorageException(centralDirSize == eocd64.centralDirSize);
                centralDirOffset = eocd64.centralDirOffset;
                centralDirSize = eocd64.centralDirSize;
            }
        }

        enforce!StorageException(centralDirOffset + centralDirSize == expectedEnd);

        reader.seekTo(centralDirOffset);
        while (reader.currentPos < expectedEnd)
        {
            enforce!StorageException(centralDirEntries.length < eocd.numCentralDirRecordsTotal);
            tmp = reader.read(4, ReadFlags.temporary);
            enforce!StorageException(tmp == ['P', 'K', 0x01, 0x02]);

            ZipEntry entry;
            readFileHeader!CentralFileHeader(entry);
            centralDirEntries ~= entry;
        }
        enforce!StorageException(reader.currentPos == expectedEnd);
        enforce!StorageException(centralDirEntries.length == eocd.numCentralDirRecordsTotal);

        usesCentralDir_ = true;
    }

    ulong currentPos() @safe
    {
        assert(currentEntry_.valid);
        return currentPosInFile;
    }
    ulong currentPosOrig() @safe
    {
        return reader.currentPos;
    }

    const(ubyte)[] read(size_t length, ReadFlags flags) @safe
    {
        assert(currentEntry_.valid);
        if (!currentEntry_.hasLength)
        {
            if (length > uint.max - 1024)
            {
                enforce!StorageException(flags & ReadFlags.allowPartial);
                length = uint.max - 1024;
            }

            const(ubyte)[] buffer = reader.read(length + (currentEntry_.isZip64 ? 24 : 16), (ReadFlags.temporary | ReadFlags.peek | ReadFlags.allowPartial));

            size_t length2 = currentCheckedPrefix;
            for (; length2 < length; length2++)
            {
                if (buffer.length < length2 + (currentEntry_.isZip64 ? 24 : 16))
                {
                    buffer = reader.read(length2 + (currentEntry_.isZip64 ? 24 : 16), (ReadFlags.temporary | ReadFlags.peek));
                }
                // Assume the signature is always there.
                if (buffer[length2] == 'P' && buffer[length2 + 1] == 'K' && buffer[length2 + 2] == 7 && buffer[length2 + 3] == 8)
                {
                    ulong compressedSize = currentEntry_.isZip64
                            ? buffer[length2 + 8 .. length2 + 16].fromLittleEndian!ulong
                            : buffer[length2 + 8 .. length2 + 12].fromLittleEndian!uint;
                    ulong uncompressedSize = currentEntry_.isZip64
                            ? buffer[length2 + 16 .. length2 + 24].fromLittleEndian!ulong
                            : buffer[length2 + 12 .. length2 + 16].fromLittleEndian!uint;
                    if (length2 > currentCheckedPrefix)
                    {
                        currentCrc32 = trustedCrc32(currentCrc32, buffer[currentCheckedPrefix .. length2]);
                        currentCheckedPrefix = length2;
                    }
                    if (buffer[length2 + 4 .. length2 + 8].fromLittleEndian!uint == currentCrc32)
                    {
                        //writefln("currentCrc32: %x  %d %d", currentCrc32, compressedSize, currentEntry_.compressedSize);
                        enforce!StorageException(uncompressedSize == currentPosInFile + currentCheckedPrefix);
                        enforce!StorageException(compressedSize == currentPosInFile + currentCheckedPrefix);
                        currentEntry_.compressedSize = currentPosInFile + currentCheckedPrefix;
                        currentEntry_.uncompressedSize = currentPosInFile + currentCheckedPrefix;
                        currentEntry_.hasLength = true;
                        break;
                    }
                }
            }
            if (length2 > currentCheckedPrefix)
            {
                currentCrc32 = trustedCrc32(currentCrc32, buffer[currentCheckedPrefix .. length2]);
                currentCheckedPrefix = length2;
            }
            if (length2 < length)
            {
                if (length2 == 0)
                {
                    if (!(flags & ReadFlags.allowEmpty))
                        throw new StorageException("unexpected end");
                }
                else
                {
                    if (!(flags & ReadFlags.allowPartial))
                        throw new StorageException("unexpected end");
                }
            }
            buffer = reader.read(length2, flags);
            if (!(flags & ReadFlags.peek))
            {
                currentPosInFile += buffer.length;
                currentCheckedPrefix = 0;
            }
            return buffer;
        }
        else
        {
            if (length > currentEntry_.compressedSize - currentPosInFile)
            {
                length = cast(size_t) (currentEntry_.compressedSize - currentPosInFile);
            }
            const(ubyte)[] buffer = reader.read(length, flags);
            if (!(flags & ReadFlags.peek))
            {
                currentPosInFile += buffer.length;
            }
            return buffer;
        }
    }

    const(ubyte)[] readAll(ReadFlags flags) @safe
    {
        assert(currentEntry_.valid);
        assert(currentPosInFile == 0);
        if (!currentEntry_.hasLength)
        {
            size_t size = 1024;
            size_t lastSize;
            while (true)
            {
                auto r = read(size, ReadFlags.temporary | ReadFlags.peek | ReadFlags.allowEmpty | ReadFlags.allowPartial);
                if (currentEntry_.hasLength)
                    break;
                enforce!StorageException(r.length > lastSize);
                lastSize = r.length;
                size *= 2;
            }
        }
        enforce!StorageException(currentEntry_.uncompressedSize <= size_t.max);
        auto r = read(cast(size_t) currentEntry_.uncompressedSize, flags);
        assert(r.length == currentEntry_.uncompressedSize);
        return r;
    }

    bool canSeekBack(bool allowDetect = true) @safe
    {
        return usesCentralDir_;
    }

    void seekTo(ulong pos) @safe
    {
        assert(currentEntry_.valid);
        if (!usesCentralDir_)
            enforce!StorageException(pos >= currentPosInFile);
        if (!currentEntry_.hasLength)
        {
            while (currentPosInFile < pos)
            {
                ulong left = pos - currentPosInFile;
                if (left > size_t.max)
                    left = size_t.max;
                auto r = read(cast(size_t) left, ReadFlags.temporary | ReadFlags.allowPartial);
                enforce!StorageException(r.length);
            }
            assert(currentPosInFile == pos);
        }
        else
        {
            enforce!StorageException(pos <= currentEntry_.uncompressedSize);
            reader.seekTo(reader.currentPos + (pos - currentPosInFile));
            currentPosInFile = pos;
        }
    }

    void seekFromBack(ulong pos) @safe
    {
        enforce!StorageException(usesCentralDir);
        assert(currentEntry_.hasLength);
        enforce!StorageException(pos <= currentEntry_.uncompressedSize);
        reader.seekTo(reader.currentPos - currentPosInFile + (currentEntry_.uncompressedSize - pos));
        currentPosInFile = currentEntry_.uncompressedSize - pos;
    }

    void seekToEnd() @safe
    {
        enforce!StorageException(currentEntry_.valid);
        while (!currentEntry_.hasLength || currentPosInFile < currentEntry_.compressedSize)
        {
            auto buffer = read(1024, ReadFlags.allowPartial | ReadFlags.allowEmpty);
            enforce!StorageException(buffer.length || (currentEntry_.hasLength && currentPosInFile == currentEntry_.compressedSize));
        }
        reader.seekTo(reader.currentPos + (currentEntry_.compressedSize - currentPosInFile));
    }

    private final void readFileHeader(T)(ref ZipEntry entry) @safe if (is(T == LocalFileHeader) || is(T == CentralFileHeader))
    {
        static if (is(T == LocalFileHeader))
            entry.localHeaderOffset = reader.currentPos;
        auto headerBuf = reader.read(T.sizeof, ReadFlags.temporary);
        auto header = fromLittleEndian!T(headerBuf);
        entry.generalPurposeFlags = header.generalPurposeFlags;
        enforce!StorageException((entry.generalPurposeFlags & ZipFlags.encrypted) == 0, "Encrypted ZIP files not supported");
        enforce!StorageException(header.compression == 0, "Compressed ZIP files not supported");
        entry.compressedSize = header.compressedSize;
        entry.uncompressedSize = header.uncompressedSize;
        static if (is(T == LocalFileHeader))
            if (entry.generalPurposeFlags & ZipFlags.lengthAtEnd)
            {
                enforce!StorageException(entry.compressedSize == 0);
                enforce!StorageException(entry.uncompressedSize == 0);
            }
        currentPosInFile = 0;
        size_t filenameLength = header.filenameLength;
        size_t extraLength = header.extraLength;
        auto filename = reader.read(filenameLength, ReadFlags.temporary);
        entry.filename = (cast(const(char)[]) filename).idup;
        auto extra = reader.read(extraLength, ReadFlags.temporary);
        entry.isZip64 = false;
        static if (is(T == LocalFileHeader))
            entry.hasLength = !(entry.generalPurposeFlags & ZipFlags.lengthAtEnd);
        static if (is(T == CentralFileHeader))
            entry.hasLength = true;
        entry.valid = true;
        static if (is(T == CentralFileHeader))
            entry.localHeaderOffset = header.localHeaderOffset;
        while (extra.length)
        {
            enforce!StorageException(extra.length >= 4);
            ushort id = extra[0 .. 2].fromLittleEndian!ushort;
            ushort size = extra[2 .. 4].fromLittleEndian!ushort;
            enforce!StorageException(extra.length >= 4 + size);
            auto extraData = extra[4 .. 4 + size];
            extra = extra[4 + size .. $];
            //writefln("  extra: 0x%04x 0x%04x %s", id, size, dataToHex(extraData));
            if (id == 0x0001)
            {
                entry.isZip64 = true;
                if (entry.uncompressedSize == 0xffffffff)
                {
                    entry.uncompressedSize = extraData[0 .. 8].fromLittleEndian!ulong;
                    extraData = extraData[8 .. $];
                }
                if (entry.compressedSize == 0xffffffff)
                {
                    entry.compressedSize = extraData[0 .. 8].fromLittleEndian!ulong;
                    extraData = extraData[8 .. $];
                }
                static if (is(T == CentralFileHeader))
                    if (entry.localHeaderOffset == 0xffffffff)
                    {
                        entry.localHeaderOffset = extraData[0 .. 8].fromLittleEndian!ulong;
                        extraData = extraData[8 .. $];
                    }
            }
        }
    }

    /**
    Try to read the next file and return if it exists

    Metadata about the file can then be queried with `currentEntry`.
    */
    bool readNextFile() @safe
    {
        if (currentEntry_.valid)
        {
            seekToEnd();
            auto tmp = reader.read(4, ReadFlags.temporary | ReadFlags.peek);
            bool hasDataDescriptor = tmp == ['P', 'K', 0x07, 0x08];
            if (currentEntry_.generalPurposeFlags & ZipFlags.lengthAtEnd)
                enforce!StorageException(hasDataDescriptor);
            if (hasDataDescriptor)
            {
                tmp = reader.read(currentEntry_.isZip64 ? 24 : 16, ReadFlags.temporary);
            }
        }

        currentEntry_ = ZipEntry.init;

        if (usesCentralDir_)
        {
            if (centralDirEntries.length == 0)
                return false;
            currentEntry_ = centralDirEntries[0];
            centralDirEntries = centralDirEntries[1 .. $];
            reader.seekTo(currentEntry_.localHeaderOffset);

            auto tmp = reader.read(4, ReadFlags.temporary);
            enforce!StorageException(tmp == ['P', 'K', 0x03, 0x04]);

            ZipEntry localEntry;
            readFileHeader!LocalFileHeader(localEntry);
            if ((localEntry.generalPurposeFlags & ZipFlags.lengthAtEnd) == 0)
            {
                enforce!StorageException(currentEntry_.compressedSize == localEntry.compressedSize);
                enforce!StorageException(currentEntry_.uncompressedSize == localEntry.uncompressedSize);
            }
        }
        else
        {
            auto tmp = reader.read(4, ReadFlags.temporary);
            if (tmp == ['P', 'K', 0x01, 0x02])
                return false;
            if (tmp == ['P', 'K', 0x05, 0x06])
                return false;
            enforce!StorageException(tmp == ['P', 'K', 0x03, 0x04]);

            readFileHeader!LocalFileHeader(currentEntry_);
        }

        currentCrc32 = 0;
        currentCheckedPrefix = 0;
        return true;
    }
}
