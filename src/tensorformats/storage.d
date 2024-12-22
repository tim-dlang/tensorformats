
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.storage;
import core.stdc.string;
import etc.c.zlib;
import std.conv;
import std.exception;
import std.mmfile;
import std.stdio;
import std.string;

/**
Exception thrown on Pickle errors
*/
class StorageException : Exception
{
    this(string msg, string file = __FILE__, size_t line = __LINE__) pure nothrow @safe
    {
        super(msg, file, line);
    }
}

/**
Flags for reading data from `StorageReader`
*/
enum ReadFlags
{
    /**
    Default flags
    */
    none,

    /**
    The returned buffer is only valid until the next read.
    */
    temporary = 0x01,

    /**
    Don't advance the position in the stream, so the next read can
    return the same data.
    */
    peek = 0x02,

    /**
    The read can return an empty buffer at the end. Otherwise an
    exception is thrown at the end.
    */
    allowEmpty = 0x04,

    /**
    The read can return a smaller buffer than requested. Otherwise an
    exception is thrown for incomplete data at the end.
    */
    allowPartial = 0x08,
}

/**
Interface for reading files or streams
*/
interface StorageReader
{
    /**
    Get the current position in the stream or file, which can be used with seekTo
    */
    ulong currentPos() @safe;

    /**
    Get the current position in the underlying stream or file

    This can be different from currentPos for wrappers.
    */
    ulong currentPosOrig() @safe;

    /**
    Read data from at the current position

    Params:
        length = Maximum or exact length of data depending on `flags`
        flags = Flags for read, see members of `ReadFlags`
    */
    const(ubyte)[] read(size_t length, ReadFlags flags) @safe
    out(r)
    {
        if ((flags & ReadFlags.allowEmpty) == 0 && length)
            assert(r.length);
        if ((flags & ReadFlags.allowPartial) == 0)
            assert(r.length == 0 || r.length == length, text(r.length, " ", length, " ", flags));
    };

    /**
    Determine if seeking can be done backward or from back of file

    If a call to `canSeekBack` returned `true`, then every further call
    also has to return `true`.

    Params:
        allowDetect = Allow to test seeking
    */
    bool canSeekBack(bool allowDetect = true) @safe;

    /**
    Seek to an absolute position in the file

    Params:
        pos = Absolute position in file
    */
    void seekTo(ulong pos) @safe in(canSeekBack(false) || pos >= currentPos());

    /**
    Seek to a position in the file from back

    Params:
        pos = Position from back of file
    */
    void seekFromBack(ulong pos) @safe in(canSeekBack(false));
}

/**
Base class for storage with common read implementation
*/
class AbstractFileStorage : StorageReader
{
    private ulong currentPos_;
    private ubyte[] temporaryBuffer;
    private size_t temporaryBufferUsedStart;
    private size_t temporaryBufferUsedLength;
    private size_t temporaryBufferUsedDirty;

    /**
    Read new data into buffer and return number of read bytes

    Params:
        buffer = Buffer for new data
    Returns:
        Number of bytes written into buffer or 0 at end of file/stream
    */
    protected abstract size_t readImpl(ubyte[] buffer) @safe;

    ulong currentPos() @safe
    {
        return currentPos_;
    }
    ulong currentPosOrig() @safe
    {
        return currentPos_;
    }

    const(ubyte)[] read(size_t length, ReadFlags flags) @safe
    {
        assert(temporaryBufferUsedStart + temporaryBufferUsedLength <= temporaryBuffer.length);
        const(ubyte)[] r;
        if ((flags & ReadFlags.temporary) == 0 && (flags & ReadFlags.peek) == 0)
        {
            ubyte[] buffer = new ubyte[length];
            if (length <= temporaryBufferUsedLength)
            {
                buffer[] = temporaryBuffer[temporaryBufferUsedStart .. temporaryBufferUsedStart + length];
                temporaryBufferUsedStart += length;
                temporaryBufferUsedLength -= length;
                temporaryBuffer[0 .. temporaryBufferUsedStart] = 0;
                r = buffer;
            }
            else
            {
                size_t oldTemporaryBufferUsedLength = temporaryBufferUsedLength;
                buffer[0 .. temporaryBufferUsedLength] = temporaryBuffer[temporaryBufferUsedStart .. temporaryBufferUsedStart + temporaryBufferUsedLength];

                temporaryBuffer[0 .. temporaryBufferUsedDirty] = 0;
                temporaryBufferUsedStart = 0;
                temporaryBufferUsedLength = 0;
                temporaryBufferUsedDirty = 0;

                while (length > oldTemporaryBufferUsedLength)
                {
                    size_t newData = readImpl(buffer[oldTemporaryBufferUsedLength .. $]);
                    oldTemporaryBufferUsedLength += newData;
                    if (newData == 0 || (flags & ReadFlags.allowPartial))
                        break;
                }
                r = buffer[0 .. oldTemporaryBufferUsedLength];
            }
            currentPos_ += r.length;
        }
        else
        {
            if (temporaryBuffer.length < length)
                temporaryBuffer.length = length;

            if (temporaryBufferUsedStart)
            {
                foreach (i; 0 .. temporaryBufferUsedLength)
                    temporaryBuffer[i] = temporaryBuffer[temporaryBufferUsedStart + i];
                //memmove(temporaryBuffer.ptr, temporaryBuffer.ptr + temporaryBufferUsedStart, temporaryBufferUsedLength);
            }
            temporaryBuffer[temporaryBufferUsedLength .. temporaryBufferUsedDirty] = 0;
            temporaryBufferUsedStart = 0;

            while (length > temporaryBufferUsedLength)
            {
                size_t newData = readImpl(temporaryBuffer[temporaryBufferUsedLength .. $]);
                temporaryBufferUsedLength += newData;
                if (newData == 0 || (flags & ReadFlags.allowPartial))
                    break;
            }

            size_t length2 = length;
            if (temporaryBufferUsedLength < length2)
                length2 = temporaryBufferUsedLength;

            r = temporaryBuffer[0 .. length2];

            if ((flags & ReadFlags.temporary) == 0)
            {
                r = r.dup;
                if ((flags & ReadFlags.peek) == 0)
                    temporaryBuffer[0 .. length2] = 0;
            }

            temporaryBufferUsedDirty = temporaryBufferUsedStart + temporaryBufferUsedLength;

            if ((flags & ReadFlags.peek) == 0)
            {
                currentPos_ += length2;
                temporaryBufferUsedLength -= length2;
                temporaryBufferUsedStart += length2;
            }
        }

        assert(temporaryBufferUsedStart + temporaryBufferUsedLength <= temporaryBuffer.length);
        if (r.length < length)
        {
            if (r.length == 0)
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
        return r;
    }

    bool canSeekBack(bool allowDetect = true) @safe
    {
        return false;
    }

    void seekTo(ulong pos) @safe
    {
        enforce!StorageException(pos >= currentPos_);
        while (currentPos_ < pos)
        {
            ulong left = pos - currentPos_;
            if (left > size_t.max)
                left = size_t.max;
            auto r = read(cast(size_t) left, ReadFlags.temporary | ReadFlags.allowPartial);
            enforce!StorageException(r.length);
        }
        assert(currentPos_ == pos);
    }

    void seekFromBack(ulong pos) @safe
    {
        assert(false);
    }

    protected void resetAfterSeek(ulong pos) @safe
    {
        currentPos_ = pos;
        temporaryBuffer[0 .. temporaryBufferUsedDirty] = 0;
        temporaryBufferUsedStart = 0;
        temporaryBufferUsedLength = 0;
        temporaryBufferUsedDirty = 0;
    }
}

/**
Storage using file
*/
class FileStorage : AbstractFileStorage
{
    protected File file;

    /**
    Open file by name
    */
    this(string filename) scope
    {
        file = File(filename, "rb");
    }

    /*
    Needs to be trusted, because rawRead is currently @system on Windows.
    See https://issues.dlang.org/show_bug.cgi?id=23421
    */
    override protected size_t readImpl(ubyte[] buffer) @trusted
    {
        buffer = file.rawRead(buffer);
        return buffer.length;
    }

    void close()
    {
        file.close();
    }

    private bool canSeekBack_;
    private bool canSeekBackChecked_;

    override bool canSeekBack(bool allowDetect = true) @safe
    {
        if (!canSeekBackChecked_ && allowDetect)
        {
            try
            {
                file.seek(0, SEEK_CUR);
                canSeekBack_ = true;
            }
            catch (ErrnoException)
            {
                canSeekBack_ = false;
            }
            canSeekBackChecked_ = true;
        }
        return canSeekBack_;
    }

    override void seekTo(ulong pos) @safe
    {
        if (!canSeekBack_)
        {
            super.seekTo(pos);
            return;
        }
        file.seek(pos, SEEK_SET);
        assert(file.tell == pos);
        resetAfterSeek(pos);
    }

    override void seekFromBack(ulong pos) @safe
    {
        assert(canSeekBack_);
        file.seek(-cast(long) pos, SEEK_END);
        resetAfterSeek(file.tell);
    }
}

// Used for tests
class FileStorageTestMinRead : FileStorage
{
    this(string filename) scope
    {
        super(filename);
    }

    override protected size_t readImpl(ubyte[] buffer) @safe
    {
        return super.readImpl(buffer[0 .. 1]);
    }
}

/**
Storage using file with gzip compression

Does not support seeking back.
*/
class GzipStorage : AbstractFileStorage
{
    private gzFile file;

    /**
    Open file by name
    */
    this(string filename) scope
    {
        file = gzopen(filename.toStringz, "rb");
        enforce!StorageException(file);
    }

    override protected size_t readImpl(ubyte[] buffer) @trusted
    {
        uint len = buffer.length < uint.max ? cast(uint) buffer.length : uint.max;
        int r = gzread(file, buffer.ptr, len);
        enforce!StorageException(r >= 0);
        return r;
    }

    void close()
    {
        gzclose_r(file);
        file = null;
    }
}

/**
Storage using memory mapped file
*/
class MmapStorage : MmFile, StorageReader
{
    /**
    Open file by name
    */
    this(string filename) scope
    {
        super(filename);

        fileLength = length;
        mappedData = cast(const(ubyte)[]) this[0 .. fileLength];
    }

    private ulong currentPos_;
    private ulong fileLength;
    private const(ubyte)[] mappedData;

    ulong currentPos() @safe
    {
        return currentPos_;
    }
    ulong currentPosOrig() @safe
    {
        return currentPos_;
    }

    const(ubyte)[] read(size_t length, ReadFlags flags) @safe
    {
        assert(currentPos_ <= this.fileLength);
        if (length > this.fileLength - currentPos_)
        {
            length = cast(size_t) (this.fileLength - currentPos_);
            if (length == 0)
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
        auto r = mappedData[cast(size_t) currentPos .. cast(size_t) (currentPos + length)];
        if ((flags & ReadFlags.peek) == 0)
            currentPos_ += length;
        return r;
    }

    bool canSeekBack(bool allowDetect = true) @safe
    {
        return true;
    }

    void seekTo(ulong pos) @safe
    {
        currentPos_ = pos;
    }

    void seekFromBack(ulong pos) @safe
    {
        enforce!StorageException(pos <= this.fileLength);
        currentPos_ = this.fileLength - pos;
    }
}

/**
Storage using memory mapped file
*/
class MemoryStorage : StorageReader
{
    /**
    Treat data in memory as file
    */
    this(const(ubyte)[] data) scope @safe
    {
        this.data = data;
    }

    private ulong currentPos_;
    private const(ubyte)[] data;

    ulong currentPos() @safe
    {
        return currentPos_;
    }
    ulong currentPosOrig() @safe
    {
        return currentPos_;
    }

    const(ubyte)[] read(size_t length, ReadFlags flags) @safe
    {
        assert(currentPos_ <= this.data.length);
        if (length > this.data.length - currentPos_)
        {
            length = cast(size_t) (this.data.length - currentPos_);
            if (length == 0)
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
        auto r = data[cast(size_t) currentPos .. cast(size_t) (currentPos + length)];
        if ((flags & ReadFlags.peek) == 0)
            currentPos_ += length;
        return r;
    }

    bool canSeekBack(bool allowDetect = true) @safe
    {
        return true;
    }

    void seekTo(ulong pos) @safe
    {
        currentPos_ = pos;
    }

    void seekFromBack(ulong pos) @safe
    {
        enforce!StorageException(pos <= this.data.length);
        currentPos_ = this.data.length - pos;
    }
}
