
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.utils;
import std.format;
import std.traits;

string dataToHex(const(ubyte)[] buf) @safe
{
    string r;
    foreach (i, b; buf)
    {
        r ~= format("%02X", b);
        if (i < buf.length && i % 4 == 3)
            r ~= " ";
    }
    return r;
}

T fromLittleEndian(T)(const ubyte[] data) @trusted
{
    assert(data.length >= T.sizeof);
    T r;
    static if (is(T == struct))
    {
        size_t offset;
        foreach (ref field; r.tupleof)
        {
            field = fromLittleEndian!(typeof(field))(data[offset .. offset + field.sizeof]);
            offset += field.sizeof;
        }
    }
    else static if(isBasicType!T)
    {
        version(LittleEndian)
            (cast(ubyte*) &r)[0 .. T.sizeof] = data[0 .. T.sizeof];
        else
            foreach (i; 0 .. T.sizeof)
                (cast(ubyte*) &r)[T.sizeof - 1 - i] = data[i];
    }
    else
        static assert(false, "fromLittleEndian not implemented for " ~ T.stringof);
    return r;
}

uint trustedCrc32(uint crc, const(ubyte)[] buf) @trusted
{
    import etc.c.zlib: crc32;
    while (buf.length)
    {
        uint len;
        if (buf.length > uint.max)
            len = uint.max;
        else
            len = cast(uint) buf.length;
        crc = crc32(crc, buf.ptr, len);
        buf = buf[len .. $];
    }
    return crc;
}
