
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

/**
This module allows to read pickle data, which is a serialization format
used by Python.

Documentation about the format can be found here:
- https://docs.python.org/3/library/pickle.html
- https://docs.python.org/3/library/pickletools.html
*/
module tensorformats.pickle;
import std.algorithm;
import std.array;
import std.bigint;
import std.conv;
import std.exception;
import std.file;
import std.format;
import std.range;
import tensorformats.storage;

/**
Exception thrown on Pickle errors
*/
class PickleException : Exception
{
    this(string msg, string file = __FILE__, size_t line = __LINE__) pure nothrow @safe
    {
        super(msg, file, line);
    }
}

private struct Opcode
{
    ubyte code;
    int version_;
}

private immutable char[2][] escapeSequences = [
    ['\\', '\\'],
    ['\"', '\"'],
    ['\'', '\''],
    ['\n', 'n'],
    ['\t', 't'],
    ['\r', 'r'],
    ['\f', 'f'],
    ['\0', '0'],
    ['\a', 'a'],
    ['\b', 'b'],
    ['\v', 'v'],
];

private string escapeChar(dchar c) @safe
{
    if (c == 0)
        return "\\x00";
    foreach (s; escapeSequences)
    {
        if (c == s[0])
            return text("\\", s[1]);
    }
    if (c < 0x20 || c == 0x7f)
        return format("\\x%02x", c);
    return text(c);
}

private string escapeString(const(char)[] s) @safe
{
    string r;
    foreach (dchar c; s)
    {
        r ~= escapeChar(c);
    }
    return r;
}

private string escapeBytes(const(char)[] s) @safe
{
    string r;
    foreach (char c; s)
    {
        if (c >= 0x80)
            r ~= format("\\x%02x", c);
        else
            r ~= escapeChar(c);
    }
    return r;
}

private string unescapeString(const(char)[] s) @safe
{
    string r;
    while (s.length)
    {
        char c = s[0];
        s = s[1 .. $];
        if (c == '\\')
        {
            enforce!PickleException(s.length);
            dchar next = s[0];
            s = s[1 .. $];
            if (next == 'x' || next == 'u' || next == 'U')
            {
                ubyte len = 0;
                if (next == 'x')
                    len = 2;
                else if (next == 'u')
                    len = 4;
                else if (next == 'U')
                    len = 8;
                char[8] unicodeHex;
                enforce!PickleException(s.length >= len);
                unicodeHex[0 .. len] = s[0 .. len];
                s = s[len .. $];
                char[] unicodeHexRef = unicodeHex[0 .. len];
                next = cast(dchar) parse!uint(unicodeHexRef, 16);
                if (unicodeHexRef.length)
                    throw new PickleException(text("escape sequence ", next, " ",
                            unicodeHex[0 .. len], "not completely parsed"));
                if (len == 2)
                    r ~= cast(char) next;
                else
                    r ~= next;
            }
            else
            {
                foreach (s2; escapeSequences)
                {
                    if (next == s2[1])
                        next = s2[0];
                }
                r ~= next;
            }
        }
        else
            r ~= c;
    }
    return r;
}

/**
Type of item in parsed pickle data
*/
enum ItemType
{
    unknown,
    none,
    int_,
    float_,
    bool_,
    bytes,
    bytearray,
    str,
    list,
    tuple,
    buffer,
    frozenset,
    set,
    dict,
    global,
    reduce,
    obj,
    persid,
    ext
}

/**
Item in tree of parsed pickle data
*/
struct Item
{
    /**
    Type of item, which determines how other fields are interpreted
    */
    ItemType type;

    /**
    Data for e.g. numbers and strings
    */
    const(ubyte)[] data;

    /**
    Child items for e.g. lists
    */
    Item*[] childs;

    /**
    Key value pairs for dictionaries and objects
    */
    Item*[2][] dictChilds;

    /**
    Extra child used with build operation
    */
    Item* buildState;

    const(char)[] toStr() @safe
    in(type.among(ItemType.str, ItemType.global))
    {
        return cast(const(char)[]) data;
    }

    void toString(scope void delegate(const(char)[]) @safe sink, size_t depth = 0) const @safe
    {
        bool multiline = childs.length + dictChilds.length > 3;
        bool needsComma;
        void printWhitespace(bool inList)
        {
            if (multiline)
            {
                sink("\n");
                if (inList)
                    sink("  ");
                foreach (_; 0 .. depth)
                    sink("  ");
            }
            else if(inList && needsComma)
                sink(" ");
        }
        switch (type)
        {
        case ItemType.none:
            sink("None");
            break;
        case ItemType.bool_:
            sink(data[0] ? "True" : "False");
            break;
        case ItemType.int_:
            sink(cast(const(char)[]) data);
            break;
        case ItemType.bytes:
            sink("b\"");
            sink(escapeBytes(cast(const(char)[]) data));
            sink("\"");
            break;
        case ItemType.str:
            sink("\"");
            sink(escapeString(cast(const(char)[]) data));
            sink("\"");
            break;
        case ItemType.list:
            assert (data.length == 0);
            sink("[");
            foreach (c; childs)
            {
                if (needsComma)
                    sink(",");
                printWhitespace(true);
                c.toString(sink, depth + 1);
                needsComma = true;
            }
            printWhitespace(false);
            sink("]");
            break;
        case ItemType.tuple:
            assert (data.length == 0);
            sink("(");
            foreach (c; childs)
            {
                if (needsComma)
                    sink(",");
                printWhitespace(true);
                c.toString(sink, depth + 1);
                needsComma = true;
            }
            if (childs.length < 2)
                sink(",");
            printWhitespace(false);
            sink(")");
            break;
        case ItemType.dict:
            assert (data.length == 0);
            sink("{");
            foreach (c; dictChilds)
            {
                if (needsComma)
                    sink(",");
                printWhitespace(true);
                c[0].toString(sink, depth + 1);
                sink(": ");
                c[1].toString(sink, depth + 1);
                needsComma = true;
            }
            printWhitespace(false);
            sink("}");
            break;
        default:
            sink(text(type));
            sink("(");
            if (data.length)
            {
                sink("b\"");
                sink(escapeBytes(cast(const(char)[]) data));
                sink("\"");
                needsComma = true;
            }
            foreach (c; childs)
            {
                if (needsComma)
                    sink(",");
                printWhitespace(true);
                c.toString(sink, depth + 1);
                needsComma = true;
            }
            foreach (c; dictChilds)
            {
                if (needsComma)
                    sink(",");
                printWhitespace(true);
                c[0].toString(sink, depth + 1);
                sink(": ");
                c[1].toString(sink, depth + 1);
                needsComma = true;
            }
            if (buildState !is null)
            {
                if (needsComma)
                    sink(",");
                printWhitespace(true);
                sink("buildState=");
                buildState.toString(sink, depth + 1);
                needsComma = true;
            }
            printWhitespace(false);
            sink(")");
        }
    }

    int opCmp(ref const Item other) const @safe
    {
        if (type < other.type)
            return -1;
        if (type > other.type)
            return 1;
        if (type == ItemType.str)
        {
            return cmp(cast(const(char)[]) data, cast(const(char)[]) other.data);
        }
        if (type == ItemType.bytes)
        {
            return cmp(data, other.data);
        }
        return 0;
    }
}

/**
Sort dictionary entries for all dictionaries in an item tree
*/
void sortDicts(Item* item) @safe
{
    bool[Item*] done;
    void visit(Item* item)
    {
        if (item is null)
            return;
        if (item in done)
            return;
        done[item] = true;
        if (item.type == ItemType.dict)
        {
            item.dictChilds.sort!((a, b) {
                return a[0].opCmp(*b[0]) < 0;
            });
        }
        foreach (c; item.childs)
            sortDicts(c);
        foreach (c; item.dictChilds)
        {
            sortDicts(c[0]);
            sortDicts(c[1]);
        }
        sortDicts(item.buildState);
    }
    visit(item);
}

private const(char)[][2] compatMapName(const(char)[] mod, const(char)[] name) @safe
{
    switch(mod)
    {
    case "BaseHTTPServer": return ["http.server", name];
    case "CGIHTTPServer": return ["http.server", name];
    case "ConfigParser": return ["configparser", name];
    case "Cookie": return ["http.cookies", name];
    case "Dialog": return ["tkinter.dialog", name];
    case "DocXMLRPCServer": return ["xmlrpc.server", name];
    case "FileDialog": return ["tkinter.filedialog", name];
    case "HTMLParser": return ["html.parser", name];
    case "Queue": return ["queue", name];
    case "ScrolledText": return ["tkinter.scrolledtext", name];
    case "SimpleDialog": return ["tkinter.simpledialog", name];
    case "SimpleHTTPServer": return ["http.server", name];
    case "SimpleXMLRPCServer": return ["xmlrpc.server", name];
    case "SocketServer": return ["socketserver", name];
    case "StringIO": return ["io", name];
    case "Tix": return ["tkinter.tix", name];
    case "Tkconstants": return ["tkinter.constants", name];
    case "Tkdnd": return ["tkinter.dnd", name];
    case "Tkinter": return ["tkinter", name];
    case "UserDict":
        switch(name)
        {
        case "IterableUserDict": return ["collections", "UserDict"];
        case "UserDict": return ["collections", "UserDict"];
        default: return ["collections", name];
        }
    case "UserList":
        switch(name)
        {
        case "UserList": return ["collections", "UserList"];
        default: return ["collections", name];
        }
    case "UserString":
        switch(name)
        {
        case "UserString": return ["collections", "UserString"];
        default: return ["collections", name];
        }
    case "__builtin__":
        switch(name)
        {
        case "basestring": return ["builtins", "str"];
        case "intern": return ["sys", "intern"];
        case "long": return ["builtins", "int"];
        case "reduce": return ["functools", "reduce"];
        case "unichr": return ["builtins", "chr"];
        case "unicode": return ["builtins", "str"];
        case "xrange": return ["builtins", "range"];
        default: return ["builtins", name];
        }
    case "_abcoll": return ["collections.abc", name];
    case "_elementtree": return ["xml.etree.ElementTree", name];
    case "_multiprocessing":
        switch(name)
        {
        case "Connection": return ["multiprocessing.connection", "Connection"];
        default: return [mod, name];
        }
    case "_socket":
        switch(name)
        {
        case "fromfd": return ["socket", "fromfd"];
        default: return [mod, name];
        }
    case "_winreg": return ["winreg", name];
    case "anydbm": return ["dbm", name];
    case "cPickle": return ["pickle", name];
    case "cStringIO": return ["io", name];
    case "commands": return ["subprocess", name];
    case "cookielib": return ["http.cookiejar", name];
    case "copy_reg": return ["copyreg", name];
    case "dbhash": return ["dbm.bsd", name];
    case "dbm": return ["dbm.ndbm", name];
    case "dumbdbm": return ["dbm.dumb", name];
    case "dummy_thread": return ["_dummy_thread", name];
    case "exceptions":
        switch(name)
        {
        case "ArithmeticError": return ["builtins", "ArithmeticError"];
        case "AssertionError": return ["builtins", "AssertionError"];
        case "AttributeError": return ["builtins", "AttributeError"];
        case "BaseException": return ["builtins", "BaseException"];
        case "BufferError": return ["builtins", "BufferError"];
        case "BytesWarning": return ["builtins", "BytesWarning"];
        case "DeprecationWarning": return ["builtins", "DeprecationWarning"];
        case "EOFError": return ["builtins", "EOFError"];
        case "EnvironmentError": return ["builtins", "EnvironmentError"];
        case "Exception": return ["builtins", "Exception"];
        case "FloatingPointError": return ["builtins", "FloatingPointError"];
        case "FutureWarning": return ["builtins", "FutureWarning"];
        case "GeneratorExit": return ["builtins", "GeneratorExit"];
        case "IOError": return ["builtins", "IOError"];
        case "ImportError": return ["builtins", "ImportError"];
        case "ImportWarning": return ["builtins", "ImportWarning"];
        case "IndentationError": return ["builtins", "IndentationError"];
        case "IndexError": return ["builtins", "IndexError"];
        case "KeyError": return ["builtins", "KeyError"];
        case "KeyboardInterrupt": return ["builtins", "KeyboardInterrupt"];
        case "LookupError": return ["builtins", "LookupError"];
        case "MemoryError": return ["builtins", "MemoryError"];
        case "NameError": return ["builtins", "NameError"];
        case "NotImplementedError": return ["builtins", "NotImplementedError"];
        case "OSError": return ["builtins", "OSError"];
        case "OverflowError": return ["builtins", "OverflowError"];
        case "PendingDeprecationWarning": return ["builtins", "PendingDeprecationWarning"];
        case "ReferenceError": return ["builtins", "ReferenceError"];
        case "RuntimeError": return ["builtins", "RuntimeError"];
        case "RuntimeWarning": return ["builtins", "RuntimeWarning"];
        case "StandardError": return ["builtins", "Exception"];
        case "StopIteration": return ["builtins", "StopIteration"];
        case "SyntaxError": return ["builtins", "SyntaxError"];
        case "SyntaxWarning": return ["builtins", "SyntaxWarning"];
        case "SystemError": return ["builtins", "SystemError"];
        case "SystemExit": return ["builtins", "SystemExit"];
        case "TabError": return ["builtins", "TabError"];
        case "TypeError": return ["builtins", "TypeError"];
        case "UnboundLocalError": return ["builtins", "UnboundLocalError"];
        case "UnicodeDecodeError": return ["builtins", "UnicodeDecodeError"];
        case "UnicodeEncodeError": return ["builtins", "UnicodeEncodeError"];
        case "UnicodeError": return ["builtins", "UnicodeError"];
        case "UnicodeTranslateError": return ["builtins", "UnicodeTranslateError"];
        case "UnicodeWarning": return ["builtins", "UnicodeWarning"];
        case "UserWarning": return ["builtins", "UserWarning"];
        case "ValueError": return ["builtins", "ValueError"];
        case "Warning": return ["builtins", "Warning"];
        case "ZeroDivisionError": return ["builtins", "ZeroDivisionError"];
        default: return [mod, name];
        }
    case "gdbm": return ["dbm.gnu", name];
    case "htmlentitydefs": return ["html.entities", name];
    case "httplib": return ["http.client", name];
    case "itertools":
        switch(name)
        {
        case "ifilter": return ["builtins", "filter"];
        case "ifilterfalse": return ["itertools", "filterfalse"];
        case "imap": return ["builtins", "map"];
        case "izip": return ["builtins", "zip"];
        case "izip_longest": return ["itertools", "zip_longest"];
        default: return [mod, name];
        }
    case "markupbase": return ["_markupbase", name];
    case "multiprocessing":
        switch(name)
        {
        case "AuthenticationError": return ["multiprocessing.context", "AuthenticationError"];
        case "BufferTooShort": return ["multiprocessing.context", "BufferTooShort"];
        case "ProcessError": return ["multiprocessing.context", "ProcessError"];
        case "TimeoutError": return ["multiprocessing.context", "TimeoutError"];
        default: return [mod, name];
        }
    case "multiprocessing.forking":
        switch(name)
        {
        case "Popen": return ["multiprocessing.popen_fork", "Popen"];
        default: return [mod, name];
        }
    case "multiprocessing.process":
        switch(name)
        {
        case "Process": return ["multiprocessing.context", "Process"];
        default: return [mod, name];
        }
    case "repr": return ["reprlib", name];
    case "robotparser": return ["urllib.robotparser", name];
    case "socket":
        switch(name)
        {
        case "_socketobject": return ["socket", "SocketType"];
        default: return [mod, name];
        }
    case "test.test_support": return ["test.support", name];
    case "thread": return ["_thread", name];
    case "tkColorChooser": return ["tkinter.colorchooser", name];
    case "tkCommonDialog": return ["tkinter.commondialog", name];
    case "tkFileDialog": return ["tkinter.filedialog", name];
    case "tkFont": return ["tkinter.font", name];
    case "tkMessageBox": return ["tkinter.messagebox", name];
    case "tkSimpleDialog": return ["tkinter.simpledialog", name];
    case "ttk": return ["tkinter.ttk", name];
    case "urllib":
        switch(name)
        {
        case "ContentTooShortError": return ["urllib.error", "ContentTooShortError"];
        case "getproxies": return ["urllib.request", "getproxies"];
        case "pathname2url": return ["urllib.request", "pathname2url"];
        case "quote": return ["urllib.parse", "quote"];
        case "quote_plus": return ["urllib.parse", "quote_plus"];
        case "unquote": return ["urllib.parse", "unquote"];
        case "unquote_plus": return ["urllib.parse", "unquote_plus"];
        case "url2pathname": return ["urllib.request", "url2pathname"];
        case "urlcleanup": return ["urllib.request", "urlcleanup"];
        case "urlencode": return ["urllib.parse", "urlencode"];
        case "urlopen": return ["urllib.request", "urlopen"];
        case "urlretrieve": return ["urllib.request", "urlretrieve"];
        default: return [mod, name];
        }
    case "urllib2":
        switch(name)
        {
        case "HTTPError": return ["urllib.error", "HTTPError"];
        case "URLError": return ["urllib.error", "URLError"];
        default: return ["urllib.request", name];
        }
    case "urlparse": return ["urllib.parse", name];
    case "whichdb":
        switch(name)
        {
        case "whichdb": return ["dbm", "whichdb"];
        default: return ["dbm", name];
        }
    case "xmlrpclib": return ["xmlrpc.client", name];
    default: return [mod, name];
    }
}

/**
Encoding used for strings with unknown encoding
*/
enum Encoding
{
    utf8, /// UTF-8 encoding
    bytes /// Interpret old strings as byte arrays without decoding
}

private struct PickleParser
{
    StorageReader reader;
    Encoding encoding;
    ubyte proto;

    auto read(ulong length, ReadFlags flags) @safe
    {
        assert(length < size_t.max);
        auto data = reader.read(cast(size_t) length, flags);
        return data;
    }

    auto readUntilNewline() @safe
    {
        for (size_t i = 0; ; i++)
        {
            if (read(i + 1, ReadFlags.temporary | ReadFlags.peek)[i] == '\n')
            {
                return read(i + 1, ReadFlags.none)[0 .. i];
            }
        }
    }

    T readLE(T)() @trusted
    {
        auto data = read(T.sizeof, ReadFlags.temporary);
        T r;
        version(LittleEndian)
            (cast(ubyte*) &r)[0 .. T.sizeof] = data[];
        else
            foreach (i; 0 .. T.sizeof)
                (cast(ubyte*) &r)[T.sizeof - 1 - i] = data[i];
        return r;
    }

    T readBE(T)() @trusted
    {
        auto data = read(T.sizeof, ReadFlags.temporary);
        T r;
        version(BigEndian)
            (cast(ubyte*) &r)[0 .. T.sizeof] = data[];
        else
            foreach (i; 0 .. T.sizeof)
                (cast(ubyte*) &r)[T.sizeof - 1 - i] = data[i];
        return r;
    }

    Appender!(Item*[]) stack;
    Appender!(size_t[]) marks;
    Item*[ulong] memo;

    size_t lastMark() @safe
    {
        if (marks.data.length)
            return marks.data[$ - 1];
        return 0;
    }

    void push(Item* item) @safe
    {
        stack.put(item);
    }

    Item* pop() @safe
    {
        enforce!PickleException(stack.data.length > lastMark);
        auto r = stack.data[$ - 1];
        stack.shrinkTo(stack.data.length - 1);
        return r;
    }

    Item*[] popMark() @safe
    {
        enforce!PickleException(marks.data.length);
        size_t mark = marks.data[$ - 1];
        marks.shrinkTo(marks.data.length - 1);
        auto r = stack.data[mark .. $].dup;
        stack.shrinkTo(mark);
        return r;
    }

    Item* top() @safe
    {
        enforce!PickleException(stack.data.length > lastMark);
        auto r = stack.data[$ - 1];
        return r;
    }

    @Opcode(0x91, 4) void loadFrozenset() @safe
    {
        Item*[] stack0 = popMark();
        Item* r = new Item(ItemType.frozenset);
        r.childs = stack0;
        push(r);
    }

    @Opcode(0x8f, 4) void loadEmptySet() @safe
    {
        Item* r = new Item(ItemType.set);
        push(r);
    }

    @Opcode('a', 0) void loadAppend() @safe
    {
        Item* stack1 = pop();
        Item* stack0 = pop();
        enforce!PickleException(stack0.type == ItemType.list);
        stack0.childs ~= stack1;
        push(stack0);
    }

    @Opcode('e', 1) void loadAppends() @safe
    {
        Item*[] stack1 = popMark();
        Item* stack0 = pop();
        enforce!PickleException(stack0.type == ItemType.list);
        stack0.childs ~= stack1;
        push(stack0);
    }

    @Opcode(0x97, 5) void loadNextBuffer() @safe
    {
        Item* r = new Item(ItemType.buffer);
        push(r);
    }

    @Opcode(0x98, 5) void loadReadonlyBuffer() @safe
    {
        Item* stack0 = pop();
        //enforce!PickleException(stack0.type == ItemType.buffer);
        Item* r = new Item(ItemType.buffer);
        r.childs = [stack0];
        push(r);
    }

    @Opcode('b', 0) void loadBuild() @safe
    {
        Item* stack1 = pop();
        Item* stack0 = pop();
        stack0.buildState = stack1;
        push(stack0);
    }

    @Opcode(0x96, 5) void loadBytearray8() @safe
    {
        auto data = read(readLE!ulong, ReadFlags.none);
        Item* r = new Item(ItemType.bytearray);
        r.data = data;
        push(r);
    }

    @Opcode('B', 3) void loadBinBytes() @safe
    {
        auto data = read(readLE!uint, ReadFlags.none);
        Item* r = new Item(ItemType.bytes);
        r.data = data;
        push(r);
    }

    @Opcode('C', 3) void loadShortBinBytes() @safe
    {
        auto data = read(readLE!ubyte(), ReadFlags.none);
        Item* r = new Item(ItemType.bytes);
        r.data = data;
        push(r);
    }

    @Opcode(0x8e, 4) void loadBinBytes8() @safe
    {
        auto data = read(readLE!ulong, ReadFlags.none);
        Item* r = new Item(ItemType.bytes);
        r.data = data;
        push(r);
    }

    @Opcode('d', 0) void loadDict() @safe
    {
        Item*[] stack0 = popMark();
        enforce!PickleException(stack0.length % 2 == 0);
        Item* r = new Item(ItemType.dict);
        r.dictChilds = new Item*[2][stack0.length / 2];
        foreach (i; 0 .. stack0.length / 2)
            r.dictChilds[i] = [stack0[i * 2], stack0[i * 2 + 1]];
        push(r);
    }

    @Opcode('}', 1) void loadEmptyDict() @safe
    {
        Item* r = new Item(ItemType.dict);
        push(r);
    }

    @Opcode('2', 0) void loadDup() @safe
    {
        push(top());
    }

    @Opcode(0x82, 2) void loadExt1() @safe
    {
        auto data = readLE!ubyte();
        Item* r = new Item(ItemType.ext);
        r.data = cast(const(ubyte)[]) text(data);
        push(r);
    }

    @Opcode(0x83, 2) void loadExt2() @safe
    {
        auto data = readLE!ushort();
        Item* r = new Item(ItemType.ext);
        r.data = cast(const(ubyte)[]) text(data);
        push(r);
    }

    @Opcode(0x84, 2) void loadExt4() @safe
    {
        auto data = readLE!int();
        Item* r = new Item(ItemType.ext);
        r.data = cast(const(ubyte)[]) text(data);
        push(r);
    }

    @Opcode('F', 0) void loadFloat() @safe
    {
        auto data = readUntilNewline();
        Item* r = new Item(ItemType.float_);
        r.data = cast(const(ubyte)[]) format("%f", (cast(const(char)[]) data).to!double);
        push(r);
    }

    @Opcode('G', 1) void loadBinFloat() @safe
    {
        auto data = readBE!double();
        Item* r = new Item(ItemType.float_);
        r.data = cast(const(ubyte)[]) format("%f", data);
        push(r);
    }

    @Opcode(0x95, 4) void loadFrame() @safe
    {
        auto data = readLE!ulong();
        read(data, ReadFlags.peek | ReadFlags.temporary | ReadFlags.allowEmpty | ReadFlags.allowPartial);
    }

    @Opcode('g', 0) void loadGet() @safe
    {
        auto data = readUntilNewline();
        auto index = (cast(const(char)[]) data).to!ulong;
        auto entry = index in memo;
        enforce!PickleException(entry);
        push(*entry);
    }

    @Opcode('h', 1) void loadBinGet() @safe
    {
        auto data = readLE!ubyte();
        auto entry = data in memo;
        enforce!PickleException(entry);
        push(*entry);
    }

    @Opcode('j', 1) void loadLongBinGet() @safe
    {
        auto data = readLE!uint();
        auto entry = data in memo;
        enforce!PickleException(entry);
        push(*entry);
    }

    @Opcode('c', 0) void loadGlobal() @safe
    {
        auto data = readUntilNewline();
        auto data2 = readUntilNewline();
        Item* r = new Item(ItemType.global);
        const(char)[][2] mapped = [cast(const(char)[]) data, cast(const(char)[]) data2];
        if (this.proto < 3)
            mapped = compatMapName(mapped[0], mapped[1]);
        r.data = cast(immutable(ubyte)[]) text(mapped[0], ".", mapped[1]);
        push(r);
    }

    @Opcode(0x93, 4) void loadStackGlobal() @safe
    {
        Item* stack1 = pop();
        enforce!PickleException(stack1.type == ItemType.str);
        Item* stack0 = pop();
        enforce!PickleException(stack0.type == ItemType.str);
        Item* r = new Item(ItemType.global);
        const(char)[][2] mapped = [cast(const(char)[]) stack0.data, cast(const(char)[]) stack1.data];
        if (this.proto < 3)
            mapped = compatMapName(mapped[0], mapped[1]);
        r.data = cast(immutable(ubyte)[]) text(mapped[0], ".", mapped[1]);
        push(r);
    }

    @Opcode('i', 0) void loadInst() @safe
    {
        auto data = readUntilNewline();
        auto data2 = readUntilNewline();
        Item*[] stack0 = popMark();
        Item* g = new Item(ItemType.global);
        const(char)[][2] mapped = [cast(const(char)[]) data, cast(const(char)[]) data2];
        if (this.proto < 3)
            mapped = compatMapName(mapped[0], mapped[1]);
        g.data = cast(immutable(ubyte)[]) text(mapped[0], ".", mapped[1]);
        Item* r = new Item(ItemType.obj);
        r.childs = g ~ stack0;
        push(r);
    }

    @Opcode('I', 0) void loadInt() @safe
    {
        auto data = readUntilNewline();
        if (data.endsWith("L"))
            data = data[0 .. $ - 1];
        Item* r;
        if (data == "00" || data == "01")
        {
            r = new Item(ItemType.bool_);
            r.data = [data[1] != '0'];
        }
        else
        {
            r = new Item(ItemType.int_);
            (cast(const(char)[]) data).to!long;
            r.data = data;
        }
        push(r);
    }

    @Opcode('J', 1) void loadBinInt() @safe
    {
        auto data = readLE!int();
        Item* r = new Item(ItemType.int_);
        r.data = cast(const(ubyte)[]) text(data);
        push(r);
    }

    @Opcode('K', 1) void loadBinInt1() @safe
    {
        auto data = readLE!ubyte();
        Item* r = new Item(ItemType.int_);
        r.data = cast(const(ubyte)[]) text(data);
        push(r);
    }

    @Opcode('M', 1) void loadBinInt2() @safe
    {
        auto data = readLE!ushort();
        Item* r = new Item(ItemType.int_);
        r.data = cast(const(ubyte)[]) text(data);
        push(r);
    }

    @Opcode('s', 0) void loadSetItem() @safe
    {
        Item* stack2 = pop();
        Item* stack1 = pop();
        Item* stack0 = pop();
        //enforce!PickleException(stack0.type == ItemType.dict);
        stack0.dictChilds ~= [stack1, stack2];
        push(stack0);
    }

    @Opcode(0x90, 4) void loadAddItems() @safe
    {
        Item*[] stack1 = popMark();
        Item* stack0 = pop();
        enforce!PickleException(stack0.type == ItemType.set);
        stack0.childs ~= stack1;
        push(stack0);
    }

    @Opcode('u', 1) void loadSetItems() @safe
    {
        Item*[] stack1 = popMark();
        Item* stack0 = pop();
        enforce!PickleException(stack1.length % 2 == 0);
        //enforce!PickleException(stack0.type == ItemType.dict);
        stack0.dictChilds = new Item*[2][stack1.length / 2];
        foreach (i; 0 .. stack1.length / 2)
            stack0.dictChilds[i] = [stack1[i * 2], stack1[i * 2 + 1]];
        push(stack0);
    }

    @Opcode('l', 0) void loadList() @safe
    {
        Item*[] stack0 = popMark();
        Item* r = new Item(ItemType.list);
        r.childs = stack0;
        push(r);
    }

    @Opcode(']', 1) void loadEmptyList() @safe
    {
        Item* r = new Item(ItemType.list);
        push(r);
    }

    @Opcode('L', 0) void loadLong() @safe
    {
        auto data = readUntilNewline();
        if (data.endsWith("L"))
            data = data[0 .. $ - 1];
        Item* r = new Item(ItemType.int_);
        r.data = data;
        push(r);
    }

    static string bigIntToString(const(ubyte)[] data) @safe
    {
        if (data.length && data[$ - 1] & 0x80)
        {
            auto b = BigInt(true, iota(data.length, 0, -1).map!(i => cast(ubyte)(~data[i - 1]))) - 1;
            return text(b);
        }
        else
        {
            auto b = BigInt(false, iota(data.length, 0, -1).map!(i => data[i - 1]));
            return text(b);
        }
    }

    @Opcode(0x8a, 2) void loadLong1() @safe
    {
        auto data = read(readLE!ubyte(), ReadFlags.temporary);
        Item* r = new Item(ItemType.int_);
        r.data = cast(const(ubyte)[]) bigIntToString(data);
        push(r);
    }

    @Opcode(0x8b, 2) void loadLong4() @safe
    {
        auto len = readLE!int;
        enforce!PickleException(len >= 0);
        auto data = read(len, ReadFlags.temporary);
        Item* r = new Item(ItemType.int_);
        r.data = cast(const(ubyte)[]) bigIntToString(data);
        push(r);
    }

    @Opcode('(', 0) void loadMark() @safe
    {
        marks.put(stack.data.length);
    }

    @Opcode('1', 1) void loadPopMark() @safe
    {
        popMark();
    }

    @Opcode(0x94, 4) void loadMemoize() @safe
    {
        memo[memo.length] = top();
    }

    @Opcode(0x89, 2) void loadNewFalse() @safe
    {
        Item* r = new Item(ItemType.bool_);
        r.data = [0];
        push(r);
    }

    @Opcode(0x88, 2) void loadNewTrue() @safe
    {
        Item* r = new Item(ItemType.bool_);
        r.data = [1];
        push(r);
    }

    @Opcode('N', 0) void loadNone() @safe
    {
        Item* r = new Item(ItemType.none);
        push(r);
    }

    @Opcode('o', 1) void loadObj() @safe
    {
        Item*[] stack0 = popMark();
        Item* r = new Item(ItemType.obj);
        enforce!PickleException(stack0.length);
        r.childs = stack0[];
        push(r);
    }

    @Opcode(0x81, 2) void loadNewObj() @safe
    {
        Item* stack1 = pop();
        Item* stack0 = pop();
        Item* r = new Item(ItemType.obj);
        r.childs = [stack0, stack1];
        push(r);
    }

    @Opcode(0x92, 4) void loadNewObjEx() @safe
    {
        Item* stack2 = pop();
        Item* stack1 = pop();
        Item* stack0 = pop();
        Item* r = new Item(ItemType.obj);
        r.childs = [stack0, stack1, stack2];
        push(r);
    }

    @Opcode('P', 0) void loadPersid() @safe
    {
        auto data = readUntilNewline();
        Item* r = new Item(ItemType.persid);
        r.data = data;
        push(r);
    }

    @Opcode('Q', 1) void loadBinPersid() @safe
    {
        Item* stack0 = pop();
        Item* r = new Item(ItemType.persid);
        r.childs = [stack0];
        push(r);
    }

    @Opcode('0', 0) void loadPop() @safe
    {
        /*Item* stack0 = */pop();
    }

    @Opcode(0x80, 2) void loadProto() @safe
    {
        this.proto = readLE!ubyte();
    }

    @Opcode('p', 0) void loadPut() @safe
    {
        auto data = readUntilNewline();
        memo[(cast(const(char)[]) data).to!ulong] = top();
    }

    @Opcode('q', 1) void loadBinPut() @safe
    {
        auto data = readLE!ubyte();
        memo[data] = top();
    }

    @Opcode('r', 1) void loadLongBinPut() @safe
    {
        auto data = readLE!uint();
        memo[data] = top();
    }

    @Opcode('R', 0) void loadReduce() @safe
    {
        Item* stack1 = pop();
        Item* stack0 = pop();
        Item* r;
        if (stack0.type == ItemType.global
            && stack0.data == "copyreg._reconstructor"
            && stack1.type == ItemType.tuple
            && stack1.childs.length == 3
            && stack1.childs[0].type == ItemType.global
            && stack1.childs[1].type == ItemType.global
            && stack1.childs[1].data == "builtins.object"
            && stack1.childs[2].type == ItemType.none)
        {
            r = new Item(ItemType.obj);
            r.childs = [stack1.childs[0], new Item(ItemType.tuple)];
        }
        else
        {
            r = new Item(ItemType.reduce);
            r.childs = [stack0, stack1];
        }
        push(r);
    }

    @Opcode('S', 0) void loadString() @safe
    {
        auto data = readUntilNewline();
        enforce!PickleException(data.length >= 2 && data[0] == data[$ - 1] && data[0].among('\'', '"'));
        data = cast(const(ubyte)[]) unescapeString(cast(const(char)[]) data[1 .. $ - 1]);
        Item* r = new Item(encoding == Encoding.bytes ? ItemType.bytes : ItemType.str);
        r.data = data;
        push(r);
    }

    @Opcode('T', 1) void loadBinString() @safe
    {
        auto len = readLE!int;
        enforce!PickleException(len >= 0);
        auto data = read(len, ReadFlags.none);
        Item* r = new Item(encoding == Encoding.bytes ? ItemType.bytes : ItemType.str);
        r.data = data;
        push(r);
    }

    @Opcode('U', 1) void loadShortBinString() @safe
    {
        auto data = read(readLE!ubyte(), ReadFlags.none);
        Item* r = new Item(encoding == Encoding.bytes ? ItemType.bytes : ItemType.str);
        r.data = data;
        push(r);
    }

    @Opcode('t', 0) void loadTuple() @safe
    {
        Item*[] stack0 = popMark();
        Item* r = new Item(ItemType.tuple);
        r.childs = stack0;
        push(r);
    }

    @Opcode(')', 1) void loadEmptyTuple() @safe
    {
        Item* r = new Item(ItemType.tuple);
        push(r);
    }

    @Opcode(0x85, 2) void loadTuple1() @safe
    {
        Item* stack0 = pop();
        Item* r = new Item(ItemType.tuple);
        r.childs = [stack0];
        push(r);
    }

    @Opcode(0x86, 2) void loadTuple2() @safe
    {
        Item* stack1 = pop();
        Item* stack0 = pop();
        Item* r = new Item(ItemType.tuple);
        r.childs = [stack0, stack1];
        push(r);
    }

    @Opcode(0x87, 2) void loadTuple3() @safe
    {
        Item* stack2 = pop();
        Item* stack1 = pop();
        Item* stack0 = pop();
        Item* r = new Item(ItemType.tuple);
        r.childs = [stack0, stack1, stack2];
        push(r);
    }

    @Opcode('V', 0) void loadUnicode() @safe
    {
        auto data = readUntilNewline();
        Item* r = new Item(ItemType.str);
        r.data = cast(const(ubyte)[]) unescapeString(cast(const(char)[]) data);
        push(r);
    }

    @Opcode('X', 1) void loadBinUnicode() @safe
    {
        auto data = read(readLE!uint, ReadFlags.none);
        Item* r = new Item(ItemType.str);
        r.data = data;
        push(r);
    }

    @Opcode(0x8c, 4) void loadShortBinUnicode() @safe
    {
        auto data = read(readLE!ubyte(), ReadFlags.none);
        Item* r = new Item(ItemType.str);
        r.data = data;
        push(r);
    }

    @Opcode(0x8d, 4) void loadBinUnicode8() @safe
    {
        auto data = read(readLE!ulong, ReadFlags.none);
        Item* r = new Item(ItemType.str);
        r.data = data;
        push(r);
    }

    static string generateSwitchCode() @safe
    {
        string code;
        static foreach (name; __traits(allMembers, typeof(this)))
            static if (name.startsWith("load"))
            {{
                enum Opcode opcode = __traits(getAttributes, __traits(getMember, typeof(this), name))[0];
                code ~= text("case ", opcode.code, ":\n");
                code ~= text(name, "();\n");
                code ~= text("break;\n");
            }}

        return code;
    }

    Item* parse() @safe
    {
        while (true)
        {
            ubyte opcode = readLE!ubyte;
            switch (opcode)
            {
                mixin(generateSwitchCode());
            case '.':
                enforce!PickleException(marks.data.length == 0);
                enforce!PickleException(stack.data.length == 1);
                return stack.data[0];
            default:
                enforce!PickleException(false, format("Unknown opcode 0x%02x", opcode));
            }
        }
    }
}

/**
Parse pickle data and return item tree

Params:
    reader = Reader for pickle data
    encoding = Encoding used for strings with unknown encoding
*/
Item* parsePickle(StorageReader reader, Encoding encoding = Encoding.utf8) @safe
{
    PickleParser parser;
    parser.reader = reader;
    parser.encoding = encoding;
    return parser.parse();
}

/**
Parse pickle data and return item tree

Params:
    buffer = Buffer with pickle data
    encoding = Encoding used for strings with unknown encoding
*/
Item* parsePickle(const(ubyte)[] buffer, Encoding encoding = Encoding.utf8) @safe
{
    auto reader = new MemoryStorage(cast(const(ubyte)[]) buffer);
    return parsePickle(reader, encoding);
}
