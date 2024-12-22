import std.conv;
import std.stdio;
import tensorformats.pickle;
import tensorformats.storage;

// Many tests are based on tests from https://github.com/python/cpython/blob/main/Lib/test/pickletester.py

void testPickle(string buffer, string expected, Encoding encoding = Encoding.utf8) @safe
{
    Item* item = parsePickle(cast(const(ubyte)[]) buffer, encoding);
    sortDicts(item);
    string s = text(*item);
    if (s != expected)
        writeln("got: ", s, "\nexpected: ", expected);
    assert(s == expected);
}

void testPickleBad(string buffer) @safe
{
    bool hasException;
    try
    {
        parsePickle(cast(const(ubyte)[]) buffer);
    }
    catch(PickleException e)
    {
        hasException = true;
    }
    catch(StorageException e)
    {
        hasException = true;
    }
    catch(ConvException e)
    {
        hasException = true;
    }
    assert(hasException);
}

unittest
{
    testPickle("\x88.", "True");
}

enum commonExpectedData = q{[
  0,
  1,
  float_(b"2.000000"),
  reduce(global(b"builtins.complex"), (float_(b"3.000000"), float_(b"0.000000"))),
  1,
  -1,
  255,
  -255,
  -256,
  65535,
  -65535,
  -65536,
  2147483647,
  -2147483647,
  -2147483648,
  (
    "abc",
    "abc",
    obj(global(b"__main__.C"), (,), buildState={"bar": 2, "foo": 1}),
    obj(global(b"__main__.C"), (,), buildState={"bar": 2, "foo": 1})
  ),
  (
    "abc",
    "abc",
    obj(global(b"__main__.C"), (,), buildState={"bar": 2, "foo": 1}),
    obj(global(b"__main__.C"), (,), buildState={"bar": 2, "foo": 1})
  ),
  5
]};

// AbstractUnpickleTests.test_load_from_data0
unittest
{
    testPickle("(lp0\nL0L\naL1L\naF2.0\nac__builtin__\ncomplex\np1\n(F3.0\nF0.0\ntp2\nRp3\naL1L\naL-1L\naL255L\naL-255L\naL-256L\naL65535L\naL-65535L\naL-65536L\naL2147483647L\naL-2147483647L\naL-2147483648L\na(Vabc\np4\ng4\nccopy_reg\n_reconstructor\np5\n(c__main__\nC\np6\nc__builtin__\nobject\np7\nNtp8\nRp9\n(dp10\nVfoo\np11\nL1L\nsVbar\np12\nL2L\nsbg9\ntp13\nag13\naL5L\na.",
               commonExpectedData);
}

// AbstractUnpickleTests.test_load_from_data1
unittest
{
    testPickle("]q\x00(K\x00K\x01G@\x00\x00\x00\x00\x00\x00\x00c__builtin__\ncomplex\nq\x01(G@\x08\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00tq\x02Rq\x03K\x01J\xff\xff\xff\xffK\xffJ\x01\xff\xff\xffJ\x00\xff\xff\xffM\xff\xffJ\x01\x00\xff\xffJ\x00\x00\xff\xffJ\xff\xff\xff\x7fJ\x01\x00\x00\x80J\x00\x00\x00\x80(X\x03\x00\x00\x00abcq\x04h\x04ccopy_reg\n_reconstructor\nq\x05(c__main__\nC\nq\x06c__builtin__\nobject\nq\x07Ntq\x08Rq\t}q\n(X\x03\x00\x00\x00fooq\x0bK\x01X\x03\x00\x00\x00barq\x0cK\x02ubh\ttq\rh\rK\x05e.",
               commonExpectedData);
}

// AbstractUnpickleTests.test_load_from_data2
unittest
{
    testPickle("\x80\x02]q\x00(K\x00K\x01G@\x00\x00\x00\x00\x00\x00\x00c__builtin__\ncomplex\nq\x01G@\x08\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x86q\x02Rq\x03K\x01J\xff\xff\xff\xffK\xffJ\x01\xff\xff\xffJ\x00\xff\xff\xffM\xff\xffJ\x01\x00\xff\xffJ\x00\x00\xff\xffJ\xff\xff\xff\x7fJ\x01\x00\x00\x80J\x00\x00\x00\x80(X\x03\x00\x00\x00abcq\x04h\x04c__main__\nC\nq\x05)\x81q\x06}q\x07(X\x03\x00\x00\x00fooq\x08K\x01X\x03\x00\x00\x00barq\tK\x02ubh\x06tq\nh\nK\x05e.",
               commonExpectedData);
}

// AbstractUnpickleTests.test_load_from_data3
unittest
{
    testPickle("\x80\x03]q\x00(K\x00K\x01G@\x00\x00\x00\x00\x00\x00\x00cbuiltins\ncomplex\nq\x01G@\x08\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x86q\x02Rq\x03K\x01J\xff\xff\xff\xffK\xffJ\x01\xff\xff\xffJ\x00\xff\xff\xffM\xff\xffJ\x01\x00\xff\xffJ\x00\x00\xff\xffJ\xff\xff\xff\x7fJ\x01\x00\x00\x80J\x00\x00\x00\x80(X\x03\x00\x00\x00abcq\x04h\x04c__main__\nC\nq\x05)\x81q\x06}q\x07(X\x03\x00\x00\x00barq\x08K\x02X\x03\x00\x00\x00fooq\tK\x01ubh\x06tq\nh\nK\x05e.",
               commonExpectedData);
}

// AbstractUnpickleTests.test_load_from_data4
unittest
{
    testPickle("\x80\x04\x95\xa8\x00\x00\x00\x00\x00\x00\x00]\x94(K\x00K\x01G@\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x94\x8c\x07complex\x94\x93\x94G@\x08\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00\x86\x94R\x94K\x01J\xff\xff\xff\xffK\xffJ\x01\xff\xff\xffJ\x00\xff\xff\xffM\xff\xffJ\x01\x00\xff\xffJ\x00\x00\xff\xffJ\xff\xff\xff\x7fJ\x01\x00\x00\x80J\x00\x00\x00\x80(\x8c\x03abc\x94h\x06\x8c\x08__main__\x94\x8c\x01C\x94\x93\x94)\x81\x94}\x94(\x8c\x03bar\x94K\x02\x8c\x03foo\x94K\x01ubh\nt\x94h\x0eK\x05e.",
               commonExpectedData);
}

// AbstractUnpickleTests.test_load_classic_instance
unittest
{
    testPickle("(i__main__\nC\np0\n(dp1\nb.",
               q{obj(global(b"__main__.C"), buildState={})});
    testPickle("(c__main__\nC\nq\x00oq\x01}q\x02b.",
               q{obj(global(b"__main__.C"), buildState={})});
    testPickle("\x80\x02(c__main__\nC\nq\x00oq\x01}q\x02b.",
               q{obj(global(b"__main__.C"), buildState={})});
}

// AbstractUnpickleTests.test_maxint64
unittest
{
    testPickle("I9223372036854775807\n.",
               q{9223372036854775807});
    testPickleBad("I9223372036854775807JUNK\n.");
}

// AbstractUnpickleTests.test_unpickle_from_2x
unittest
{
    testPickle("\x80\x02c__builtin__\nset\nq\x00]q\x01(K\x01K\x02e\x85q\x02Rq\x03.",
               q{reduce(global(b"builtins.set"), ([1, 2],))});
    testPickle("\x80\x02c__builtin__\nxrange\nq\x00K\x00K\x05K\x01\x87q\x01Rq\x02.",
               q{reduce(global(b"builtins.range"), (0, 5, 1))});
    testPickle("\x80\x02cCookie\nSimpleCookie\nq\x00)\x81q\x01U\x03keyq\x02cCookie\nMorsel\nq\x03)\x81q\x04(U\x07commentq\x05U\x00q\x06U\x06domainq\x07h\x06U\x06secureq\x08h\x06U\x07expiresq\th\x06U\x07max-ageq\nh\x06U\x07versionq\x0bh\x06U\x04pathq\x0ch\x06U\x08httponlyq\rh\x06u}q\x0e(U\x0bcoded_valueq\x0fU\x05valueq\x10h\x10h\x10h\x02h\x02ubs}q\x11b.",
               q{obj(global(b"http.cookies.SimpleCookie"), (,), "key": obj(
    global(b"http.cookies.Morsel"),
    (,),
    "comment": "",
    "domain": "",
    "secure": "",
    "expires": "",
    "max-age": "",
    "version": "",
    "path": "",
    "httponly": "",
    buildState={"coded_value": "value", "key": "key", "value": "value"}
  ), buildState={})});
    testPickle("\x80\x02cexceptions\nArithmeticError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.ArithmeticError"), (,))});
    testPickle("\x80\x02cexceptions\nAssertionError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.AssertionError"), (,))});
    testPickle("\x80\x02cexceptions\nAttributeError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.AttributeError"), (,))});
    testPickle("\x80\x02cexceptions\nBaseException\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.BaseException"), (,))});
    testPickle("\x80\x02cexceptions\nBufferError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.BufferError"), (,))});
    testPickle("\x80\x02cexceptions\nBytesWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.BytesWarning"), (,))});
    testPickle("\x80\x02cexceptions\nDeprecationWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.DeprecationWarning"), (,))});
    testPickle("\x80\x02cexceptions\nEOFError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.EOFError"), (,))});
    testPickle("\x80\x02cexceptions\nOSError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.OSError"), (,))});
    testPickle("\x80\x02cexceptions\nException\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.Exception"), (,))});
    testPickle("\x80\x02cexceptions\nFloatingPointError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.FloatingPointError"), (,))});
    testPickle("\x80\x02cexceptions\nFutureWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.FutureWarning"), (,))});
    testPickle("\x80\x02cexceptions\nGeneratorExit\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.GeneratorExit"), (,))});
    testPickle("\x80\x02cexceptions\nOSError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.OSError"), (,))});
    testPickle("\x80\x02cexceptions\nImportError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.ImportError"), (,))});
    testPickle("\x80\x02cexceptions\nImportWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.ImportWarning"), (,))});
    testPickle("\x80\x02cexceptions\nIndentationError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.IndentationError"), (,))});
    testPickle("\x80\x02cexceptions\nIndexError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.IndexError"), (,))});
    testPickle("\x80\x02cexceptions\nKeyError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.KeyError"), (,))});
    testPickle("\x80\x02cexceptions\nKeyboardInterrupt\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.KeyboardInterrupt"), (,))});
    testPickle("\x80\x02cexceptions\nLookupError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.LookupError"), (,))});
    testPickle("\x80\x02cexceptions\nMemoryError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.MemoryError"), (,))});
    testPickle("\x80\x02cexceptions\nNameError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.NameError"), (,))});
    testPickle("\x80\x02cexceptions\nNotImplementedError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.NotImplementedError"), (,))});
    testPickle("\x80\x02cexceptions\nOSError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.OSError"), (,))});
    testPickle("\x80\x02cexceptions\nOverflowError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.OverflowError"), (,))});
    testPickle("\x80\x02cexceptions\nPendingDeprecationWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.PendingDeprecationWarning"), (,))});
    testPickle("\x80\x02cexceptions\nReferenceError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.ReferenceError"), (,))});
    testPickle("\x80\x02cexceptions\nRuntimeError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.RuntimeError"), (,))});
    testPickle("\x80\x02cexceptions\nRuntimeWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.RuntimeWarning"), (,))});
    testPickle("\x80\x02cexceptions\nStopIteration\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.StopIteration"), (,))});
    testPickle("\x80\x02cexceptions\nSyntaxError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.SyntaxError"), (,))});
    testPickle("\x80\x02cexceptions\nSyntaxWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.SyntaxWarning"), (,))});
    testPickle("\x80\x02cexceptions\nSystemError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.SystemError"), (,))});
    testPickle("\x80\x02cexceptions\nSystemExit\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.SystemExit"), (,))});
    testPickle("\x80\x02cexceptions\nTabError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.TabError"), (,))});
    testPickle("\x80\x02cexceptions\nTypeError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.TypeError"), (,))});
    testPickle("\x80\x02cexceptions\nUnboundLocalError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.UnboundLocalError"), (,))});
    testPickle("\x80\x02cexceptions\nUnicodeError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.UnicodeError"), (,))});
    testPickle("\x80\x02cexceptions\nUnicodeWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.UnicodeWarning"), (,))});
    testPickle("\x80\x02cexceptions\nUserWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.UserWarning"), (,))});
    testPickle("\x80\x02cexceptions\nValueError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.ValueError"), (,))});
    testPickle("\x80\x02cexceptions\nWarning\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.Warning"), (,))});
    testPickle("\x80\x02cexceptions\nZeroDivisionError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.ZeroDivisionError"), (,))});
    testPickle("\x80\x02cexceptions\nStandardError\nq\x00)Rq\x01.",
               q{reduce(global(b"builtins.Exception"), (,))});
    testPickle("\x80\x02cexceptions\nUnicodeEncodeError\nq\x00(U\x05asciiq\x01X\x03\x00\x00\x00fooq\x02K\x00K\x01U\x03badq\x03tq\x04Rq\x05.",
               q{reduce(global(b"builtins.UnicodeEncodeError"), (
    "ascii",
    "foo",
    0,
    1,
    "bad"
  ))});
}

// AbstractUnpickleTests.test_load_python2_str_as_bytes
unittest
{
    testPickle("S'a\\x00\\xa0'\n.",
               q{b"a\x00\xa0"},
               Encoding.bytes);
    testPickle("U\x03a\x00\xa0.",
               q{b"a\x00\xa0"},
               Encoding.bytes);
    testPickle("\x80\x02U\x03a\x00\xa0.",
               q{b"a\x00\xa0"},
               Encoding.bytes);
}

// AbstractUnpickleTests.test_load_python2_unicode_as_str
unittest
{
    testPickle("V\\u03c0\n.",
               q{"π"},
               Encoding.bytes);
    testPickle("X\x02\x00\x00\x00\xcf\x80.",
               q{"π"},
               Encoding.bytes);
    testPickle("\x80\x02X\x02\x00\x00\x00\xcf\x80.",
               q{"π"},
               Encoding.bytes);
}

// AbstractUnpickleTests.test_load_long_python2_str_as_bytes
unittest
{
    testPickle("T,\x01\x00\x00xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.",
               q{b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
               Encoding.bytes);
}

// AbstractUnpickleTests.test_constants
unittest
{
    testPickle("N.",
               q{None});
    testPickle("\x88.",
               q{True});
    testPickle("\x89.",
               q{False});
    testPickle("I01\n.",
               q{True});
    testPickle("I00\n.",
               q{False});
}

// AbstractUnpickleTests.test_empty_bytestring
unittest
{
    testPickle("\x80\x03U\x00q\x00.",
               q{""},
               Encoding.utf8 /*"koi8-r"*/);
}

// AbstractUnpickleTests.test_short_binbytes
unittest
{
    testPickle("\x80\x03C\x04\xe2\x82\xac\x00.",
               q{b"\xe2\x82\xac\x00"});
}

// AbstractUnpickleTests.test_binbytes
unittest
{
    testPickle("\x80\x03B\x04\x00\x00\x00\xe2\x82\xac\x00.",
               q{b"\xe2\x82\xac\x00"});
}

// AbstractUnpickleTests.test_short_binunicode
unittest
{
    testPickle("\x80\x04\x8c\x04\xe2\x82\xac\x00.",
               q{"€\x00"});
}

// AbstractUnpickleTests.test_misc_get
unittest
{
    testPickleBad("g0\np0");
    testPickleBad("jens:");
    testPickleBad("hens:");
    testPickle("((Kdtp0\nh\x00l.))",
               q{[(100,), (100,)]});
}

// AbstractUnpickleTests.test_binbytes8
unittest
{
    testPickle("\x80\x04\x8e\x04\x00\x00\x00\x00\x00\x00\x00\xe2\x82\xac\x00.",
               q{b"\xe2\x82\xac\x00"});
}

// AbstractUnpickleTests.test_binunicode8
unittest
{
    testPickle("\x80\x04\x8d\x04\x00\x00\x00\x00\x00\x00\x00\xe2\x82\xac\x00.",
               q{"€\x00"});
}

// AbstractUnpickleTests.test_bytearray8
unittest
{
    testPickle("\x80\x05\x96\x03\x00\x00\x00\x00\x00\x00\x00xxx.",
               q{bytearray(b"xxx")});
}

// AbstractUnpickleTests.test_get
unittest
{
    testPickle("((lp100000\ng100000\nt.",
               q{([], [])});
}

// AbstractUnpickleTests.test_binget
unittest
{
    testPickle("(]q\xffh\xfft.",
               q{([], [])});
}

// AbstractUnpickleTests.test_long_binget
unittest
{
    testPickle("(]r\x00\x00\x01\x00j\x00\x00\x01\x00t.",
               q{([], [])});
}

// AbstractUnpickleTests.test_dup
unittest
{
    testPickle("((l2t.",
               q{([], [])});
}

// AbstractUnpickleTests.test_negative_put
unittest
{
    testPickleBad("Va\np-1\n.");
}

// AbstractUnpickleTests.test_badly_escaped_string
unittest
{
    testPickleBad("S'\\'\n.");
}

// AbstractUnpickleTests.test_badly_quoted_string
unittest
{
    testPickleBad("S'\n.");
    testPickleBad("S\"\n.");
    testPickleBad("S' \n.");
    testPickleBad("S\" \n.");
    testPickleBad("S\'\"\n.");
    testPickleBad("S\"\'\n.");
    testPickleBad("S' ' \n.");
    testPickleBad("S\" \" \n.");
    testPickleBad("S ''\n.");
    testPickleBad("S \"\"\n.");
    testPickleBad("S \n.");
    testPickleBad("S\n.");
    testPickleBad("S.");
}

// AbstractUnpickleTests.test_correctly_quoted_string
unittest
{
    testPickle("S''\n.",
               q{""});
    testPickle("S\"\"\n.",
               q{""});
    testPickle("S\"\\n\"\n.",
               q{"\n"});
    testPickle("S'\\n'\n.",
               q{"\n"});
}

// AbstractUnpickleTests.test_frame_readline
unittest
{
    testPickle("\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00I42\n.",
               q{42});
}

// AbstractUnpickleTests.test_compat_unpickle
unittest
{
    testPickle("\x80\x02c__builtin__\nxrange\nK\x01K\x07K\x01\x87R.",
               q{reduce(global(b"builtins.range"), (1, 7, 1))});
    testPickle("\x80\x02c__builtin__\nreduce\n.",
               q{global(b"functools.reduce")});
    testPickle("\x80\x02cwhichdb\nwhichdb\n.",
               q{global(b"dbm.whichdb")});
    testPickle("\x80\x02cexceptions\nException\nU\x03ugh\x85R.",
               q{reduce(global(b"builtins.Exception"), ("ugh",))});
    testPickle("\x80\x02cexceptions\nStandardError\nU\x03ugh\x85R.",
               q{reduce(global(b"builtins.Exception"), ("ugh",))});
    testPickle("\x80\x02(cUserDict\nUserDict\no}U\x04data}K\x01K\x02ssb.",
               q{obj(global(b"collections.UserDict"), buildState={"data": {1: 2}})});
    testPickle("\x80\x02(cUserDict\nIterableUserDict\no}U\x04data}K\x01K\x02ssb.",
               q{obj(global(b"collections.UserDict"), buildState={"data": {1: 2}})});
}

// AbstractUnpickleTests.test_bad_reduce
unittest
{
    testPickle("cbuiltins\nint\n)R.",
               q{reduce(global(b"builtins.int"), (,))});
    //testPickleBad("N)R.");
    //testPickleBad("cbuiltins\nint\nNR.");
}

// AbstractUnpickleTests.test_bad_newobj
unittest
{
    testPickle("cbuiltins\nint\n)\x81.",
               q{obj(global(b"builtins.int"), (,))});
    //testPickleBad("cbuiltins\nlen\n)\x81.");
    //testPickleBad("cbuiltins\nint\nN\x81.");
}

// AbstractUnpickleTests.test_bad_newobj_ex
unittest
{
    testPickle("cbuiltins\nint\n)}\x92.",
               q{obj(global(b"builtins.int"), (,), {})});
    //testPickleBad("cbuiltins\nlen\n)}\x92.");
    //testPickleBad("cbuiltins\nint\nN}\x92.");
    //testPickleBad("cbuiltins\nint\n)N\x92.");
}

// AbstractUnpickleTests.test_bad_stack
unittest
{
    testPickleBad(".");
    testPickleBad("0");
    testPickleBad("1");
    testPickleBad("2");
    testPickleBad("(2");
    testPickleBad("R");
    testPickleBad(")R");
    testPickleBad("a");
    testPickleBad("Na");
    testPickleBad("b");
    testPickleBad("Nb");
    testPickleBad("d");
    testPickleBad("e");
    testPickleBad("(e");
    testPickleBad("ibuiltins\nlist\n");
    testPickleBad("l");
    testPickleBad("o");
    testPickleBad("(o");
    testPickleBad("p1\n");
    testPickleBad("q\x00");
    testPickleBad("r\x00\x00\x00\x00");
    testPickleBad("s");
    testPickleBad("Ns");
    testPickleBad("NNs");
    testPickleBad("t");
    testPickleBad("u");
    testPickleBad("(u");
    testPickleBad("}(Nu");
    testPickleBad("\x81");
    testPickleBad(")\x81");
    testPickleBad("\x85");
    testPickleBad("\x86");
    testPickleBad("N\x86");
    testPickleBad("\x87");
    testPickleBad("N\x87");
    testPickleBad("NN\x87");
    testPickleBad("\x90");
    testPickleBad("(\x90");
    testPickleBad("\x91");
    testPickleBad("\x92");
    testPickleBad(")}\x92");
    testPickleBad("\x93");
    testPickleBad("Vlist\n\x93");
    testPickleBad("\x94");
}

// AbstractUnpickleTests.test_bad_mark
unittest
{
    testPickleBad("N(.");
    testPickleBad("N(2");
    testPickleBad("cbuiltins\nlist\n)(R");
    testPickleBad("cbuiltins\nlist\n()R");
    testPickleBad("]N(a");
    testPickleBad("cbuiltins\nValueError\n)R}(b");
    testPickleBad("cbuiltins\nValueError\n)R(}b");
    testPickleBad("(Nd");
    testPickleBad("N(p1\n");
    testPickleBad("N(q\x00");
    testPickleBad("N(r\x00\x00\x00\x00");
    testPickleBad("}NN(s");
    testPickleBad("}N(Ns");
    testPickleBad("}(NNs");
    testPickleBad("}((u");
    testPickleBad("cbuiltins\nlist\n)(\x81");
    testPickleBad("cbuiltins\nlist\n()\x81");
    testPickleBad("N(\x85");
    testPickleBad("NN(\x86");
    testPickleBad("N(N\x86");
    testPickleBad("NNN(\x87");
    testPickleBad("NN(N\x87");
    testPickleBad("N(NN\x87");
    testPickleBad("]((\x90");
    testPickleBad("cbuiltins\nlist\n)}(\x92");
    testPickleBad("cbuiltins\nlist\n)(}\x92");
    testPickleBad("cbuiltins\nlist\n()}\x92");
    testPickleBad("Vbuiltins\n(Vlist\n\x93");
    testPickleBad("Vbuiltins\nVlist\n(\x93");
    testPickleBad("N(\x94");
}

// AbstractUnpickleTests.test_truncated_data
unittest
{
    testPickleBad("");
    testPickleBad("N");
    testPickleBad("B");
    testPickleBad("B\x03\x00\x00");
    testPickleBad("B\x03\x00\x00\x00");
    testPickleBad("B\x03\x00\x00\x00ab");
    testPickleBad("C");
    testPickleBad("C\x03");
    testPickleBad("C\x03ab");
    testPickleBad("F");
    testPickleBad("F0.0");
    testPickleBad("F0.00");
    testPickleBad("G");
    testPickleBad("G\x00\x00\x00\x00\x00\x00\x00");
    testPickleBad("I");
    testPickleBad("I0");
    testPickleBad("J");
    testPickleBad("J\x00\x00\x00");
    testPickleBad("K");
    testPickleBad("L");
    testPickleBad("L0");
    testPickleBad("L10");
    testPickleBad("L0L");
    testPickleBad("L10L");
    testPickleBad("M");
    testPickleBad("M\x00");
    testPickleBad("S");
    testPickleBad("S'abc'");
    testPickleBad("T");
    testPickleBad("T\x03\x00\x00");
    testPickleBad("T\x03\x00\x00\x00");
    testPickleBad("T\x03\x00\x00\x00ab");
    testPickleBad("U");
    testPickleBad("U\x03");
    testPickleBad("U\x03ab");
    testPickleBad("V");
    testPickleBad("Vabc");
    testPickleBad("X");
    testPickleBad("X\x03\x00\x00");
    testPickleBad("X\x03\x00\x00\x00");
    testPickleBad("X\x03\x00\x00\x00ab");
    testPickleBad("(c");
    testPickleBad("(cbuiltins");
    testPickleBad("(cbuiltins\n");
    testPickleBad("(cbuiltins\nlist");
    testPickleBad("Ng");
    testPickleBad("Ng0");
    testPickleBad("(i");
    testPickleBad("(ibuiltins");
    testPickleBad("(ibuiltins\n");
    testPickleBad("(ibuiltins\nlist");
    testPickleBad("Nh");
    testPickleBad("Nj");
    testPickleBad("Nj\x00\x00\x00");
    testPickleBad("Np");
    testPickleBad("Np0");
    testPickleBad("Nq");
    testPickleBad("Nr");
    testPickleBad("Nr\x00\x00\x00");
    testPickleBad("\x80");
    testPickleBad("\x82");
    testPickleBad("\x83");
    testPickleBad("\x84\x01");
    testPickleBad("\x84");
    testPickleBad("\x84\x01\x00\x00");
    testPickleBad("\x8a");
    testPickleBad("\x8b");
    testPickleBad("\x8b\x00\x00\x00");
    testPickleBad("\x8c");
    testPickleBad("\x8c\x03");
    testPickleBad("\x8c\x03ab");
    testPickleBad("\x8d");
    testPickleBad("\x8d\x03\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x8d\x03\x00\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x8d\x03\x00\x00\x00\x00\x00\x00\x00ab");
    testPickleBad("\x8e");
    testPickleBad("\x8e\x03\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x8e\x03\x00\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x8e\x03\x00\x00\x00\x00\x00\x00\x00ab");
    testPickleBad("\x96");
    testPickleBad("\x96\x03\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x96\x03\x00\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x96\x03\x00\x00\x00\x00\x00\x00\x00ab");
    testPickleBad("\x95");
    testPickleBad("\x95\x02\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x95\x02\x00\x00\x00\x00\x00\x00\x00");
    testPickleBad("\x95\x02\x00\x00\x00\x00\x00\x00\x00N");
}

unittest
{
    testPickle("\x80\x02\x8a\x00.",
               q{0});
    testPickle("\x80\x02\x8a\x01\x01.",
               q{1});
    testPickle("\x80\x02\x8a\x01\x7f.",
               q{127});
    testPickle("\x80\x02\x8a\x01\x80.",
               q{-128});
    testPickle("\x80\x02\x8a\x01\xff.",
               q{-1});
    testPickle("\x80\x02\x8a\x08\xff\xff\xff\xff\xff\xff\xff\x7f.",
               q{9223372036854775807});
    testPickle("\x80\x02\x8a\x08\xff\xff\xff\xff\xff\xff\xff\xff.",
               q{-1});
    testPickle("\x80\x02\x8a\x08\x00\x00\x00\x00\x00\x00\x00\x80.",
               q{-9223372036854775808});
    testPickle("\x80\x02\x8a\t\xff\xff\xff\xff\xff\xff\xff\xff\xff.",
               q{-1});
    testPickle("\x80\x02\x8a\t\xff\xff\xff\xff\xff\xff\xff\xff\x7f.",
               "2361183241434822606847");
    testPickle("\x80\x02\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\x01.",
               "18446744073709551616");
    testPickle("\x80\x02\x8a\t\x00\x00\x00\x00\x00\x00\x00\x00\x80.",
               "-2361183241434822606848");
    testPickle("\x80\x02\x8b\x00\x00\x00\x00.",
               q{0});
    testPickle("\x80\x02\x8b\x01\x00\x00\x00\x01.",
               q{1});
    testPickle("\x80\x02\x8b\x01\x00\x00\x00\x7f.",
               q{127});
    testPickle("\x80\x02\x8b\x01\x00\x00\x00\x80.",
               q{-128});
    testPickle("\x80\x02\x8b\x01\x00\x00\x00\xff.",
               q{-1});
    testPickle("\x80\x02\x8b\x08\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\x7f.",
               q{9223372036854775807});
    testPickle("\x80\x02\x8b\x08\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff.",
               q{-1});
    testPickle("\x80\x02\x8b\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80.",
               q{-9223372036854775808});
    testPickle("\x80\x02\x8b\t\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff.",
               q{-1});
    testPickle("\x80\x02\x8b\t\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\xff\x7f.",
               "2361183241434822606847");
    testPickle("\x80\x02\x8b\t\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01.",
               "18446744073709551616");
    testPickle("\x80\x02\x8b\t\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80.",
               "-2361183241434822606848");
}
