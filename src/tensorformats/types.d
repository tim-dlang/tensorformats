
//          Copyright Tim Schendekehl 2024.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          https://www.boost.org/LICENSE_1_0.txt)

module tensorformats.types;
import std.complex;
import std.conv;
import std.format;
import std.meta;
import std.numeric;

alias Half = CustomFloat!16;
static assert(Half.sizeof == 2);

alias BFloat16 = CustomFloat!(7, 8);
static assert(Half.sizeof == 2);

// See https://arxiv.org/abs/2209.05433 "FP8 Formats for Deep Learning"
alias F8E5M2 = CustomFloat!(2, 5, CustomFloatFlags.signed | CustomFloatFlags.storeNormalized | CustomFloatFlags.allowDenorm | CustomFloatFlags.infinity | CustomFloatFlags.nan, 15);
static assert(F8E5M2.sizeof == 1);
// TODO: F8E4M3 has a NAN value, but it is stored differently.
alias F8E4M3 = CustomFloat!(3, 4, CustomFloatFlags.signed | CustomFloatFlags.storeNormalized | CustomFloatFlags.allowDenorm/* | CustomFloatFlags.nan*/, 7);
static assert(F8E4M3.sizeof == 1);

struct ComplexHalf
{
    Half re, im;
    string toString() const @safe
    {
        // Make local copy, because CustomFloat.get is not usable with const.
        // See https://issues.dlang.org/show_bug.cgi?id=24851
        Half re = this.re;
        Half im = this.im;
        return format("%g%+gi", re.get!float, im.get!float);
    }
}

private struct TypeUDA(T)
{
    alias Type = T;
    size_t expectedSize;
}

/**
Type of values in tensors
*/
enum ValueType
{
    unknown,
    @TypeUDA!float(4) float_,
    @TypeUDA!double(8) double_,
    @TypeUDA!Half(2) half,
    @TypeUDA!BFloat16(2) bfloat16,
    @TypeUDA!ubyte(1) uint8,
    @TypeUDA!ushort(2) uint16,
    @TypeUDA!uint(4) uint32,
    @TypeUDA!ulong(8) uint64,
    @TypeUDA!byte(1) int8,
    @TypeUDA!short(2) int16,
    @TypeUDA!int(4) int32,
    @TypeUDA!long(8) int64,
    @TypeUDA!F8E5M2(1) f8_e5m2,
    @TypeUDA!F8E4M3(1) f8_e4m3,
    @TypeUDA!bool(1) bool_,
    @TypeUDA!(Complex!float)(8) cfloat_,
    @TypeUDA!(Complex!double)(16) cdouble_,
    @TypeUDA!ComplexHalf(4) chalf,
}
private template TypeUDAFor(string name)
{
    enum TypeUDAFor = __traits(getAttributes, __traits(getMember, ValueType, name))[0];
}
private template TypeUDAFor(ValueType valueType)
{
    static assert(valueType.stringof[0 .. 10] == "ValueType.");
    enum TypeUDAFor = __traits(getAttributes, __traits(getMember, ValueType, valueType.stringof[10 .. $]))[0];
}

/**
Get value type enum for compile time type
*/
template valueTypeFor(T)
{
    static if(is(T == void))
        enum valueTypeFor = ValueType.unknown;
    static foreach (name; __traits(allMembers, ValueType))
    {
        static if (__traits(getMember, ValueType, name) != ValueType.unknown)
        {
            static if (is(TypeUDAFor!name.Type == T))
                enum valueTypeFor = __traits(getMember, ValueType, name);
        }
    }
}

/**
Get compile time type for value type enum
*/
alias TypeForValueType(ValueType valueType) = TypeUDAFor!valueType.Type;

/**
Get element size for value type enum
*/
size_t valueTypeSizeof(ValueType valueType) @safe
{
    switch (valueType)
    {
        static foreach (name; __traits(allMembers, ValueType))
        {
            static if (__traits(getMember, ValueType, name) != ValueType.unknown)
            {
                static assert(TypeUDAFor!name.Type.sizeof
                    == TypeUDAFor!name.expectedSize);
                case __traits(getMember, ValueType, name):
                    return TypeUDAFor!name.Type.sizeof;
            }
        }
    default: return 0;
    }
}

/**
Convert value with runtime type to string

Params:
    valueType = type of value
    data = buffer containing value of type `valueType`
*/
string valueToString(ValueType valueType, const ubyte[] data)
{
    switch (valueType)
    {
        static foreach (name; __traits(allMembers, ValueType))
        {
            static if (__traits(getMember, ValueType, name) != ValueType.unknown)
            {
                case __traits(getMember, ValueType, name):
                {
                    assert(data.length == TypeUDAFor!name.Type.sizeof);
                    TypeUDAFor!name.Type value;
                    (cast(ubyte*) &value)[0 .. TypeUDAFor!name.Type.sizeof] = data[];
                    return text(value);
                }
            }
        }
    default: return "unknown";
    }
}

/**
Convert value with runtime type to compile time type

Params:
    valueType = type of value
    data = buffer containing value of type `valueType`
*/
T valueToType(T)(ValueType valueType, const ubyte[] data)
{
    switch (valueType)
    {
        static foreach (name; __traits(allMembers, ValueType))
        {
            static if (__traits(getMember, ValueType, name) != ValueType.unknown)
            {
                case __traits(getMember, ValueType, name):
                {
                    assert(data.length == TypeUDAFor!name.Type.sizeof);
                    TypeUDAFor!name.Type value;
                    (cast(ubyte*) &value)[0 .. TypeUDAFor!name.Type.sizeof] = data[];
                    static if (is(TypeUDAFor!name.Type : T))
                    {
                        return value;
                    }
                    else static if (__traits(compiles, TypeUDAFor!name.Type.get!T))
                    {
                        return value.get!T;
                    }
                    else static if (is(TypeUDAFor!name.Type == ComplexHalf) && is(T == Complex!double))
                    {
                        return Complex!double(value.re.get!double, value.im.get!double);
                    }
                    else static if (is(TypeUDAFor!name.Type == ComplexHalf) && is(T == Complex!float))
                    {
                        return Complex!float(value.re.get!float, value.im.get!float);
                    }
                    else static if (is(TypeUDAFor!name.Type == Complex!float) && is(T == Complex!double))
                    {
                        return Complex!double(value.re, value.im);
                    }
                    else
                        assert(false, name ~ " " ~ TypeUDAFor!name.Type.stringof);
                }
            }
        }
    default: assert(false, text(valueType));
    }
}
