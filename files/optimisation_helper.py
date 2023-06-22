import math

## 8 bit
# Map the input range of [-1, 1] in float32 to [0, 255] in int8
def float32_to_uint8(f):
    i = math.floor((f + 1.0) * 127.5) # use floor to get integer
    # clip to [0, 255]
    i = max(0, min(i, 255))
    return i
    
# Map the input range of [0, 255] in int8 to [-1, 1] in float32
def uint8_to_float32(i):
    f = (i / 127) - 1.0
    # clip to [-1, 1]
    f = max(-1.0, min(f, 1.0))
    return f

def quant_dequant_8bit(f):
    return uint8_to_float32(float32_to_uint8(f))


#######################################################
# Tests
def test_float32_to_uint8_upper_bound():
    assert float32_to_uint8(1.0) == 255

def test_float32_to_uint8_lower_bound():
    assert float32_to_uint8(-1.0) == 0
    
def test_float32_to_uint8_midpoint():
    assert float32_to_uint8(0.0) == 127

def test_uint8_to_float32_upper_bound():
    assert uint8_to_float32(255) == 1.0

def test_uint8_to_float32_lower_bound():
    assert uint8_to_float32(0) == -1.0

def test_uint8_to_float32_midpoint():
    assert uint8_to_float32(127) == 0.0

#######################################################

## 7 bit
# Map the input range of [-1, 1] in float32 to [0, 127] in int8
def float32_to_bit7(f):
    i = math.floor((f + 1.0) * 63.5)
    # clip to [0, 127]
    i = max(0, min(i, 127))
    return i

# Map the input range of [0, 127] in int8 to [-1, 1] in float32
def bit7_to_float32(i):
    f = (i / 63) - 1.0
    # clip to [-1, 1]
    f = max(-1.0, min(f, 1.0))
    return f

def quant_dequant_7bit(f):
    return bit7_to_float32(float32_to_bit7(f))

## 6 bit
def float32_to_bit6(f):
    i = math.floor((f + 1.0) * 31.5)
    # clip to [0, 63]
    i = max(0, min(i, 63))
    return i

def bit6_to_float32(i):
    f = (i / 31) - 1.0
    # clip to [-1, 1]
    f = max(-1.0, min(f, 1.0))
    return f

def quant_dequant_6bit(f):
    return bit6_to_float32(float32_to_bit6(f))

## 5 bit
def float32_to_bit5(f):
    i = math.floor((f + 1.0) * 15.5)
    # clip to [0, 31]
    i = max(0, min(i, 31))
    return i

def bit5_to_float32(i):
    f = (i / 15) - 1.0
    # clip to [-1, 1]
    f = max(-1.0, min(f, 1.0))
    return f

def quant_dequant_5bit(f):
    return bit5_to_float32(float32_to_bit5(f))

## 4 bit
def float32_to_bit4(f):
    i = math.floor((f + 1.0) * 7.5)
    # clip to [0, 15]
    i = max(0, min(i, 15))
    return i

def bit4_to_float32(i):
    f = (i / 7) - 1.0
    # clip to [-1, 1]
    f = max(-1.0, min(f, 1.0))
    return f

def quant_dequant_4bit(f):
    return bit4_to_float32(float32_to_bit4(f))

## 3 bit
def float32_to_bit3(f):
    i = math.floor((f + 1.0) * 3.5)
    # clip to [0, 7]
    i = max(0, min(i, 7))
    return i

def bit3_to_float32(i):
    f = (i / 3) - 1.0
    # clip to [-1, 1]
    f = max(-1.0, min(f, 1.0))
    return f

def quant_dequant_3bit(f):
    return bit3_to_float32(float32_to_bit3(f))
########################################
# Assumes float32 follows IEEE 754 standard
def float32_to_fixed12(f):
    ## Use 1 bit for sign, 1 bit for integer part, 10 bits for decimal part
    # Note this assumes the float number is in the range of [-1, 1]

    # Extracting sign bit
    sign_bit = 1 if f < 0 else 0
    # Converting the float number to absolute value
    abs_float_num = abs(f)
    # Extracting integer and decimal parts
    integer_part = int(abs_float_num)
    decimal_part = abs_float_num - integer_part

    # Converting the integer part to binary
    integer_binary = bin(integer_part)[2:].zfill(1)

    # Converting the decimal part to binary
    decimal_binary = ""
    for _ in range(10):
        decimal_part *= 2
        bit = int(decimal_part)
        decimal_binary += str(bit)
        decimal_part -= bit

    # Combining the sign, integer, and decimal parts
    fixed_number = sign_bit << 11 | int(integer_binary, 2) << 10 | int(decimal_binary, 2)

    return fixed_number

def fixed12_to_float32(fixed): 
    # Ensure fixed is an integer
    fixed = int(fixed)
    # Extracting sign bit
    # 0x800 = 1000 0000 0000
    sign_bit = (fixed & 0x800) >> 11
    # Extracting integer part
    # 0x400 = 01000 0000 0000
    integer_part = (fixed & 0x400) >> 10
    # Extracting decimal part
    # 0x3FF = 0011 1111 1111
    decimal_part = fixed & 0x3FF

    # Converting decimal part to float
    decimal_float = 0.0
    factor = 0.5
    for i in range(10):
        decimal_float += ((decimal_part >> (9 - i)) & 1) * factor
        factor /= 2

    # Combining sign, integer, and decimal parts to form the float number
    float_num = (-1) ** sign_bit * (integer_part + decimal_float)

    return float_num

def quant_dequant_fixed12(f):
    fixed_ = float32_to_fixed12(f)
    fixed_ = int(fixed_)
    float_ = fixed12_to_float32(fixed_)
    float_ = float(float_)
    return float_


def float32_to_fixed24(f):
    # Use 1 bit for sign, 1 bit for integer part, 22 bits for decimal part
    # Note this assumes the float number is in the range of [-1, 1]
    
    # Extracting sign bit
    sign_bit = 1 if f < 0 else 0
    # Converting the float number to absolute value
    abs_float_num = abs(f)
    # Extracting integer and decimal parts
    integer_part = int(abs_float_num)
    decimal_part = abs_float_num - integer_part
    # Converting the integer part to binary
    integer_binary = bin(integer_part)[2:].zfill(1)

    # Converting the decimal part to binary
    decimal_binary = ""
    for _ in range(22):
        decimal_part *= 2
        bit = int(decimal_part)
        decimal_binary += str(bit)
        decimal_part -= bit

    # Combining the sign, integer, and decimal parts
    fixed_number = (sign_bit << 23) | (int(integer_binary, 2) << 22) | int(decimal_binary, 2)

    return fixed_number


def fixed24_to_float32(fixed):
    # Ensure fixed is an integer
    fixed = int(fixed)
    # Extracting sign bit
    sign_bit = (fixed & 0x800000) >> 23
    # Extracting integer part
    integer_part = (fixed & 0x400000) >> 22
    # Extracting decimal part
    decimal_part = fixed & 0x3FFFFF

    # Converting decimal part to float
    decimal_float = 0.0
    factor = 0.5
    for i in range(22):
        decimal_float += ((decimal_part >> (21 - i)) & 1) * factor
        factor /= 2

    # Combining sign, integer, and decimal parts to form the float number
    float_num = (-1) ** sign_bit * (integer_part + decimal_float)

    return float_num

def quant_dequant_fixed24(f):
    fixed_ = float32_to_fixed24(f)
    fixed_ = int(fixed_)
    float_ = fixed24_to_float32(fixed_)
    float_ = float(float_)
    return float_


########################################
def quant_dequant_binarise(f):
    return 1.0 if f > 0 else -1.0

def quant_dequant_ternise(f):
    return 1.0 if f > 0.33 else -1.0 if f < -0.33 else 0.0

import torch
def quant_dequant_float16(f):
    f = torch.tensor(f, dtype=torch.float16)
    return f.item()

########################################

if __name__ == "__main__":
    # test_float32_to_uint8_lower_bound()
    # test_float32_to_uint8_upper_bound()
    # test_float32_to_uint8_midpoint()
    # test_uint8_to_float32_lower_bound()
    # test_uint8_to_float32_upper_bound()
    # test_uint8_to_float32_midpoint()
    # print(quant_dequant_8bit(0.5))
    
    # Problem, for small numbers, the quantised number is always 0
    # b = 0.0001
    # fixed_output = float32_to_fixed12(b)
    # print(fixed_output)
    # a = fixed12_to_float32(fixed_output)
    # print(a)
    a = 0.000001
    fixed = float32_to_fixed24(a)
    print(fixed)
    b = fixed24_to_float32(fixed)
    print(b)
    