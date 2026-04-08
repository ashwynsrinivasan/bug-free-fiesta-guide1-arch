import math

def decode_linear11(high_byte: int, low_byte: int) -> float:
    """
    Decodes a 2-byte value using the 11-Bit Linear Data Format.

    This format uses a 16-bit word composed of a signed 5-bit exponent (N)
    and a signed 11-bit mantissa (Y). The final value is calculated as Y * 2^N.

    Args:
        high_byte: The most significant byte of the 2-byte word.
        low_byte: The least significant byte of the 2-byte word.

    Returns:
        The decoded floating-point value.
    """
    # Combine the two bytes into a single 16-bit integer word
    data_word = (high_byte << 8) | low_byte

    # --- Step 1: Extract and decode the 5-bit exponent (N) ---
    # N is stored in the 5 most significant bits (bits 15-11)
    raw_n = data_word >> 11

    # Convert N from 5-bit two's complement format. The sign bit is the 5th bit (0b10000 = 16).
    # If the sign bit is set, the value is negative.
    if raw_n >= 16:
        signed_n = raw_n - 32
    else:
        signed_n = raw_n

    # --- Step 2: Extract and decode the 11-bit mantissa (Y) ---
    # Y is stored in the 11 least significant bits (bits 10-0).
    # We use a bitmask (0x7FF) to isolate these bits.
    raw_y = data_word & 0x7FF

    # Convert Y from 11-bit two's complement format. The sign bit is the 11th bit (0b10000000000 = 1024).
    if raw_y >= 1024:
        signed_y = raw_y - 2048
    else:
        signed_y = raw_y

    # --- Step 3: Calculate the final value ---
    # Apply the formula: Value = Y * 2^N
    value = signed_y * (2**signed_n)

    return value

# --- Example Usage ---

# Example 1: A positive value
# Let's say the device returns 0x18A0
# High Byte = 0x18 (00011000), Low Byte = 0xA0 (10100000)
high1 = 0x18
low1 = 0xA0
decoded_value1 = decode_linear11(high1, low1)
print(f"Input: {hex(high1)}, {hex(low1)} -> Decoded Value: {decoded_value1}")
# Expected calculation:
# N = 0b00011 = 3
# Y = 0b00010100000 = 160
# Value = 160 * (2^3) = 160 * 8 = 1280.0

# Example 2: A negative value with a negative exponent
# Let's say the device returns 0xFB50
# High Byte = 0xFB (11111011), Low Byte = 0x50 (01010000)
high2 = 0xFB
low2 = 0x50
decoded_value2 = decode_linear11(high2, low2)
print(f"Input: {hex(high2)}, {hex(low2)} -> Decoded Value: {decoded_value2}")
# Expected calculation:
# N = 0b11111 -> (31 - 32) = -1
# Y = 0b01101010000 = 848
# Value = 848 * (2^-1) = 848 * 0.5 = 424.0

high3 = [6146, 6146, 39427, 32514, 19715]	[33249, 23265, 38624, 9696, 480]

