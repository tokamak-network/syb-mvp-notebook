import random

# Fake etheruem address generator
def generate_random_eth_address():
    return f"0x{random.randint(0, 0xffffffffffffffffffffffffffffffffffffffff):040x}"

def generate_mul_eth_addresses(n):
    return [generate_random_eth_address() for _ in range(n)]
