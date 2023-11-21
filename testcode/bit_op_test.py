import torch
from time import time

def create_random_matrix(m, n):
    # Create an empty matrix of the desired dimensions
    matrix = torch.empty((m, n), dtype=torch.float)

    # Fill the matrix with random 32-bit integers using torch.randint
    matrix.random_(-1, 1)

    return matrix

def create_random_binary_matrix_8b(m, n):
    # Create an empty matrix of the desired dimensions
    matrix = torch.empty((m, n), dtype=torch.uint8)
    # matrix = torch.zeros((m, n), dtype=torch.quint4x2)

    # Fill the matrix with random 32-bit integers using torch.randint
    matrix.random_(0, 2**8 - 1)

    return matrix


def create_random_binary_matrix_4b(m, n):
    # Create an empty matrix of the desired dimensions
    matrix_low = torch.empty((m, n), dtype=torch.uint8)
    matrix_high = torch.empty((m, n), dtype=torch.uint8)

    matrix = torch.empty((m, n), dtype=torch.uint8)

    # Fill the matrix with random 32-bit integers using torch.randint
    matrix_low.random_(0, 2**4 - 1)
    matrix_high.random_(0, 2**4 - 1)

    matrix = matrix_low + (matrix_high << 4)

    import pdb
    pdb.set_trace()

    return matrix


def create_random_binary_matrix_2b(m, n):
    # Create an empty matrix of the desired dimensions
    matrix_b12 = torch.empty((m, n), dtype=torch.uint8)
    matrix_b34 = torch.empty((m, n), dtype=torch.uint8)
    matrix_b56 = torch.empty((m, n), dtype=torch.uint8)
    matrix_b78 = torch.empty((m, n), dtype=torch.uint8)

    matrix = torch.empty((m, n), dtype=torch.uint8)

    matrix = torch.empty((m, n), dtype=torch.uint8)
    # Fill the matrix with random 32-bit integers using torch.randint
    matrix_b12.random_(0, 2**2 - 1)
    matrix_b34.random_(0, 2**2 - 1)
    matrix_b56.random_(0, 2**2 - 1)
    matrix_b78.random_(0, 2**2 - 1)

    matrix = matrix_b12 + (matrix_b34 << 2) + (matrix_b56 << 4) + (matrix_b78 << 6)

    import pdb
    pdb.set_trace()

    return matrix

t0 = time()
query_set = create_random_matrix(5, 512).cuda()
db_set = create_random_matrix(2000000, 512).cuda()

query_set_binary = create_random_binary_matrix_8b(5, 64).cuda()
db_set_binary = create_random_binary_matrix_8b(2000000, 64).cuda()

# query_set = create_random_matrix(5000, 512)
# db_set = create_random_matrix(5000, 512)

# query_set_binary = create_random_binary_matrix_8b(5000, 64)
# db_set_binary = create_random_binary_matrix_8b(5000, 64)

t1 = time()
print('Time to create random matrices: {}'.format(t1 - t0))
inner_product = query_set@db_set.T
t2 = time()
print('Time to compute rankings: {}'.format(t2 - t1))
# xor_result = query_set_binary[:,None,:].bitwise_xor(db_set_binary[None,:,:])
# xor_result_sum = xor_result.sum(dim=2)
t3 = time()
# print('Time to compute rankings_binary: {}'.format(t3 - t2))

# 2 bits quantization
query_low = query_set_binary & 0b1111
query_high = query_set_binary >> 4
db_low = db_set_binary & 0b1111
db_high = db_set_binary >> 4

xor_result_low = query_low[:,None,:].bitwise_xor(db_low[None,:,:])
xor_result_high = query_high[:,None,:].bitwise_xor(db_high[None,:,:])

xor_result_sum = torch.zeros((5, 2000000), dtype=torch.int32).cuda()
xor_result_sum = xor_result_low.sum(dim=2) + xor_result_high.sum(dim=2) << 1

t4 = time()
print('Time to compute rankings_binary_2b: {}'.format(t4 - t3))

# 4 bits quantization
query_b12 = query_set_binary & 0b11
query_b34 = (query_set_binary >> 2) & 0b11
query_b56 = (query_set_binary >> 4) & 0b11
query_b78 = (query_set_binary >> 6)

db_b12 = db_set_binary & 0b11
db_b34 = (db_set_binary >> 2) & 0b11
db_b56 = (db_set_binary >> 4) & 0b11
db_b78 = (db_set_binary >> 6)

xor_result_b12 = query_b12[:,None,:].bitwise_xor(db_b12[None,:,:])
xor_result_b34 = query_b34[:,None,:].bitwise_xor(db_b34[None,:,:])
xor_result_b56 = query_b56[:,None,:].bitwise_xor(db_b56[None,:,:])
xor_result_b78 = query_b78[:,None,:].bitwise_xor(db_b78[None,:,:])


xor_result_sum = torch.zeros((5, 2000000), dtype=torch.int32).cuda()
xor_result_sum = xor_result_b12.sum(dim=2) + (xor_result_b34.sum(dim=2) << 1) + (xor_result_b56.sum(dim=2) << 2) + (xor_result_b78.sum(dim=2) << 3)

t5 = time()
print('Time to compute rankings_binary_4b: {}'.format(t5 - t4))

import pdb
pdb.set_trace()