using QuantumOptics
using BenchmarkTools

suite = BenchmarkGroup()
suite["dense"] = BenchmarkGroup()
suite["sparse"] = BenchmarkGroup()

op_dense = N->DenseOperator(GenericBasis(N), ones(Complex128, N, N))
op_sparse = N->sparse(op_dense(N))

for N in [2, 10, 50]
    suite["dense"][N] = @benchmarkable tensor($(op_dense(N)), $(op_dense(N)))
    suite["sparse"][N] = @benchmarkable tensor($(op_sparse(N)), $(op_sparse(N)))
end
