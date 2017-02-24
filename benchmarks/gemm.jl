using QuantumOptics
using BenchmarkTools

suite = BenchmarkGroup()
suite["dense"] = BenchmarkGroup()
suite["sparse-dense"] = BenchmarkGroup()
suite["dense-sparse"] = BenchmarkGroup()

op_dense = N->DenseOperator(GenericBasis(N), ones(Complex128, N, N))
op_sparse = N->sparse(op_dense(N))
alpha = complex(1.)
beta = complex(0.)

for N in [5, 50, 500]
    suite["dense"][N] = @benchmarkable operators.gemm!($alpha, $(op_dense(N)), $(op_dense(N)), $beta, $(op_dense(N)))
    suite["sparse-dense"][N] = @benchmarkable operators.gemm!($alpha, $(op_sparse(N)), $(op_dense(N)), $beta, $(op_dense(N)))
    suite["dense-sparse"][N] = @benchmarkable operators.gemm!($alpha, $(op_dense(N)), $(op_sparse(N)), $beta, $(op_dense(N)))
end
