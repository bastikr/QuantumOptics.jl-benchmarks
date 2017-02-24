using QuantumOptics
using BenchmarkTools

suite = BenchmarkGroup()
suite["dense-ket"] = BenchmarkGroup()
suite["bra-dense"] = BenchmarkGroup()
suite["sparse-ket"] = BenchmarkGroup()
suite["bra-sparse"] = BenchmarkGroup()

op_dense = N->DenseOperator(GenericBasis(N), ones(Complex128, N, N))
op_sparse = N->sparse(op_dense(N))
alpha = complex(1.)
beta = complex(0.)

for N in [5, 50, 500]
    b = GenericBasis(N)
    A = DenseOperator(b, ones(Complex128, N, N))
    Asparse = sparse(A)
    x = Ket(b, ones(Complex128, N))
    y = Ket(b, ones(Complex128, N))
    x_ = dagger(x)
    y_ = dagger(y)
    suite["dense-ket"][N] = @benchmarkable operators.gemv!($alpha, $A, $x, $beta, $y)
    suite["bra-dense"][N] = @benchmarkable operators.gemv!($alpha, $x_, $A, $beta, $y_)
    suite["sparse-ket"][N] = @benchmarkable operators.gemv!($alpha, $Asparse, $x, $beta, $y)
    suite["bra-sparse"][N] = @benchmarkable operators.gemv!($alpha, $x_, $Asparse, $beta, $y_)
end
