using QuantumOptics
using BenchmarkTools

suite = BenchmarkGroup([],
    "vector" => BenchmarkGroup([],
        "dense" => BenchmarkGroup(),
        "sparse" => BenchmarkGroup()),
    "dict" => BenchmarkGroup([],
        "dense" => BenchmarkGroup(),
        "sparse" => BenchmarkGroup())
    )

op_dense = N->DenseOperator(GenericBasis(N), ones(Complex128, N, N))
op_denseN = (N, rank) -> tensor([op_dense(N) for i=1:rank]...)
op_sparse = N -> sparse(op_dense(N))
op_sparseN = (N, rank) -> sparse(op_denseN(N, rank))

indices = [2, 4]
for N in [2, 3, 4]
    b = tensor([GenericBasis(N) for i=1:5]...)
    ops = [op_dense(N), op_dense(N)]
    suite["vector"]["dense"][N] = @benchmarkable embed($b, $indices, $ops)
    ops = [op_sparse(N), op_sparse(N)]
    suite["vector"]["sparse"][N] = @benchmarkable embed($b, $indices, $ops)
    ops = Dict(indices=>op_denseN(N, 2))
    suite["dict"]["dense"][N] = @benchmarkable embed($b, $ops)
    ops = Dict(indices=>op_sparseN(N, 2))
    suite["dict"]["sparse"][N] = @benchmarkable embed($b, $ops)
end
