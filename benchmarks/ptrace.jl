using QuantumOptics
using BenchmarkTools

suite = BenchmarkGroup()
suite["dense"] = BenchmarkGroup()
suite["sparse"] = BenchmarkGroup()

op_dense = N->DenseOperator(GenericBasis(N), ones(Complex128, N, N))
op_denseN = (N, rank) -> tensor([op_dense(N) for i=1:rank]...)
op_sparse = N -> sparse(op_dense(N))
op_sparseN = (N, rank) -> sparse(op_denseN(N, rank))

indices = [1, 3]
for rank in [3, 4, 5]
    suite["dense"][rank] = @benchmarkable ptrace($(op_denseN(3, rank)), $indices)
    suite["sparse"][rank] = @benchmarkable ptrace($(op_sparseN(3, rank)), $indices)
end
