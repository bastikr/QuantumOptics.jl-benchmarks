using BenchmarkTools
using JLD

const suites = BenchmarkGroup()

names = [
    "tensorproduct.jl",
    "embed.jl",
    "ptrace.jl",
    "gemm.jl",
    "gemv.jl",
    "timeevolution_example1.jl"
]

detected_benchmarks = readdir("benchmarks")
unused_benchmarks = setdiff(detected_benchmarks, names)

if length(unused_benchmarks) != 0
    error("The following benchmarks are not used:\n", join(unused_benchmarks, "\n"))
end

cd("../QuantumOptics.jl")
commitID = readstring(`git rev-parse --verify HEAD`)[1:10]
cd("../QuantumOptics.jl-benchmarks")
println("Benchmarking commit ", commitID)

println("Gathering benchmarks:")
for name=names
    println("    $name")
    include(joinpath("benchmarks", name))
    suites[name] = suite
end


# If a cache of tuned parameters already exists, use it, otherwise, tune and cache
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `suite` every time the file is included.
paramspath = joinpath(dirname(@__FILE__), "params.jld")

if isfile(paramspath)
    println("Tuning parameters found - loading ...")
    loadparams!(suites, BenchmarkTools.load(paramspath, "suites"), :evals, :samples);
else
    println("Tune parameters ...")
    tune!(suites)
    BenchmarkTools.save(paramspath, "suites", params(suites));
end

println("Running benchmarks ...")
result = run(suites, verbose=true)

println("Storing results in ")
outputpath = joinpath(dirname(@__FILE__), "params.jld")
resultpath = joinpath(outputpath, "results/result-$(commitID).jld")
save(resultpath, "result", result)
