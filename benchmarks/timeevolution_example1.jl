using QuantumOptics
using BenchmarkTools

suite = BenchmarkGroup(["timeevolution"],
    "master" => BenchmarkGroup(["master"],
            "dense" => BenchmarkGroup(),
            "sparse" => BenchmarkGroup()
        ),
    "schroedinger" => BenchmarkGroup(["schroedinger"],
            "dense" => BenchmarkGroup(),
            "sparse" => BenchmarkGroup()
        ),
    "mcwf" => BenchmarkGroup(["mcwf"],
            "dense" => BenchmarkGroup(),
            "sparse" => BenchmarkGroup()
        )
    )


ωc = 1.2
ωa = 0.9
g = 1.0
η = 1.1
γ = 0.5
κ = 1.1

T = Float64[0.,1.]

spinbasis = SpinBasis(1//2)

sx = sigmax(spinbasis)
sy = sigmay(spinbasis)
sz = sigmaz(spinbasis)
sp = sigmap(spinbasis)
sm = sigmam(spinbasis)

function generate_system(Ncutoff)
    fockbasis = FockBasis(Ncutoff)
    a = destroy(fockbasis)
    at = create(fockbasis)

    basis = tensor(spinbasis, fockbasis)
    Ha = embed(basis, 1, 0.5*ωa*sz)
    Hc = embed(basis, 2, ωc*number(fockbasis) + η*(a + at))
    Hint = sm ⊗ at + sp ⊗ a
    H = Ha + Hc + Hint

    J = [embed(basis, 1, sm), embed(basis, 2, destroy(fockbasis))]
    Γ = [γ, κ]

    Hdense = full(H)
    Jdense = [full(j) for j in J]

    Ψ₀ = spinup(spinbasis) ⊗ fockstate(fockbasis, 2)
    ρ₀ = Ψ₀ ⊗ dagger(Ψ₀)
    Ψ₀, ρ₀, H, Hdense, J, Jdense, Γ
end

for Ncutoff in [3, 10, 100]
    Ψ₀, ρ₀, H, Hdense, J, Jdense, Γ = generate_system(Ncutoff)
    suite["master"]["dense"][Ncutoff] = @benchmarkable timeevolution.master($T, $ρ₀, $Hdense, $Jdense; Gamma=$Γ, reltol=1e-6)
    suite["master"]["sparse"][Ncutoff] = @benchmarkable timeevolution.master($T, $ρ₀, $H, $J; Gamma=$Γ, reltol=1e-6)
    suite["schroedinger"]["dense"][Ncutoff] = @benchmarkable timeevolution.schroedinger($T, $Ψ₀, $Hdense; reltol=1e-6)
    suite["schroedinger"]["sparse"][Ncutoff] = @benchmarkable timeevolution.schroedinger($T, $Ψ₀, $H; reltol=1e-6)
    suite["mcwf"]["dense"][Ncutoff] = @benchmarkable timeevolution.mcwf($T, $Ψ₀, $Hdense, $Jdense; reltol=1e-6)
    suite["mcwf"]["sparse"][Ncutoff] = @benchmarkable timeevolution.mcwf($T, $Ψ₀, $H, $J; reltol=1e-6)
end
