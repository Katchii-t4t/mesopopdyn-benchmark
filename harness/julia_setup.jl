# julia_setup.jl -- install the packages the Julia meso benchmark needs.
# Run once:  julia julia_setup.jl
using Pkg
Pkg.add(["MAT", "Distributions"])
using MAT, Distributions   # force precompile now
println("JULIA_PKGS_OK")
