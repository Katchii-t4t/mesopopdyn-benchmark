# benchmark_julia.jl -- Julia backend for the mesoscopic PD14 benchmark.
#
# Runs the meso simulation (sim5.jl, from the upstream repo) at several network
# sizes, with a warmup run (excluded) and repeats, and writes JSON in the same
# schema as benchmark.py / benchmark_nest.py so compare_backends.py can merge it.
#
# Faithful to upstream PDmodel_mesojulia.jl: weights are converted to per-spike
# (J[i,j] = weights[i,j] / N_scaled[j]); since the .mat stores full-size coupling
# (Js * N_full), this preserves total coupling under down-scaling (J ~ 1/scaling),
# i.e. the same convention as the corrected model_setup.py.
#
# Population order is E-first [L23e,L4e,L5e,L6e,L23i,L4i,L5i,L6i] (save_params_for_julia.py).
#
# Run (inside WSL, nest env):
#   julia benchmark_julia.jl
#
# Karthik Senthil Ganesh, TU Berlin / BCCN internship, 2026.

using MAT, Printf

const UPSTREAM = joinpath(@__DIR__, "..", "upstream_reference", "benchmarking")
include(joinpath(UPSTREAM, "sim5.jl"))

# published PD14 rates, E-first order
const TARGET = [0.971, 4.746, 8.142, 0.991, 2.868, 5.396, 9.078, 7.523]
const POPS   = ["L23e","L4e","L5e","L6e","L23i","L4i","L5i","L6i"]

function build_params(matfile, scaling)
    f = matopen(matfile); p = read(f); close(f)
    M = 8
    p["N"] = round.(Int, vec(p["N"]) * scaling)
    W = p["weights"]
    J = zeros(M, M)
    for i in 1:M, j in 1:M
        J[i, j] = W[i, j] / p["N"][j]
    end
    p["weights"] = J
    return p, M
end

function run_one(matfile, scaling, T, dt, dt_rec, repeats; seed=30)
    p, M = build_params(matfile, scaling)
    Nbin = round(Int, T / dt_rec)
    p["Iext"] = zeros(Float32, Nbin, M)   # B = 0, no stimulus
    Ntot = sum(p["N"])

    # warmup (JIT compile) -- excluded
    pw, _ = build_params(matfile, scaling)
    nbw = round(Int, 0.05 / dt_rec)
    pw["Iext"] = zeros(Float32, nbw, M)
    sim(0.05, dt, dt_rec, pw, M, seed)

    times = Float64[]
    local rate
    for _ in 1:repeats
        pr, _ = build_params(matfile, scaling)
        pr["Iext"] = zeros(Float32, Nbin, M)
        t0 = time()
        rate, _ = sim(T, dt, dt_rec, pr, M, seed)
        push!(times, time() - t0)
    end

    # mean rate per population over steady state (t > 0.5 s)
    n0 = round(Int, 0.5 / dt_rec)
    meanrate = [sum(rate[i, n0:end]) / length(rate[i, n0:end]) for i in 1:M]
    abserr = sum(abs.(meanrate .- TARGET)) / M
    tmin = minimum(times)
    return Ntot, tmin, sum(times)/length(times), meanrate, abserr
end

function main()
    matfile = joinpath(UPSTREAM, "params_PDmodel.mat")
    scalings = [0.1, 0.5, 1.0]
    # T kept short + 1 repeat: the upstream pure-Julia sim5.jl is unoptimised
    # (heavy allocations / type instability), so it is far slower than the
    # Numba/C/NEST backends. Cost-per-biosecond still compares fairly.
    T = 2.0; dt = 1e-4; dt_rec = 1e-3; repeats = 1

    results = String[]
    println("=== mesopopdyn Julia benchmark ===")
    for s in scalings
        Ntot, tmin, tmean, meanrate, abserr = run_one(matfile, s, T, dt, dt_rec, repeats)
        @printf("[run ] meso_julia scaling=%-4g N=%-6d min=%.3fs (rt x%.2f) err=%.3fHz\n",
                s, Ntot, tmin, T/tmin, abserr)
        rates_str = join([@sprintf("%.4f", r) for r in meanrate], ", ")
        push!(results, @sprintf(
          "{\"backend\":\"meso_julia\",\"scaling\":%g,\"N_total\":%d,\"T\":%g,\"repeats\":%d,\"status\":\"ok\",\"time_min_s\":%.6f,\"time_mean_s\":%.6f,\"realtime_factor\":%.4f,\"mean_abs_rate_err_hz\":%.6f,\"mean_rate_hz\":[%s]}",
          s, Ntot, T, repeats, tmin, tmean, T/tmin, abserr, rates_str))
    end

    open(joinpath(@__DIR__, "..", "results", "benchmark_julia_latest.json"), "w") do io
        write(io, "{\"meta\":{\"backend\":\"meso_julia\",\"pop_labels\":[\"" *
                  join(POPS, "\",\"") * "\"]},\"results\":[" *
                  join(results, ",") * "]}")
    end
    println("\nsaved: benchmark_julia_latest.json")
end

main()
