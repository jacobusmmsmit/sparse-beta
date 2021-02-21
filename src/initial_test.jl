using LightGraphs
using LinearAlgebra
using StatsBase
using GraphPlot
using Optim
using ReverseDiff
using ForwardDiff

include("src/estimating_functions.jl")
include("src/generators.jl")

n = 50
beta = zeros(n)
for i in sample(collect(1:n), Int(floor(n / 10)))
    beta[i] = sqrt(log(n))
end
mu = -log(n)

g = sparse_beta_graph(beta, mu)

layout = (args...) -> spring_layout(args...; C=9)
gplot(g, layout=layout)

# Parameter Estimation

function sparse_beta_loglik(g::Graph, mu, beta)
    beta = exp.(log.(beta))
    tot = 0
    for i in 1:n
        tot -= degree(g)[i] * beta[i]
        for j in i:n
            tot += log(1 + exp(mu + beta[i] + beta[j]))
        end
    end
    return -(g.ne * mu) + tot
end

function sparse_beta_loglik_rp(g::Graph, mu, gamma)
    beta = exp.(gamma)
    tot = 0
    for i in 1:n
        tot -= degree(g)[i] * beta[i]
        for j in i:n
            tot += log(1 + exp(mu + beta[i] + beta[j]))
        end
    end
    return -(g.ne * mu) + tot
end

f = mubeta -> sparse_beta_loglik_rp(g, mubeta[1], mubeta[2:end])

tape = ReverseDiff.GradientTape(f, vcat(-1.5, repeat([1], n)))
(storage, params) -> ReverseDiff.gradient!(storage, tape, params)

optimize(
    mubeta -> sparse_beta_loglik_rp(g, mubeta[1], mubeta[2:end]),
    (storage, params) -> ReverseDiff.gradient!(storage, tape, params),
    # repeat([0], n+1),
    # repeat([Inf], n+1),
    vcat(-1.5, repeat([1], n)),
    LBFGS(),
    Optim.Options(show_every=1))

# # optimize(mubeta -> sparse_beta_loglik(g, mubeta[1], mubeta[2:end]), vcat(-1.5, repeat([0], n)))

# tape = ReverseDiff.GradientTape(mubeta -> sparse_beta_loglik(g, mubeta[1], mubeta[2:end]), vcat(-1.5, repeat([0], n)))
# f(mubeta) = sparse_beta_loglik(g, mubeta[1], mubeta[2:end])
# f_tape = ReverseDiff.GradientTape(f, vcat(-1.5, repeat([0], n)))
# g!(storage, params) = ReverseDiff.gradient!(storage, f_tape, params)

# optimize(
#     f,
#     g!(storage, params),
#     vcat(-1.5, repeat([0], n)),
#     LBFGS(),
#     Optim.Options(
#         show_every=1))
