using Plots
using LightGraphs
using LinearAlgebra
using StatsBase
using StatsFuns
using GraphPlot

include("estimating_functions.jl")
include("generators.jl")

n = 500
beta = zeros(n)
for i in sample(collect(1:n), Int(floor(n / 10)))
    beta[i] = sqrt(log(n))
end
mu = -log(n)

g = sparse_beta_graph(beta, mu)

i = rand((collect(1:500)[beta .> 0]))
i_not = rand((collect(1:500)[beta .== 0]))
beta_mi = beta[1:end .!= i]
beta_i = beta[i]
deg = degree(g)[i]

beta_mi_not = beta[1:end .!= i_not]
beta_i_not = beta[i_not]
deg2 = degree(g)[i_not]



partial_sparse_beta_conditional_loglik(beta_i) = sparse_beta_conditional_loglik(g::Graph, mu, beta_i, beta_mi, deg)
partial_sparse_beta_conditional_loglik2(beta_i) = sparse_beta_conditional_loglik(g::Graph, mu, beta_i, beta_mi_not, deg2)

l = @layout [ a  b ]
p1 = Plots.plot(collect(0:0.01:4), exp.(partial_sparse_beta_conditional_loglik.(collect(0:0.01:4))))
p2 = Plots.plot(collect(0:0.01:4), exp.(partial_sparse_beta_conditional_loglik2.(collect(0:0.01:4))))
plot(p1, p2, layout=l)
