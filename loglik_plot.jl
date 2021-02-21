using Plots
using LightGraphs
using LinearAlgebra
using StatsBase
using StatsFuns
using GraphPlot

"""
UniDirectionalSparseBeta
"""
function sparse_beta_graph(beta::Vector, mu::Real; seed::Integer=-1)
    @assert all(beta .>= 0) "beta must be non-negative"
    @assert 0 in beta "beta must contain at least one zero"
    n = length(beta)
    coefM = UpperTriangular(beta' .+ beta)
    g = SimpleGraph(n)
    for i in 1:n
        for j in i:n
            expcoef_ij = exp(mu + coefM[i,j])
            if rand() < (expcoef_ij / (1 + expcoef_ij))
                add_edge!(g, i, j)
            end
        end
    end
    return g
end


function sparse_beta_loglik(g::Graph, mu, beta)
    tot = 0
    for i in 1:n
        tot -= degree(g)[i] * beta[i]
        for j in i:n
            tot += log(1 + exp(mu + beta[i] + beta[j]))
        end
    end
    return -(g.ne * mu) + tot
end

function sparse_beta_conditional_loglik(g::Graph, mu, beta_i, beta_mi, deg)
    tot = deg * beta_i
    for b in beta_mi
        tot -= log1pexp(mu + beta_i + b)
    end
    return tot
end

n = 500
beta = zeros(n)
for i in sample(collect(1:n), Int(floor(n / 10)))
    beta[i] = sqrt(log(n))
end
mu = -log(n)

g = sparse_beta_graph(beta, mu)

beta_i = beta[1]
# beta_mi = [b for (i, b) in enumerate(beta) if i != 1]
beta_mi = beta[2:end]
deg = degree(g)[1]

partial_sparse_beta_conditional_loglik(beta_i) = sparse_beta_conditional_loglik(g::Graph, mu, beta_i, beta_mi, deg)

Plots.plot(collect(0:0.01:4), exp.(partial_sparse_beta_conditional_loglik.(collect(0:0.01:4))))