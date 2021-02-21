using LightGraphs
using LinearAlgebra
using StatsBase

function erdos_renyi(n::Integer, p::Real; is_directed=false, seed::Integer=-1)
    m = is_directed ? n * (n - 1) : div(n * (n - 1), 2)
    ne = randbn(m, p, seed)
    return is_directed ? SimpleDiGraph(n, ne, seed=seed) : SimpleGraph(n, ne, seed=seed)
end

G = erdos_renyi(100, .1)


"""
UniDirectionalBeta
"""
function beta_graph(beta::Vector; seed::Integer=-1)
    n = length(beta)
    coefM = UpperTriangular(beta' .+ beta)
    g = SimpleGraph(n)
    for i in 1:n
        for j in i:n
            expcoef_ij = exp(coefM[i,j])
            if rand() < (expcoef_ij / (1 + expcoef_ij))
                add_edge!(g, i, j)
            end
        end
    end
    return g
end


"""
BiDirectionalBeta
"""
function beta_graph(alpha::Vector, beta::Vector; seed::Integer=-1)
    n = length(beta)
    @assert length(alpha) == n "alpha and beta must be the same length"
    coefM = alpha' .+ beta
    g = SimpleDiGraph(n)
    for i in 1:n
        for j in 1:n
            expcoef_ij = exp(coefM[i,j])
            if rand() < (expcoef_ij / (1 + expcoef_ij))
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

"""
UniDirectionalSparseBeta
"""
function sparse_beta_graph(beta::Vector, mu::Real; seed::Integer=-1)
    @assert all(beta .>= 0) "beta must be non-negative"
    @assert minimum(beta) == 0 "beta must contain at least one zero"
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


n = 500
beta = zeros(n)
for i in sample(collect(1:n), Int(floor(n / 10)))
    beta[i] = sqrt(log(n))
end
mu = -log(n)

g = sparse_beta_graph(beta, mu)

using GraphPlot

layout = (args...) -> spring_layout(args...; C=10)
gplot(g, layout=layout)