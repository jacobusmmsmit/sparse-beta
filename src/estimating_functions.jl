using LightGraphs

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

function sparse_beta_conditional_loglik(g::Graph, mu, beta, i,)
    beta_i = beta[i]
    tot = degree(g)[i] * beta_i
    for b in beta[1:end .!= i]
        tot -= log1pexp(mu + beta_i + b)
    end
    return tot
end

function sparse_beta_conditional_loglik(g::Graph, mu, beta_i, beta_mi, deg)
    tot = deg * beta_i
    for b in beta[1:end .!= i]
        tot -= log1pexp(mu + beta_i + b)
    end
    return tot
end