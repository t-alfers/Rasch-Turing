# Import Packages
using DelimitedFiles, DataFrames;
using ReverseDiff, Turing;
using StatsFuns: logistic;
## using BenchmarkTools;

# helper functions
function intArray(a::CategoricalArray)
    map(x -> Int(x), CategoricalArrays.order(a.pool)[a.refs]);
end

# data handling stuff
verbal = readdlm("./dichotom.txt", '\t', Int, '\n'; header = true);

dfVerbal = DataFrame(verbal[1]);
rename!(dfVerbal,  Symbol.(permutedims(verbal[2])[:]));
dfVerbal.id = 1:nrow(dfVerbal);

verbal_responses = stack(dfVerbal[:, Not([:Anger, :Sex])], Not(:id), :id);
verbal_responses.variable = intArray(verbal_responses.variable);

# statistical modeling
@model irt_1pl(y, ii, jj) = begin
    I = maximum(ii);
    J = maximum(jj);

    # prior item parameter
    itempar_σ ~ truncated(Cauchy(0, 5), 0, Inf);
    itempar ~ filldist(Normal(0, itempar_σ), I)

    # prior person parameter
    perspar_σ ~ truncated(Cauchy(0, 5), 0, Inf);
    perspar ~ filldist(Normal(0, perspar_σ), J);

    # target distribution
    η = logistic.(perspar[jj] - itempar[ii])
    y .~ Bernoulli.(η)
end

Turing.setadbackend(:reversediff)
chn = sample(irt_1pl(verbal_responses.value, verbal_responses.variable, verbal_responses.id), NUTS(1000, 0.8), 2000, progress = true)
# NUTS(1000, 0.8), 2000 :: 276.295 s (5937990687 allocations: 294.14 GiB)
