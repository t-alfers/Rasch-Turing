# Import Packages
using DelimitedFiles, DataFrames;
using ReverseDiff, Turing, Memoization, LazyArrays;
using StatsFuns: logistic;
## using BenchmarkTools;

# helper functions
function intArray(a::CategoricalArray)
    map(x -> Int(x), CategoricalArrays.order(a.pool)[a.refs]);
end
lazyarray(f, x) = LazyArray(Base.broadcasted(f, x))
safelogistic(x::T) where {T} = logistic(x) * (1 - 2 * eps(T)) + eps(T)

# data handling stuff
verbal = readdlm("./dichotom.txt", '\t', Int, '\n'; header = true);

dfVerbal = DataFrame(verbal[1]);
rename!(dfVerbal,  Symbol.(permutedims(verbal[2])[:]));
dfVerbal.id = 1:nrow(dfVerbal);

verbal_responses = DataFrames.stack(dfVerbal[:, Not([:Anger, :Sex])], Not(:id), :id);
verbal_responses.variable = intArray(verbal_responses.variable);

# statistical modeling
@model rasch(y, ii, jj) = begin
  I = maximum(ii);
  J = maximum(jj);

  # prior item parameter
  itempar_σ ~ truncated(Cauchy(0, 5), 0, Inf);
  itempar ~ filldist(Normal(0, itempar_σ), I)

  # prior person parameter
  perspar_σ ~ truncated(Cauchy(0, 5), 0, Inf);
  perspar ~ filldist(Normal(0, perspar_σ), J);

  # target distribution
  y ~ arraydist(lazyarray(x -> Bernoulli(safelogistic(x)), perspar[jj] .- itempar[ii]))
end

Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
chn = sample(rasch(verbal_responses.value, verbal_responses.variable, verbal_responses.id), NUTS(1000, 0.8), 2000, progress = false)
# NUTS(1000, 0.8), 2000 :: 43.783335 seconds (15.58 M allocations: 3.886 GiB, 0.78% gc time)
