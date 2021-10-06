# Import Packages
using Random, Distributions, Turing, Memoization, LazyArrays, ReverseDiff
using CSV, DataFrames, CategoricalArrays

# helper functions
lazyarray(f, x) = LazyArray(Base.broadcasted(f, x))

# Settings for Turing
Random.seed!(1234)
Turing.setrdcache(true)
Turing.setadbackend(:reversediff)

# Data Handling stuff
url = "https://raw.githubusercontent.com/t-alfers/Rasch-Turing/master/dichotom.txt"
data = CSV.read(download(url), DataFrame, delim = "\t")
data.person = collect(1:nrow(data));
data_responses = data[:, Not([:Anger, :Sex])];

data_long = DataFrames.stack(data_responses, Not(:person), :person);
data_long.variable = CategoricalArray(data_long.variable)
levels!(data_long.variable, unique(data_long.variable))
data_long.variable = levelcode.(data_long.variable)

# statistical modeling
@model rasch(y, ii, jj) = begin
  I = maximum(ii);
  J = maximum(jj);
  N = I * J;

  # prior item parameter
  β_σ ~ truncated(Cauchy(0, 5), 0, Inf);
  β ~ filldist(Normal(0, β_σ), I);

  # prior person parameter
  θ_σ ~ truncated(Cauchy(0, 5), 0, Inf);
  θ ~ filldist(Normal(0, θ_σ), J);

  # target distribution
  y ~ arraydist(lazyarray(x -> BernoulliLogit(x), θ[jj] .- β[ii]))
end

@time chn = sample(rasch(data_long.value, data_long.variable, data_long.person),
             NUTS(1000, 0.8), 2000, progress = true)

# LazyArrays 0.16.16
# NUTS(1000, 0.8), 2000 :: 43.783335 seconds (15.58 M allocations: 3.886 GiB, 0.78% gc time)
#
# LaryArrays 0.22.2
# NUTS(1000, 0.8), 2000 :: 354.621508 seconds (16.82 M allocations: 3.857 GiB, 0.09% gc time)
#
