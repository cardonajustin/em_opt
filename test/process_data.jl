using Serialization, Statistics, Plots


ZS = []
NS = []
MS = []
SS = []
for i in [2, 4, 8, 16]
	n_volume = i
	dx=1e-3
	zs = deserialize("data/zeros_"*string(n_volume)*"_"*string(dx)*".dat")
	ns = deserialize("data/iters_"*string(n_volume)*"_"*string(dx)*".dat")
	global ZS = vcat(ZS, zs[:])
	global NS = vcat(NS, ns[:])
	global MS = vcat(MS, [mean(ns[:])])
	global SS = vcat(SS, [std(ns[:])])
end

plot(map(n-> 3*n^3, [2, 4, 8, 16]), MS, ribbon=SS, fillalpha=.5, legend=false, xlabel="System size", ylabel="Expected Number of Inverse Solves", xaxis=:log)
savefig("data/perturbation_iters.png")
histogram(NS[:], bins=2:1:10, legend=false, normalize=:probability)
savefig("data/histogram.png")
@show mean(NS[:])
@show std(NS[:])
@show MS
