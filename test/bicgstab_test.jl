include("../src/constraint_function.jl")
using .constraint_function, Statistics

mvps = []
for i in 1:1000
	@show i
	C = ConstraintFunction(16)
	_, mvp = C(rand(Float64))
	global mvps = vcat(mvps, [mvp])
end
@show mean(mvps)
@show std(mvps)
