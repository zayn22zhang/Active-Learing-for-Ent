# Thin bridge: adaptation calls the pinned upstream implementation unchanged.
module AL4QEDOhst
using LinearAlgebra, Random
using MultiStates
using Convex, SCS, Mosek, MosekTools, MathOptInterface
const MOI = MathOptInterface
runtime_info() = (string(VERSION), string(Base.pkgversion(Convex)),
                  string(Base.pkgversion(SCS)), string(Base.pkgversion(MosekTools)))

function query(mat, dims, seed, nvert, niter, tol, solver_name)
    Random.seed!(seed)
    rho = MultiState(Matrix{ComplexF64}(mat), Vector{Int}(dims))
    solver = solver_name == "MOSEK" ? MosekTools.Optimizer :
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs"=>1e-7, "eps_rel"=>1e-7, "max_iters"=>100000)
    polytope = RandomBlochPolytope(rho.dims[1], nvert)
    result = RobustnessToSeparabilityByBlochPolytope(
        rho, polytope, niter, solver, true, tol, true)
    upstream_chi = Float64(result[1])
    isfinite(upstream_chi) || error("Upstream returned nonfinite visibility")
    # Upstream substitutes tiny-weight factors and omits status in its return.
    # Re-solve once on its final adapted polytope to obtain an intact certificate.
    vertices = Matrix{ComplexF64}[]
    for v in result[3]
        h = Hermitian((v+v')/2)
        minimum(eigvals(h)) >= -1e-6 || error("Non-PSD upstream vertex")
        e = eigen(h)
        positive = e.vectors * Diagonal(max.(e.values,0.0)) * e.vectors'
        real(tr(positive)) > 1e-12 || continue
        push!(vertices,positive/real(tr(positive)))
    end
    isempty(vertices) && error("Upstream returned an empty polytope")
    da, db = rho.dims
    dim = da*db
    t = Variable()
    factors = [HermitianSemidefinite(db) for _ in vertices]
    expression = sum(kron(v,z) for (v,z) in zip(vertices,factors))
    identity = Matrix{ComplexF64}(I,dim,dim)/dim
    problem = maximize(t,[t>=0,t<=1,t*rho.mat+(1-t)*identity==expression])
    solve!(problem,solver; silent_solver=true)
    problem.status == MOI.OPTIMAL || error("Final verification failed: $(problem.status)")
    t_value = evaluate(t)
    chi = t_value isa Number ? Float64(t_value) : Float64(only(t_value))
    values = [Matrix{ComplexF64}(evaluate(z)) for z in factors]
    reconstruction = sum(kron(v,z) for (v,z) in zip(vertices,values))
    residual = norm(chi*rho.mat+(1-chi)*identity-reconstruction)
    min_eigenvalue = minimum(minimum(eigvals(Hermitian((z+z')/2))) for z in values)
    (isfinite(chi) && -1e-6 <= chi <= 1+1e-6 && residual <= 1e-5 && min_eigenvalue >= -1e-6) ||
        error("Primal validation failed: residual=$residual, min_eigenvalue=$min_eigenvalue")
    return (clamp(chi,0.0,1.0),residual,upstream_chi)
end
end
