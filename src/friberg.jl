using OrdinaryDiffEq

log_transform(x) = log10.(x)
inverse_log_transform(x) = 10.0 .^ x

"""
    E_drug_step(t, slope, therapies_tuple)

Drug effect function for Friberg model.

# Arguments
- `t`: time in hours
- `slope`: E_eff individual parameter of drug effect
- `therapies_tuple`: treatment times and dose information, tuple (therapies, dosage)

# Returns
- Drug effect at time t
"""
function E_drug_step(t, slope, therapies_tuple)
    therapies, dosage = therapies_tuple

    if t < minimum(therapies)
        return zero(slope)
    end

    local_start = maximum(therapies[therapies .<= t])
    local_dosage = dosage[findfirst(==(local_start), therapies)]
    t_loc = t - local_start

    if t_loc <= 23
        return slope * local_dosage
    else
        return zero(slope)
    end
end

"""
    friberg_drug!(dy, y, p, t)

Differential equations defining Friberg model including cell loss due to
E_drug. Original publication: Friberg et al., JCO. 20, 4713–4721 (2002),
https://doi.org/10.1200/jco.2002.02.140

# Arguments
- `dy`: derivative vector (output)
- `y`: vec of compartments [prol, t1, t2, t3, circ]
- `p`: parameters tuple (ktr, gamma, c0, slope, therapies)
- `t`: time in hours
"""
function friberg_drug!(dy, y, p, t)
    ktr, gamma, c0, slope, therapies = p
    prol, t1, t2, t3, circ = y

    Edrug = E_drug_step(t, slope, therapies)

    dy[1] = ktr * prol * (1 - Edrug) * (c0 / circ)^gamma - ktr * prol
    dy[2] = ktr * prol - ktr * t1
    dy[3] = ktr * t1 - ktr * t2
    dy[4] = ktr * t2 - ktr * t3
    dy[5] = ktr * t3 - ktr * circ
    return nothing
end

"""
    solve_friberg(gamma, MTT, slope, c0, therapies_tuple, end_fu, t_eval)

Solve Friberg model equations for given parameters and therapy plan.
Therapy starting time points given in days and standardized doses.
Original publication: Friberg et al., JCO. 20, 4713–4721 (2002),
https://doi.org/10.1200/jco.2002.02.140

# Arguments
- `gamma`: exponent for feedback
- `MTT`: cell maturation time in hours
- `slope`: slope for linear drug effect for drug function
- `c0`: baseline of circulating blood cells/l
- `therapies_tuple`: (therapy days, dosage in average dosage units)
- `end_fu`: follow up after last periodic treatment in days
- `t_eval`: evaluation time points in days

# Returns
- Number of circulating blood cells/l at t_eval
"""
function solve_friberg(gamma::Real, MTT::Real, slope::Real, c0::Real,
                       therapies_tuple, end_fu::Real, t_eval::AbstractVector{<:Real})
    therapies, dosage = therapies_tuple
    therapies_hours = Float64.(therapies) .* 24.0
    dosage_f = Float64.(dosage)

    T = promote_type(typeof(gamma), typeof(MTT), typeof(slope), typeof(c0))
    ktr = 4 / MTT
    y0 = fill(T(c0), 5)

    tmax = maximum(therapies_hours) + end_fu * 24.0
    t_eval_hours = Float64.(t_eval) .* 24.0

    p = (ktr, gamma, c0, slope, (therapies_hours, dosage_f))
    tstops = vcat(therapies_hours, therapies_hours .+ 23.0)

    prob = ODEProblem(friberg_drug!, y0, (0.0, tmax), p)
    sol = solve(prob, Tsit5(); saveat = t_eval_hours, tstops = tstops,
                reltol = 1e-6, abstol = 1e-9, dtmax = 24.0)

    return [sol.u[findfirst(==(t), sol.t)][5] for t in t_eval_hours]
end

"""
    poplog_fit_friberg_mse(params, pop_params, y_arr, t_arr, therapies)

Loss function to calibrate Friberg model given population context,
on logarithmic scale to emphasize nadir calibration.

# Arguments
- `params`: model parameter guess [gamma, MTT, slope, c0/10^9]
- `pop_params`: population parameters [gamma, MTT, slope, c0/10^9]
- `y_arr`: observation to calibrate the Friberg model
- `t_arr`: observation times in days
- `therapies`: treatment times in days and accompanying doses

# Returns
- Loss function, mse + population penalty
"""
function poplog_fit_friberg_mse(params::AbstractVector{<:Real}, pop_params::AbstractVector{<:Real},
                                y_arr::AbstractVector{<:Real}, t_arr::AbstractVector{<:Real},
                                therapies)
    gamma, MTT, slope, c0t = params
    c0 = c0t * 1e9

    y_pred = solve_friberg(gamma, MTT, slope, c0, therapies,
                           t_arr[end] + 1 - therapies[1][end], t_arr)

    y_log = log10.(y_pred)
    y_arr_log = log10.(y_arr)
    finite_mask = isfinite.(y_log)

    @assert length(y_log) == length(y_arr_log) "Length mismatch: y_pred=$(length(y_pred)), y_arr=$(length(y_arr))"

    mse = sum((y_log[finite_mask] .- y_arr_log[finite_mask]) .^ 2) / length(y_arr)
    param_loss = ((log.(params) .- log.(pop_params)) .^ 2) ./ (5.0^2)

    return mse + sum(param_loss)
end

"""
    friberg(params, t_arr, therapies)

Generate Friberg model prediction with given parameters.

# Arguments
- `params`: model parameter guess [gamma, MTT, slope, c0/10^9]
- `t_arr`: days to evaluate
- `therapies`: treatment in administration days and accompanying doses

# Returns
- Predicted time series
"""
function friberg(params::AbstractVector{<:Real}, t_arr::AbstractVector{<:Real}, therapies)
    gamma, MTT, slope, c0t = params
    c0 = c0t * 1e9
    return solve_friberg(gamma, MTT, slope, c0, therapies,
                         t_arr[end] + 1 - therapies[1][end], t_arr)
end

"""
    test_friberg(pc, therapies, t_arr, scale_fn)

Helper function for testing Friberg model in context of transfer learning.

# Arguments
- `pc`: Friberg model parameters [gamma, MTT, slope, c0/10^9]
- `therapies`: treatment in administration days and accompanying doses
- `t_arr`: evaluation times in days, daily and starting at day 0
- `scale_fn`: function that scales a vector of log10 values

# Returns
- Tuple of (X_raw, y_scaled, y_raw)
"""
function test_friberg(pc::AbstractVector{<:Real}, therapies,
                      t_arr::AbstractVector{<:Real}, scale_fn)
    gamma, MTT, slope, c0t = pc
    c0 = c0t * 1e9

    y_raw = solve_friberg(gamma, MTT, slope, c0, therapies,
                          t_arr[end] - therapies[1][end], t_arr)

    X_raw = zeros(length(y_raw))
    X_raw[Int.(therapies[1]) .+ 1] .= therapies[2]

    y = scale_fn(log_transform(y_raw))

    return X_raw, y, y_raw
end