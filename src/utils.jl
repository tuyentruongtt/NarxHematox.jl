const ACTIVATIONS = Dict("sigmoid" => σ, "linear" => identity, "tanh" => tanh, "relu" => relu)

activation(f::Function) = f
activation(name::AbstractString) = ACTIVATIONS[name]

"""
    InputNoise(stddev)

Adds Gaussian noise to the input during training only.
"""
struct InputNoise
    stddev::Float32
end

(m::InputNoise)(x) = Flux.NNlib.within_gradient(x) ? x .+ m.stddev .* randn(Float32, size(x)) : x

"""
    Prune(delta)

Magnitude pruning: sets weights with `abs(w) < delta` to zero.
"""
struct Prune
    delta::Float32
end

(p::Prune)(w) = w .* (abs.(w) .>= p.delta)

"""
    LP_regularizer(lp, p)

LP penalty: `lp * sum(abs(x)^p)` over non-zero entries.
"""
struct LP_regularizer
    lp::Float32
    p::Float32
end

LP_regularizer(; lp = 0.01, p = 1.0) = LP_regularizer(Float32(lp), Float32(p))

(r::LP_regularizer)(x) = r.lp * sum(abs.(x[x .!= 0]) .^ r.p)

"""
    prune!(model, p::Prune)

Apply magnitude pruning to the input kernels, as in the Keras constraints.
Call after each optimiser step.
"""
function prune!(m::ARX_RNN_ES, p::Prune)
    m.cell.Wi .= p(m.cell.Wi)
    m.dense.weight .= p(m.dense.weight)
    return m
end

function prune!(m::ARX_net_lagged, p::Prune)
    for layer in m.mlp.layers
        layer isa Dense && (layer.weight .= p(layer.weight))
    end
    return m
end

"""
    regularization(model, reg)

Weight penalty to add to the loss: L2 on the output layer for `ARX_RNN_ES`,
LP on the hidden and output layers for an `ARX_net_lagged` built by `buildmlp`.
"""
regularization(m::ARX_RNN_ES, out_l2::Real) = out_l2 * sum(abs2, m.dense.weight)
regularization(m::ARX_net_lagged, r::LP_regularizer) = r(m.mlp[end-1].weight) + r(m.mlp[end].weight)

"""
    mse_missing(y_true, y_pred)

MSE ignoring non-finite targets.
"""
function mse_missing(y_true, y_pred)
    yt, yp = vec(y_true), vec(y_pred)
    mask = isfinite.(yt)
    d = yp[mask] .- yt[mask]
    return sum(abs2, d) / length(d)
end

"""
    step_mse_missing(y_true, y_pred, s)

MSE ignoring non-finite targets, plus neighbouring time steps weighted by `s`.
"""
function step_mse_missing(y_true, y_pred, s)
    yt, yp = vec(y_true), vec(y_pred)
    mask = isfinite.(yt)
    mask_minus = vcat(mask[2:end], mask[1:1])
    mask_plus = vcat(mask[end:end], mask[1:end-1])
    target = yt[mask]
    terms = (yp[mask] .- target) .^ 2 .+
            s .* (0.3f0 .* (yp[mask_minus] .- target) .^ 2 .+
                  0.3f0 .* (yp[mask_plus] .- target) .^ 2)
    return sum(terms) / length(terms)
end

"""
    smoothing_penalty(y_pred; k, c)

Penalty on deviations from the local mean over `k` neighbours, weighted by `c`.
Uses the same window as the Python version (2k values divided by 2k + 1).
"""
function smoothing_penalty(y_pred; k, c)
    y = vec(y_pred)
    n = length(y)
    window = y[1:n-2k]
    for j in 1:2k-1
        window = window .+ y[1+j:n-2k+j]
    end
    dev = y[k+1:n-k] .- window ./ (2k + 1)
    return c * sum(abs2, dev)
end

"""
    smse_smoothing(y_true, y_pred; k, c, s)

`step_mse_missing` plus `smoothing_penalty`.
"""
smse_smoothing(y_true, y_pred; k, c, s) =
    step_mse_missing(y_true, y_pred, s) + smoothing_penalty(y_pred; k = k, c = c)

"""
    combined_loss(y_true, y_pred, loss_fns)

Sum of several loss functions `(y_true, y_pred) -> scalar`.
"""
combined_loss(y_true, y_pred, loss_fns) = sum(f(y_true, y_pred) for f in loss_fns)

"""
    EarlyStoppingAveraged(av_epochs, patience; min_delta = 0, restore_best_weights = true)

Stops training when the loss averaged over `av_epochs` epochs stops improving.
"""
mutable struct EarlyStoppingAveraged
    av_epochs::Int
    patience::Int
    min_delta::Float64
    restore::Bool
    best_av_loss::Float64
    best_epoch::Int
    best_model::Any
end

EarlyStoppingAveraged(av_epochs::Int, patience::Int; min_delta = 0.0, restore_best_weights = true) =
    EarlyStoppingAveraged(av_epochs, patience, min_delta, restore_best_weights, Inf, 0, nothing)

function on_train_begin!(es::EarlyStoppingAveraged, model)
    es.best_av_loss = Inf
    es.best_epoch = 0
    es.best_model = deepcopy(model)
    return nothing
end

"""
    on_epoch_end!(es, model, history) -> Bool

Call after each epoch with the vector of monitored losses so far.
Returns `true` if training should stop.
"""
function on_epoch_end!(es::EarlyStoppingAveraged, model, history::AbstractVector)
    epoch = length(history) - 1
    epoch < es.av_epochs && return false

    av_loss = sum(history[end-es.av_epochs+1:end]) / es.av_epochs
    if av_loss < es.best_av_loss
        if es.best_av_loss - av_loss < es.min_delta && epoch - es.best_epoch >= es.patience
            es.restore && Flux.loadmodel!(model, es.best_model)
            return true
        end
        es.best_epoch = epoch
        es.best_av_loss = av_loss
        es.restore && (es.best_model = deepcopy(model))
    end

    if epoch - es.best_epoch >= es.patience
        es.restore && Flux.loadmodel!(model, es.best_model)
        return true
    end
    return false
end

"""
    buildrnn(hp)

Build an `ARX_RNN_ES` with a `TFGRUCell` from a hyperparameter `Dict` with keys
`:units, :lag, :stddev, :activation, :out_act, :out_bias` and optional `:in_size` (default 2).
"""
function buildrnn(hp::AbstractDict)
    activation(hp[:activation]) === σ || error("TFGRUCell only supports sigmoid activation")
    inp = hp[:stddev] != 0 ? InputNoise(hp[:stddev]) : identity
    cell = TFGRUCell(get(hp, :in_size, 2), hp[:units])
    out = Dense(hp[:units] => 1, activation(hp[:out_act]); bias = hp[:out_bias])
    return ARX_RNN_ES(cell, hp[:lag]; inp = inp, out = out)
end

"""
    buildmlp(hp)

Build the inner MLP for `ARX_net_lagged` from a hyperparameter `Dict` with keys
`:xlag, :ylag, :units, :stddev, :activation, :out_act, :out_bias`.
"""
function buildmlp(hp::AbstractDict)
    init(dims...) = 0.3f0 .* randn(Float32, dims...)
    inp = hp[:stddev] != 0 ? InputNoise(hp[:stddev]) : identity
    hidden = Dense(hp[:xlag] + hp[:ylag] => hp[:units], activation(hp[:activation]);
                   bias = false, init = init)
    out = Dense(hp[:units] => 1, activation(hp[:out_act]); bias = hp[:out_bias], init = init)
    return Chain(inp, hidden, out)
end

"""
    buildNARX(hp)

Build an `ARX_net_lagged` from a hyperparameter `Dict` (keys of `buildmlp` plus `:es`).
"""
buildNARX(hp::AbstractDict) = ARX_net_lagged(buildmlp(hp), hp[:xlag], hp[:ylag]; es = hp[:es])

"""
    with_input_noise(model, stddev)

Copy of `model` that shares its weights, with Gaussian input noise `stddev` during training.
"""
function with_input_noise(m::ARX_RNN_ES, stddev)
    inp = stddev != 0 ? InputNoise(stddev) : identity
    return ARX_RNN_ES(m.lags, m.cell, inp, m.dense, m.alpha, m.constraint)
end

function with_input_noise(m::ARX_net_lagged, stddev)
    inp = stddev != 0 ? InputNoise(stddev) : identity
    return ARX_net_lagged(m.xlag, m.ylag, Chain(inp, m.mlp[end-1], m.mlp[end]), m.alpha, m.es, m.constraint)
end