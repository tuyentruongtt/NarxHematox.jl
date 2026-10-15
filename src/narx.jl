using Flux
using Zygote

struct MinMax
    w_min::Float32
    w_max::Float32
end

(c::MinMax)(w) = clamp.(w, c.w_min, c.w_max)

"""
    TFGRUCell(in_size, units)

GRU cell matching TensorFlow's GRU (`reset_after=true`, sigmoid activation).
Gate order: z, r, h.
"""
struct TFGRUCell
    Wi::Matrix{Float32}
    Wh::Matrix{Float32}
    b::Vector{Float32}
end

function TFGRUCell(in_size::Int, units::Int)
    Wi = Flux.glorot_uniform(in_size, 3units)
    Wh = Flux.glorot_uniform(units, 3units)
    b = zeros(Float32, 6units)
    return TFGRUCell(Wi, Wh, b)
end

Flux.@layer TFGRUCell

initial_state(m::TFGRUCell) = zeros(Float32, size(m.Wh, 1), 1)

function (m::TFGRUCell)(x::AbstractVecOrMat, h::AbstractVecOrMat)
    u = size(m.Wh, 1)
    gx = x' * m.Wi .+ reshape(m.b[1:3u], 1, :)
    gh = h' * m.Wh .+ reshape(m.b[3u+1:end], 1, :)
    z = σ.(gx[:, 1:u] .+ gh[:, 1:u])
    r = σ.(gx[:, u+1:2u] .+ gh[:, u+1:2u])
    h̃ = σ.(gx[:, 2u+1:end] .+ r .* gh[:, 2u+1:end])
    h_new = permutedims(z .* h' .+ (1 .- z) .* h̃)
    return h_new, h_new
end

"""
    ARX_RNN_ES(cell, lags; inp = identity, out = Dense(units => 1))

ARX-RNN with exponential smoothing.
Input shape: (features, time, batch). Output shape: (1, time - lags, batch).
"""
struct ARX_RNN_ES{C,I,D}
    lags::Int
    cell::C
    inp::I
    dense::D
    alpha::Matrix{Float32}
    constraint::MinMax
end

function ARX_RNN_ES(cell, lags::Int; inp = identity,
                    out = Dense(size(initial_state(cell), 1) => 1))
    return ARX_RNN_ES(lags, cell, inp, out, ones(Float32, 1, 1), MinMax(0f0, 1f0))
end

Flux.@layer ARX_RNN_ES
Flux.trainable(m::ARX_RNN_ES) = (cell = m.cell, dense = m.dense, alpha = m.alpha)

function warmup(m::ARX_RNN_ES, inputs)
    state = initial_state(m.cell)
    out = state
    for t in 1:m.lags
        out, state = m.cell(m.inp(inputs[:, t, :]), state)
    end
    return m.dense(out), state
end

function (m::ARX_RNN_ES)(inputs)
    lags = m.lags
    n = size(inputs, 2) - lags
    batch = size(inputs, 3)
    alpha = m.constraint(m.alpha)

    prediction, state = warmup(m, inputs)
    es_pred = prediction
    buf = Zygote.Buffer(similar(inputs, batch, n))
    buf[:, 1] = vec(es_pred)

    for i in 1:(n - 1)
        x_in = m.inp(vcat(inputs[1:1, lags + i, :], prediction))
        out, state = m.cell(x_in, state)
        prediction = m.dense(out)
        es_pred = alpha .* prediction .+ (1 .- alpha) .* es_pred
        buf[:, i + 1] = vec(es_pred)
    end

    return reshape(permutedims(copy(buf)), 1, n, batch)
end

"""
    ARX_net_lagged(mlp, xlag, ylag; es = false)

ARX-FNN with optional exponential smoothing (`es`).
Input shape: (features, time, batch), features ordered as [y lags..., x lags...].
Output shape: (1, time - 1, batch).
"""
struct ARX_net_lagged{M}
    xlag::Int
    ylag::Int
    mlp::M
    alpha::Matrix{Float32}
    es::Bool
    constraint::MinMax
end

function ARX_net_lagged(mlp, xlag::Int, ylag::Int; es::Bool = false)
    return ARX_net_lagged(xlag, ylag, mlp, ones(Float32, 1, 1), es, MinMax(0f0, 1f0))
end

Flux.@layer ARX_net_lagged
Flux.trainable(m::ARX_net_lagged) = m.es ? (mlp = m.mlp, alpha = m.alpha) : (mlp = m.mlp,)

function (m::ARX_net_lagged)(inputs)
    ylag = m.ylag
    n = size(inputs, 2) - 1
    batch = size(inputs, 3)
    alpha = m.constraint(m.alpha)

    prediction = m.mlp(inputs[:, 1:1, :])
    reg_input = vcat(inputs[2:ylag+1, 1:1, :], prediction)
    new_inputs = vcat(reg_input, inputs[ylag+2:end, 2:2, :])
    es_pred = prediction
    buf = Zygote.Buffer(similar(inputs, batch, n))
    buf[:, 1] = vec(prediction)

    for i in 1:(n - 1)
        prediction = m.mlp(new_inputs)
        reg_input = vcat(reg_input[2:end, :, :], prediction)
        new_inputs = vcat(reg_input, inputs[ylag+2:end, i+2:i+2, :])
        es_pred = alpha .* prediction .+ (1 .- alpha) .* es_pred
        buf[:, i + 1] = vec(es_pred)
    end

    return reshape(permutedims(copy(buf)), 1, n, batch)
end