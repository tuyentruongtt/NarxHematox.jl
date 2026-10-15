using HDF5

"""
    index_weights_path(celltype)

Path to the bundled pretrained index weights, `celltype` is `"GRU"` or `"FNN"`.
"""
index_weights_path(celltype::AbstractString) =
    joinpath(@__DIR__, "..", "data", "index_weights_$(celltype).h5")

"""
    load_weights!(model, path)

Load Keras `.h5` weights saved by the original Python implementation
into an `ARX_RNN_ES` (GRU) or an `ARX_net_lagged` built by `buildmlp` (FNN).
"""
function load_weights!(m::ARX_RNN_ES, path::AbstractString)
    h5open(path, "r") do f
        g = "rnn/arx_rnn_es/rnn/gru_cell/"
        m.cell.Wi .= permutedims(read(f[g * "kernel:0"]))
        m.cell.Wh .= permutedims(read(f[g * "recurrent_kernel:0"]))
        m.cell.b .= vec(read(f[g * "bias:0"]))
        m.dense.weight .= read(f["dense/arx_rnn_es/dense/kernel:0"])
        m.alpha .= read(f["top_level_model_weights/ES:0"])
    end
    return m
end

function load_weights!(m::ARX_net_lagged, path::AbstractString)
    h5open(path, "r") do f
        m.mlp[end-1].weight .= read(f["sequential/hidden/kernel:0"])
        m.mlp[end].weight .= read(f["sequential/out/kernel:0"])
        m.alpha .= read(f["top_level_model_weights/ES:0"])
    end
    return m
end