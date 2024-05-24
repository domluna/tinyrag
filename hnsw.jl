include("heap.jl")

mutable struct HNSW
    const connectivity::Int
    const mL::Float64

    const graphs::Dict{Int,Dict{Int,Vector{Int}}}

    enter_point::Int
    data::Vector{Int}

    function HNSW(; connectivity = 16, mL = 1 / log(connectivity))
        new(connectivity, mL, Dict{Int,Dict{Int,Vector{Int}}}(), 1, Int[])
    end
end

function _distance(x::Int, y::Int)::Int
    return abs(x - y)
end

function _get_level(hnsw::HNSW)::Int
    floor(Int, (-log(rand()) * hnsw.mL) + 1)
end

function _search_layer(
    hnsw::HNSW,
    query::Int,
    ep::Int,
    expansion_factor::Int,
    level::Int;
    k::Int = 0,
)::Vector{Int}
    visited = Set{Int}([ep])
    candidates = MinHeap(expansion_factor)
    W = MaxHeap(k > 0 ? k : expansion_factor)

    d = _distance(query, hnsw.data[ep])
    insert!(candidates, d => ep)
    insert!(W, d => ep)

    while length(candidates) > 0
        d_c, c = pop!(candidates)
        d_f = W[1].first

        if d_c > d_f
            break
        end

        for e in get(hnsw.graphs[level], c, Int[])
            if e ∈ visited
                continue
            end
            push!(visited, e)
            d_e = _distance(query, hnsw.data[e])
            d_f = W[1].first

            if d_e < d_f || length(W) < expansion_factor
                # insert! will automatically remove the largest element if the heap is full
                insert!(W, d_e => e)
                insert!(candidates, d_e => e)
            end
        end
    end

    mx = hnsw.connectivity
    k = min(length(W), mx)
    sort!(W.data, by = x -> x[1])
    return Int[W.data[i][2] for i = 1:k]
end

function Base.insert!(hnsw::HNSW, q::Int; expansion_factor::Int = 128)
    push!(hnsw.data, q)
    ind = length(hnsw.data)
    ep = hnsw.enter_point
    l = _get_level(hnsw)
    new_entry_point = l > length(hnsw.graphs)

    if !haskey(hnsw.graphs, l)
        hnsw.graphs[l] = Dict{Int,Vector{Int}}()
        for level = l-1:-1:1
            if !haskey(hnsw.graphs, level)
                hnsw.graphs[level] = Dict{Int,Vector{Int}}()
            end
        end
    end
    L = length(hnsw.graphs)

    for level = L:-1:l+1
        W = _search_layer(hnsw, q, ep, 1, level)
        ep = W[1]
    end

    for level = l:-1:1
        neighbors = _search_layer(hnsw, q, ep, expansion_factor, level)
        # bi-directional connection
        # hnsw.graphs[level][ind] = neighbors
        hnsw.graphs[level][ind] = Int[]
        for n in neighbors
            if n == ind
                continue
            end
            push!(hnsw.graphs[level][ind], n)
            if !haskey(hnsw.graphs[level], n)
                hnsw.graphs[level][n] = Int[ind]
            else
                if ind ∉ hnsw.graphs[level][n]
                    push!(hnsw.graphs[level][n], ind)
                end
            end
        end

        # shrink the neighbors if necessary
        mx = hnsw.connectivity
        for n in neighbors
            if length(hnsw.graphs[level][n]) > 2 * mx
                hnsw.graphs[level][n] = sort!(
                    hnsw.graphs[level][n],
                    by = x -> _distance(hnsw.data[x], hnsw.data[n]),
                )[1:mx]
            end
        end
        ep = neighbors[1]
    end

    if new_entry_point
        hnsw.enter_point = ep
    end
    return
end

function search(hnsw::HNSW, query::Int, k::Int; expansion_search::Int = 64)
    ep = hnsw.enter_point
    # @info "Searching for $k neighbors" query ep
    L = length(hnsw.graphs)
    for level = L:-1:2
        # @time W = _search_layer(hnsw, query, ep, 1, level)
        W = _search_layer(hnsw, query, ep, 1, level)
        ep = W[1]
    end
    # @time W = _search_layer(hnsw, query, ep, expansion_search, 1; k=k)
    W = _search_layer(hnsw, query, ep, expansion_search, 1; k = k)
    # return W
    inds = [W[i] for i = 1:k]
    vals = [hnsw.data[i] for i in inds]
    return (inds, vals)
end

function run0(n::Int)::HNSW
    hnsw = HNSW(; connectivity = 16)
    rng = 1:1_000_000
    for _ = 1:n
        insert!(hnsw, rand(rng))
    end
    search0(rand(rng), hnsw, 5)
    return hnsw
end

function search0(query::Int, hnsw::HNSW, k::Int)
    (inds, vals) = search(hnsw, query, k)
    distances = [_distance(query, data_point) for data_point in hnsw.data]
    k_nearest_manual = sortperm(distances)[1:k]

    @info "Searching for $k neighbors" query
    println("HNSW inds ", sort(inds))
    println("Manual inds ", sort(k_nearest_manual))
    println("HNSW vals ", sort(vals))
    println("Manual vals ", sort([hnsw.data[i] for i in k_nearest_manual]))

    # print length of each graph layer
    for lc = 1:length(hnsw.graphs)
        println("Layer $lc: length = $(length(hnsw.graphs[lc]))")
    end
    return
end

function search0(hnsw::HNSW, k::Int)
    q = rand(1:1_000_000)
    search0(q, hnsw, k)
end
