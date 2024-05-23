include("heap.jl")

@kwdef mutable struct HNSW
    const connectivity::Int
    const connectivity0::Int
    const expansion_factor::Int
    const mL::Float64
    const num_layers::Int

    enter_point::Int
    data::Vector{Int}
    graphs::Dict{Int,Dict{Int,Vector{Int}}}
    counts::Dict{Int,Int}

    function HNSW(num_layers::Int; connectivity=16, connectivity0=32, expansion_factor=128)
        graphs = Dict{Int,Dict{Int,Vector{Int}}}()
        for i = 1:num_layers
            d = Dict{Int,Vector{Int}}()
            d[1] = Int[]
            graphs[i] = d
        end
        mL = 1 / log(connectivity)
        new(
            connectivity,
            connectivity0,
            expansion_factor,
            mL,
            num_layers,
            1,
            Int[],
            graphs,
            Dict{Int,Int}(),
        )
    end
end

function _distance(x::Int, y::Int)::Int
    return abs(x - y)
end

function _get_level(hnsw::HNSW)::Int
    l = min(floor(Int, (-log(rand()) * hnsw.mL) + 1), hnsw.num_layers)
    if haskey(hnsw.counts, l)
        hnsw.counts[l] += 1
    else
        hnsw.counts[l] = 1
    end
    return l
end

function _search_layer(hnsw::HNSW, query::Int, ep::Int, expansion_factor::Int, level::Int)::Vector{Int}
    visited = Set{Int}([ep])
    candidates = MinHeap(expansion_factor * 5)
    W = MaxHeap(expansion_factor)

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
            if e ∉ visited
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
    end

    mx = level == 1 ? hnsw.connectivity0 : hnsw.connectivity
    k = min(length(W), mx)
    sort!(W.data, by=x -> x[1])
    return Int[W.data[i][2] for i = 1:k]
end

function Base.insert!(hnsw::HNSW, q::Int)
    push!(hnsw.data, q)
    q_i = length(hnsw.data)
    L = length(hnsw.graphs)

    if q_i == 1
        for level = L:-1:1
            hnsw.graphs[level][1] = Int[]
        end
        return
    end

    ep = hnsw.enter_point
    l = _get_level(hnsw)

    for level = L:-1:l+1
        W = _search_layer(hnsw, q, ep, 1, level)
        ep = W[1]
    end

    for level = l:-1:1
        W = _search_layer(hnsw, q, ep, hnsw.expansion_factor, level)
        neighbors = W

        # bi-directional connection
        hnsw.graphs[level][q_i] = neighbors
        for n in neighbors
            if !haskey(hnsw.graphs[level], n)
                hnsw.graphs[level][n] = Int[q_i]
            else
                if q_i ∉ hnsw.graphs[level][n]
                    push!(hnsw.graphs[level][n], q_i)
                end
            end
        end

        # shrink the neighbors if necessary
        mx = level == 1 ? hnsw.connectivity0 : hnsw.connectivity
        for n in neighbors
            if length(hnsw.graphs[level][n]) > mx
                hnsw.graphs[level][n] = sort!(
                    hnsw.graphs[level][n],
                    by=x -> _distance(hnsw.data[x], hnsw.data[n]),
                )[1:mx]
            end
        end
        ep = W[1]
    end

    hnsw.enter_point = ep
    return
end

function search(hnsw::HNSW, query::Int, k::Int; expansion_search::Int=64)
    @info "Searching for $k neighbors" query
    ep = hnsw.enter_point
    L = length(hnsw.graphs)
    for level = L:-1:2
        W = _search_layer(hnsw, query, ep, 1, level)
        ep = W[1]
    end
    W = _search_layer(hnsw, query, ep, expansion_search, 1)[1:k]
    return (W, [hnsw.data[n] for n in W])
end

function run0()::HNSW
    hnsw = HNSW(10)
    for _ = 1:1000
        insert!(hnsw, rand(1:10_000))
    end

    query = rand(1:10_000)
    k = 5
    (inds, vals) = search(hnsw, query, k)
    distances = [_distance(query, data_point) for data_point in hnsw.data]
    k_nearest_manual = sortperm(distances)[1:k]

    println("HNSW inds ", sort(inds))
    println("Manual inds ", sort(k_nearest_manual))
    println("HNSW vals ", sort(vals))
    println("Manual vals ", sort([hnsw.data[i] for i in k_nearest_manual]))

    # print length of each graph layer
    for lc = 1:hnsw.num_layers
        println("Layer $lc: length = $(length(hnsw.graphs[lc]))")
    end

    println("Counts")
    # print the counts
    for (k, v) in hnsw.counts
        println("Level $k: $v")
    end

    return hnsw
end
