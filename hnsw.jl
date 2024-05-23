include("heap.jl")

mutable struct HNSW
    const M::Int
    const Mmax0::Int
    const ef_construction::Int
    const mL::Float64
    const num_layers::Int

    enter_point::Int
    data::Vector{Int}
    graphs::Dict{Int,Dict{Int,Vector{Int}}}
    counts::Dict{Int,Int}

    function HNSW(num_layers::Int, M=16, Mmax0=32, ef_construction=200)
        graphs = Dict{Int,Dict{Int,Vector{Int}}}()
        for i = 1:num_layers
            d = Dict{Int,Vector{Int}}()
            d[1] = Int[]
            graphs[i] = d
        end
        new(
            M,
            Mmax0,
            ef_construction,
            1 / log(M),
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
    v = min(floor(Int, (-log(rand()) * hnsw.mL) + 1), hnsw.num_layers)
    if haskey(hnsw.counts, v)
        hnsw.counts[v] += 1
    else
        hnsw.counts[v] = 1
    end
    return v
end

function _search_layer(hnsw::HNSW, query::Int, ep::Int, ef::Int, lc::Int)::Vector{Int}
    visited = Set{Int}([ep])
    d = _distance(query, hnsw.data[ep])

    # this might need to be larger?
    candidates = MinHeap(ef)
    W = MaxHeap(ef)

    insert!(candidates, d => ep)
    insert!(W, d => ep)

    while length(candidates) > 0
        d_c, c = pop!(candidates)
        d_f = W[1].first

        if d_c > d_f
            break
        end

        for e in get(hnsw.graphs[lc], c, Int[])
            if e ∉ visited
                push!(visited, e)
                d_e = _distance(query, hnsw.data[e])
                d_f = W[1].first

                if d_e < d_f || length(W) < ef
                    # insert! will automatically remove the largest element if the heap is full
                    insert!(W, d_e => e)
                    insert!(candidates, d_e => e)
                end
            end
        end
    end
    mx = lc == 1 ? hnsw.Mmax0 : hnsw.M
    k = min(length(W), mx)
    data = W.data
    sort!(data, by=x -> x[1])[1:k]
    return Int[data[i][2] for i = 1:k]
end

function Base.insert!(hnsw::HNSW, q::Int)
    push!(hnsw.data, q)
    q_i = length(hnsw.data)
    L = length(hnsw.graphs)

    if q_i == 1
        for lc = L:-1:1
            hnsw.graphs[lc][1] = Int[]
        end
        return
    end

    ep = hnsw.enter_point
    l = min(_get_level(hnsw), L)
    for lc = L:-1:l+1
        W = _search_layer(hnsw, q, ep, 1, lc)
        ep = W[1]
    end

    for lc = l:-1:1
        W = _search_layer(hnsw, q, ep, hnsw.ef_construction, lc)
        neighbors = W

        # bi-directional connection
        hnsw.graphs[lc][q_i] = neighbors
        for n in neighbors
            if !haskey(hnsw.graphs[lc], n)
                hnsw.graphs[lc][n] = Int[q_i]
            else
                if q_i ∉ hnsw.graphs[lc][n]
                    push!(hnsw.graphs[lc][n], q_i)
                end
            end
        end

        # shrink the neighbors
        mx = lc == 1 ? hnsw.Mmax0 : hnsw.M
        for n in neighbors
            if length(hnsw.graphs[lc][n]) > mx
                hnsw.graphs[lc][n] = sort!(
                    hnsw.graphs[lc][n],
                    by=x -> _distance(hnsw.data[x], hnsw.data[n]),
                )[1:mx]
            end
        end
        ep = W[1]
    end

    hnsw.enter_point = ep
end

function search(hnsw::HNSW, query::Int, k::Int, ef::Int=100)
    @info "Searching for $k neighbors" query
    ep = hnsw.enter_point
    L = length(hnsw.graphs)
    for lc = L:-1:2
        W = _search_layer(hnsw, query, ep, 1, lc)
        ep = W[1]
    end
    W = _search_layer(hnsw, query, ep, ef, 1)
    inds = W[1:k]
    return (inds, [hnsw.data[n] for n in inds])
end

function run0()::HNSW
    hnsw = HNSW(10, 16, 32, 200)
    for _ = 1:1000
        insert!(hnsw, rand(1:10_000))
    end
    #
    # # Perform a k-NN search
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
