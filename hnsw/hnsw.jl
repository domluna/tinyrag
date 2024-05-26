using StaticArrays

include("heap.jl")

mutable struct HNSW
    # set this higher based on the dimensionality of the data
    # the paper suggests a range from 5 to 48
    const connectivity::Int
    # the number of connections to keep at the zeroth layer
    # the paper suggests this to be twice the connectivity
    # lower and the search is not as good, higher and the search is too slow
    const connectivity0::Int
    # optimal value seems to be 1 / log(connectivity)
    const mL::Float64

    const graphs::Dict{Int,Dict{Int,Vector{Int}}}

    enter_point::Int
    data::Vector{SVector{8,UInt64}}

    function HNSW(connectivity::Int; connectivity0::Int=connectivity * 2, mL::Float64=1 / log(connectivity))
        v = Vector{SVector{8,UInt64}}[]
        new(connectivity, connectivity0, mL, Dict{Int,Dict{Int,Vector{Int}}}(), 1, v)
    end
end

@inline function hamming_distance(x1::T, x2::T)::Int where {T<:Integer}
    return Int(count_ones(x1 ⊻ x2))
end

@inline function hamming_distance(
    x1::AbstractArray{T},
    x2::AbstractArray{T},
)::Int where {T<:Integer}
    s = 0
    @inbounds @simd for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

function rand_level(hnsw::HNSW)::Int
    floor(Int, (-log(rand()) * hnsw.mL) + 1)
end

function _search_layer(
    hnsw::HNSW,
    query::SVector{8,UInt64},
    ep::Int,
    expansion_factor::Int,
    maximum_candidates::Int,
    level::Int;
)::Vector{Int}
    # should be infinite, but we can use a large number
    candidates = MinHeap(maximum_candidates)
    W = MaxHeap(expansion_factor)
    return _search_layer(hnsw, candidates, W, query, ep, level)
end

function _search_layer(
    hnsw::HNSW,
    candidates::MinHeap,
    W::MaxHeap,
    query::SVector{8,UInt64},
    ep::Int,
    level::Int;
)::Vector{Int}
    visited = Set{Int}([ep])
    d = hamming_distance(query, hnsw.data[ep])
    insert!(candidates, d => ep)
    insert!(W, d => ep)


    while length(candidates) > 0
        d_c, c = pop!(candidates)

        # no need to search further
        if d_c > W[1].first
            break
        end
        if !haskey(hnsw.graphs[level], c)
            continue
        end

        for e in hnsw.graphs[level][c]
            if in(e, visited)
                continue
            end
            push!(visited, e)

            d_e = hamming_distance(query, hnsw.data[e])
            if d_e < W[1].first || length(W) < W.k
                # insert! will automatically remove the largest element if the heap is full
                insert!(W, d_e => e)
                insert!(candidates, d_e => e)
            end
        end
    end
    # @info "" level length(visited)
    sort!(W.data, by=x -> x[1])
    # @info "" W.data
    # println(nodes_searched)
    return Int[W.data[i][2] for i = 1:length(W)]
end


function Base.insert!(hnsw::HNSW, q::SVector{8,UInt64}; expansion_factor::Int=100, maximum_candidates::Int=1000)
    candidates = MinHeap(maximum_candidates)
    W = MaxHeap(expansion_factor)
    insert!(hnsw, candidates, W, q)
end

function Base.insert!(hnsw::HNSW, candidates::MinHeap, W::MaxHeap, q::SVector{8,UInt64})
    push!(hnsw.data, q)
    ind = length(hnsw.data)
    l = rand_level(hnsw)
    new_entry_point = l > length(hnsw.graphs)

    if !haskey(hnsw.graphs, l)
        hnsw.graphs[l] = Dict{Int,Vector{Int}}()
        for level = l-1:-1:1
            if !haskey(hnsw.graphs, level)
                hnsw.graphs[level] = Dict{Int,Vector{Int}}()
            end
        end
    end
    if ind == 1
        for level = l:-1:1
            hnsw.graphs[level][ind] = Int[]
        end
        return
    end

    L = length(hnsw.graphs)
    ep = hnsw.enter_point
    W0 = MaxHeap(1)
    for level = L:-1:l+1
        ep = _search_layer(hnsw, candidates, W0, q, ep, level)[1]
        reset!(candidates)
        reset!(W0)
    end

    for level = l:-1:1
        mx = level == 1 ? hnsw.connectivity0 : hnsw.connectivity
        neighbors = _search_layer(hnsw, candidates, W, q, ep, level)
        neighbors = neighbors[1:min(hnsw.connectivity, length(neighbors))]
        hnsw.graphs[level][ind] = neighbors
        for n in neighbors
            if !haskey(hnsw.graphs[level], n)
                hnsw.graphs[level][n] = Int[ind]
            else
                if ind ∉ hnsw.graphs[level][n]
                    push!(hnsw.graphs[level][n], ind)
                    # shrink the neighbors if necessary
                    if length(hnsw.graphs[level][n]) > mx
                        hnsw.graphs[level][n] = sort!(
                            hnsw.graphs[level][n],
                            by=x -> hamming_distance(hnsw.data[x], hnsw.data[n]),
                        )[1:mx]
                    end
                end
            end
        end

        ep = neighbors[1]

        reset!(candidates)
        reset!(W)
    end

    if new_entry_point
        hnsw.enter_point = ep
    end
    return
end

function construct(n::Int; connectivity::Int=16, expansion_factor=100, maximum_candidates::Int=1000)::HNSW
    hnsw = HNSW(connectivity; connectivity0=connectivity * 2)
    candidates = MinHeap(maximum_candidates)
    W = MaxHeap(expansion_factor)
    for _ = 1:n
        q = SVector{8,UInt64}(reinterpret(UInt64, rand(Int8, 64)))
        insert!(hnsw, candidates, W, q)
    end
    # print length of each graph layer
    for lc = 1:length(hnsw.graphs)
        println("Layer $lc: length = $(length(hnsw.graphs[lc]))")
    end
    return hnsw
end

function search(hnsw::HNSW, query::SVector{8,UInt64}, k::Int; expansion_search::Int=30, maximum_candidates=1000)::Vector{Int}
    candidates = MinHeap(maximum_candidates)
    if k > expansion_search
        expansion_search = k
    end
    W = MaxHeap(1)

    L = length(hnsw.graphs)
    ep = hnsw.enter_point
    for level = L:-1:2
        ep = ep
        ep = _search_layer(hnsw, candidates, W, query, ep, level)[1]
        reset!(candidates)
        reset!(W)
    end
    W = MaxHeap(expansion_search)
    result = _search_layer(hnsw, candidates, W, query, ep, 1)
    n = min(k, length(W))
    return result[1:n]
end

function approx_vs_exact(hnsw, query::SVector{8,UInt64}, k::Int, expansion_search::Int, maximum_candidates::Int)
    inds = search(hnsw, query, k; expansion_search=expansion_search, maximum_candidates=maximum_candidates)
    approx_dists = [hamming_distance(query, hnsw.data[i]) for i in inds]
    distances = [hamming_distance(query, hnsw.data[i]) for i in 1:length(hnsw.data)]
    k_nearest_manual = sortperm(distances)[1:k]

    println("HNSW inds ", sort(inds))
    println("Manual inds ", sort(k_nearest_manual))
    println("HNSW distances ", sort(approx_dists))
    println("Manual distances ", sort([hamming_distance(hnsw.data[i], query) for i in k_nearest_manual]))
end

function approx_vs_exact(hnsw::HNSW, k::Int; expansion_search::Int=30, maximum_candidates::Int=1000)
    q = SVector{8,UInt64}(reinterpret(UInt64, rand(Int8, 64)))
    approx_vs_exact(hnsw, q, k, expansion_search, maximum_candidates)
end
