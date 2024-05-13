using Base.Threads

BYTE = Union{Int8,UInt8}

@inline function hamming_distance(x1::T, x2::T)::Int where {T<:BYTE}
    r = x1 ⊻ x2
    c = 0
    for i = 0:7
        c += (r >> i) & 1
    end
    return Int(c)
end

@inline function hamming_distance(s1::AbstractString, s2::AbstractString)::Int
    s = 0
    for (c1, c2) in zip(s1, s2)
        if c1 != c2
            s += 1
        end
    end
    s
end

@inline function hamming_distance(
    x1::AbstractArray{T},
    x2::AbstractArray{T},
)::Int where {T<:BYTE}
    s = 0
    @inbounds @simd for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

@inline function hamming_distance1(
    x1::AbstractArray{T},
    x2::AbstractArray{T},
)::Int where {T<:BYTE}
    s = 0
    for i = 1:length(x1)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

function k_closest(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int,
) where {T<:BYTE,V<:AbstractVector{T}}
    results = Vector{Pair{Int,Int}}(undef, k)
    m = typemax(Int)
    fill!(results, (m => -1))

    @inbounds for i in eachindex(db)
        d = hamming_distance(db[i], query)
        for j = 1:k
            if d < results[j][1]
                old = results[j]
                results[j] = d => i
                for l = j+1:k-1
                    old, results[l] = results[l], old
                end
                break
            end
        end
    end

    return results
end

function k_closest_parallel(
    db::AbstractArray{V},
    query::AbstractVector{T},
    k::Int,
) where {T<:BYTE,V<:AbstractVector{T}}
    n = length(db)
    t = nthreads()
    task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
    tasks = map(task_ranges) do r
        Threads.@spawn k_closest(view(db, r), query, k)
    end
    results = fetch.(tasks)
    sort!(vcat(results...), by = x -> x[1])[1:k]
end


# usearch 1_000_000
# In [54]: %timeit binary_index.search(q, top_k)
# 46.5 µs ± 405 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
#
# With exact=True in usearch, the Julia implementation is faster than usearch.
#
# Extra
