using Base.Threads

@inline function hamming_distance(x1::T, x2::T)::Int where {T<:Union{Int8,UInt8}}
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
)::Int where {T<:Union{Int8,UInt8}}
    s = 0
    @inbounds @simd for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

@inline function hamming_distance1(
    x1::AbstractArray{T},
    x2::AbstractArray{T},
)::Int where {T<:Union{Int8,UInt8}}
    s = 0
    for i = 1:length(x1)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

mutable struct MaxHeap
    const data::Vector{Pair{Int,Int}}
    current_idx::Int # add pairs until current_idx > length(data)
    const k::Int

    function MaxHeap(k::Int)
        new(fill((typemax(Int) => -1), k), 1, k)
    end
end

function insert!(heap::MaxHeap, value::Pair{Int,Int})
    if heap.current_idx <= heap.k
        heap.data[heap.current_idx] = value
        heap.current_idx += 1
        if heap.current_idx > heap.k
            makeheap!(heap)
        end
    elseif value.first < heap.data[1].first
        heap.data[1] = value
        heapify!(heap, 1)
    end
end

function makeheap!(heap::MaxHeap)
    for i = div(heap.k, 2):-1:1
        heapify!(heap, i)
    end
end

function heapify!(heap::MaxHeap, i::Int)
    left = 2 * i
    right = 2 * i + 1
    largest = i

    if left <= length(heap.data) && heap.data[left].first > heap.data[largest].first
        largest = left
    end

    if right <= length(heap.data) && heap.data[right].first > heap.data[largest].first
        largest = right
    end

    if largest != i
        heap.data[i], heap.data[largest] = heap.data[largest], heap.data[i]
        heapify!(heap, largest)
    end
end

function k_closest(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int;
    startind::Int = 1,
) where {T<:Union{Int8,UInt8},V<:AbstractVector{T}}
    heap = MaxHeap(k)
    @inbounds for i in eachindex(db)
        d = hamming_distance(db[i], query)
        insert!(heap, d => startind + i - 1)
    end
    return sort!(heap.data, by = x -> x[1])
end

function k_closest_parallel(
    db::AbstractArray{V},
    query::AbstractVector{T},
    k::Int,
) where {T<:Union{Int8,UInt8},V<:AbstractVector{T}}
    n = length(db)
    t = nthreads()
    task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
    tasks = map(task_ranges) do r
        Threads.@spawn k_closest(view(db, r), query, k; startind = r[1])
    end
    results = fetch.(tasks)
    sort!(vcat(results...), by = x -> x[1])[1:k]
end
