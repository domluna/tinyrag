mutable struct MaxHeap
    const data::Vector{Pair{Int,Int}}
    current_idx::Int # add pairs until current_idx > length(data)
    const k::Int

    function MaxHeap(k::Int)
        new(fill((typemax(Int) => -1), k), 1, k)
    end
end

function Base.length(h::MaxHeap)
    return min(h.k, h.current_idx - 1)
end

function Base.getindex(h::MaxHeap, inds)
    h.data[inds...]
end

function reset!(heap::MaxHeap)
    heap.current_idx = 1
end

function Base.insert!(heap::MaxHeap, value::Pair{Int,Int})
    if heap.current_idx <= heap.k
        heap.data[heap.current_idx] = value
        heap.current_idx += 1
        heapify_up!(heap, heap.current_idx - 1) # Heapify up from the inserted position
    elseif value.first < heap.data[1].first
        heap.data[1] = value
        heapify_down!(heap, 1) # Heapify down from the root
    end
end

function heapify_up!(heap::MaxHeap, i::Int)
    parent = div(i, 2)
    while parent >= 1 && heap.data[i].first > heap.data[parent].first
        heap.data[i], heap.data[parent] = heap.data[parent], heap.data[i]
        i = parent
        parent = div(i, 2)
    end
end

function heapify_down!(heap::MaxHeap, i::Int)
    left = 2 * i
    right = 2 * i + 1
    largest = i

    if left <= heap.current_idx - 1 && heap.data[left].first > heap.data[largest].first
        largest = left
    end

    if right <= heap.current_idx - 1 && heap.data[right].first > heap.data[largest].first
        largest = right
    end

    if largest != i
        heap.data[i], heap.data[largest] = heap.data[largest], heap.data[i]
        heapify_down!(heap, largest)
    end
end

mutable struct MinHeap
    const data::Vector{Pair{Int,Int}}
    current_idx::Int # add pairs until current_idx > length(data)
    const k::Int

    function MinHeap(k::Int)
        new(fill((typemin(Int) => -1), k), 1, k)
    end
end

function Base.length(h::MinHeap)
    return min(h.k, h.current_idx - 1)
end

function Base.getindex(h::MinHeap, inds)
    h.data[inds...]
end

function reset!(heap::MinHeap)
    heap.current_idx = 1
end

function Base.insert!(heap::MinHeap, value::Pair{Int,Int})
    if heap.current_idx <= heap.k
        heap.data[heap.current_idx] = value
        heap.current_idx += 1
        heapify_up!(heap, heap.current_idx - 1) # Heapify up from the inserted position
    elseif value.first > heap.data[1].first
        heap.data[1] = value
        heapify_down!(heap, 1) # Heapify down from the root
    end
end

function heapify_up!(heap::MinHeap, i::Int)
    parent = div(i, 2)
    while parent >= 1 && heap.data[i].first < heap.data[parent].first
        heap.data[i], heap.data[parent] = heap.data[parent], heap.data[i]
        i = parent
        parent = div(i, 2)
    end
end

function heapify_down!(heap::MinHeap, i::Int)
    left = 2 * i
    right = 2 * i + 1
    smallest = i

    if left <= heap.current_idx - 1 && heap.data[left].first < heap.data[smallest].first
        smallest = left
    end

    if right <= heap.current_idx - 1 && heap.data[right].first < heap.data[smallest].first
        smallest = right
    end

    if smallest != i
        heap.data[i], heap.data[smallest] = heap.data[smallest], heap.data[i]
        heapify_down!(heap, smallest)
    end
end

function pop!(heap::MinHeap)::Pair{Int,Int}
    if heap.current_idx <= 1
        throw(ArgumentError("min-heap is empty"))
    end

    # Store the minimum value (originally at the root)
    min_value = heap.data[1]

    # Swap the root with the last element
    heap.data[1], heap.data[heap.current_idx-1] =
        heap.data[heap.current_idx-1], heap.data[1]
    heap.current_idx -= 1

    # Heapify down from the root
    heapify_down!(heap, 1)

    return min_value # Return the actual minimum value
end
