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
        makeheap!(heap, heap.current_idx - 1) # Heapify up from the inserted position
    elseif value.first < heap.data[1].first
        heap.data[1] = value
        heapify!(heap, 1) # Heapify down from the root
    end
end

function makeheap!(heap::MaxHeap, i::Int)
    for j = div(i, 2):-1:1
        heapify!(heap, j)
    end
end

function heapify!(heap::MaxHeap, i::Int)
    left = 2 * i
    right = 2 * i + 1
    largest = i

    if left <= length(heap) && heap.data[left].first > heap.data[largest].first
        largest = left
    end

    if right <= length(heap) && heap.data[right].first > heap.data[largest].first
        largest = right
    end

    if largest != i
        heap.data[i], heap.data[largest] = heap.data[largest], heap.data[i]
        heapify!(heap, largest)
    end
end

mutable struct MinHeap
    const data::Vector{Pair{Int,Int}}
    current_idx::Int # add pairs until current_idx > length(data)
    const k::Int

    function MinHeap(k::Int)
        new(fill((typemax(Int) => -1), k), 1, k)
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
        makeheap!(heap, heap.current_idx - 1) # Heapify up from the inserted position
    else
        ind = findmax(heap)
        if heap[ind].first < value.first
            return
        end
        heap.data[1], heap.data[ind] = heap.data[ind], heap.data[1]
        pop!(heap)
        insert!(heap, value)
    end
end

function makeheap!(heap::MinHeap, i::Int)
    for j = div(i, 2):-1:1
        heapify!(heap, j)
    end
end

function heapify!(heap::MinHeap, i::Int)
    left = 2 * i
    right = 2 * i + 1
    smallest = i

    if left <= length(heap) && heap.data[left].first < heap.data[smallest].first
        smallest = left
    end
    if right <= length(heap) && heap.data[right].first < heap.data[smallest].first
        smallest = right
    end

    if smallest != i
        heap.data[i], heap.data[smallest] = heap.data[smallest], heap.data[i]
        heapify!(heap, smallest)
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
    heapify!(heap, 1)

    return min_value # Return the actual minimum value
end


function Base.iterate(heap::MinHeap, state=1)
    nodes = heap.data::Vector{Pair{Int,Int}}
    if state > length(heap)
        return nothing
    end
    return nodes[state], state + 1
end

function findmax(heap::MinHeap)::Int
    if length(heap) == 0
        throw(ArgumentError("min-heap is empty"))
    end
    if length(heap) == 1
        return 1
    end

    n = length(heap)
    ind = div(n, 2)
    maxel = heap[ind]

    for i in ind+1:n
        if heap[i].first > maxel.first
            maxel = heap[i]
            ind = i
        end
    end
    return ind
end
