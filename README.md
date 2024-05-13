# tiny binary rag

I wanted to experiment in how fast I could do exact RAG lookups with a binary vector space.

Why binary?

It turns out the performance is [very similar to a full 32-bit vector](https://huggingface.co/blog/embedding-quantization). But we save a lot in terms of server costs, and it makes in-memory retrieval more feasible

A database that was once 1TB is now ~32GB, which can easily fit in RAM on a much cheaper setup.

Assuming each row is a 1024-dim float32 vector, that's 4096 bytes per row. or ~244 million rows.

For some added context Wikipedia, which is arguably the largest publicly available data to RAG might only be around 3.4 million rows.

https://en.wikipedia.org/wiki/Wikipedia:Size_of_Wikipedia

- 6,818,523 articles
- average 668 words per article (4.5B billion words)
- average English word is 5 characters, English text has a ratio of 4-5 characters per token
- let’s just say each word is composed of 3 tokens on average, so this is 13.5B tokens
- if we use the mixedbread model we can represent 512 tokens in a vector, assuming overlap let’s say 400 tokens new tokens in each vector with 112 being used for overlap

~3,400,000 vectors

Chances are internal RAGs are going to be significantly smaller than this.

And so this opens up the question:

"At what point is the dataset too big that exact brute search over binary vectors is not entirely feasible?"

If our dataset is smaller than this, then we can just do the dumb thing in a very small amount of code and be completely fine.

We might not need a fancy vector DB or the newest HSNW library (HSNW is approximate search anyway, we're doing exact).

I'm doing this in Julia because:

1. I like it.
2. It should be able to get comparable performance for this relative to C, C++ or Rust. So we're not missing out on potential gains.

```julia
julia> versioninfo()
Julia Version 1.11.0-beta1
Commit 08e1fc0abb9 (2024-04-10 08:40 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: macOS (arm64-apple-darwin22.4.0)
  CPU: 8 × Apple M1
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, apple-m1)
Threads: 4 default, 0 interactive, 2 GC (on 4 virtual cores)
Environment:
  JULIA_STACKTRACE_MINIMAL = true
  DYLD_LIBRARY_PATH = /Users/lunaticd/.wasmedge/lib
  JULIA_EDITOR = nvim
```

Ok, let's begin!

Each vector is going to be n bits, where n in the size of the embedding. This will vary depending on which embedder you use, there are several proprietary and open source variations. I'll be assuming we're using the mixedbread model:

- mixedbread-ai/mxbai-embed-large-v1

This model returns a 1024 element float32 vector. Binarization turns this into a 1024 bit vector, represented as a 128 element int8 vector, or 128 bytes. Furthermore, we can reduce this to [64 bytes](https://www.mixedbread.ai/blog/binary-mrl) and so that will be our final representation.

A 512 bit vector represented as 64 int8 or uint8 elements.

```julia
julia> x = rand(Int8, 64)
64-element Vector{Int8}:
   26
  123
   70
   -2
    ⋮
```

Keep in mind the representation is a byte element but we're operating at the bit level.

Comparing two bit vectors will require using the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance#). Luckily the definition is very straightforward, each time a bit in the vectors doesn't match add 1 to the distance. So two bit vectors of length 512 that are entirely different will have a hamming distance of 512, and if they are the same a distance of 0.

An easy to do this is to turn the integer representation into a bitstring and then compute the distance on that.

```julia
julia> bitstring(Int8(125))
"01111101"

julia> bitstring(Int8(33))
"00100001"

julia> function hamming_distance(s1::AbstractString, s2::AbstractString)::Int
           s = 0
           for (c1, c2) in zip(s1, s2)
               if c1 != c2
                   s += 1
               end
           end
           s
       end

julia> hamming_distance(bitstring(Int8(125)), bitstring(Int8(33))) # should be 4
4
```

Julia has a lovely benchmarking package called `Chairmarks`, which is really easy to use. Let's see how fast this implementation is.

```julia

julia> using Chairmarks

julia> @be hamming_distance(bitstring(Int8(125)), bitstring(Int8(33)))
Benchmark: 6053 samples with 317 evaluations
min    36.672 ns (4 allocs: 128 bytes)
median 38.514 ns (4 allocs: 128 bytes)
mean   46.525 ns (4 allocs: 128 bytes, 0.13% gc time)
max    8.737 μs (4 allocs: 128 bytes, 98.30% gc time)
```

This is honestly pretty good. For comparison let's do a simple addition:

```julia
julia> @be 125 + 33
Benchmark: 7274 samples with 8775 evaluations
min    1.239 ns
median 1.339 ns
mean   1.453 ns
max    6.990 ns
```

We're not going to get lower than this for an operation. The nanosecond is basically the atomic measurement unit for time when working with CPUs.

However, this does not mean each operation has to take ~1ns. There are various optimizations that hardware can do such that [several operations are done each clock cycle](https://ppc.cs.aalto.fi/ch1/#:~:text=Hence%2C%20a%20modern%204%2Dcore,billion%20clock%20cycles%20per%20second). Each core of a CPU has a number of arithmetic units which are used to process elementary operations (addition, subtraction, xor, or, and, etc), the arithmetic units are fed by what is known as pipelining. The point is there is more than 1 arithmetic unit, so we can conceivably do several additons or bit operations in the same clock cycle as long as they do not depend on eachother. Let's rewrite `hamming_distance` using only elementary operations so that we might take advantage of this.

Let's consider comparing two bits:

* 0, 0 -> 0
* 1, 0 -> 1
* 0, 1 -> 1
* 1, 1 -> 0

This is the XOR operation. We want to XOR all the bits and keep track of which ones are 1, and return the sum.
In order to do this over all the bits we can shift the integer by the number of bits, for an 8-bit number we would perform the shift 7 times.

Here's what this looks like:

```julia
BYTE = Union{Int8,UInt8}

@inline function hamming_distance(x1::T, x2::T)::Int where {T<:BYTE}
    r = x1 ⊻ x2 # xor
    # how many xored bits are 1
    c = 0
    for i in 0:7
        c += (r >> i) & 1
    end
    return Int(c)
end

# benchmark again
julia> @be hamming_distance(Int8(33), Int8(125))
Benchmark: 4797 samples with 8966 evaluations
min    2.175 ns
median 2.189 ns
mean   2.205 ns
max    5.279 ns
```

This is huge improvement, almost as fast as doing an addition operation!

Now we need to do this for an vector:

```julia
@inline function hamming_distance1(x1::AbstractArray{T}, x2::AbstractArray{T})::Int where {T<:BYTE}
    s = 0
    for i in 1:length(x1)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end
```

We expect this to take around 128ns (~2 * 64).

```
julia> @be hamming_distance1(q1, q2)
Benchmark: 4469 samples with 256 evaluations
min    69.500 ns
median 75.035 ns
mean   80.240 ns
max    189.941 ns
```

There are a few more things we can do do further optimize this loop, the compiler sometimes adds these automatically but we can manually annotate this be certain.

```julia
@inline function hamming_distance(x1::AbstractArray{T}, x2::AbstractArray{T})::Int where {T<:BYTE}
    s = 0
    @inbounds @simd for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end
```

- `@inbounds` removes any boundary checking.
- `@simd` adds tells the compiler to use SIMD instructions if possible.

```julia
julia> @be hamming_distance(q1, q2)
Benchmark: 4441 samples with 323 evaluations
min    52.759 ns
median 56.245 ns
mean   61.721 ns
max    163.827 ns
```

That's a decent improvement. Sometimes this benchmark is 20-30ns on a fresh REPL session. I think it depends current background usage and how well SIMD can be utilized at that time.


```julia
julia> q1 = rand(Int8, 64);

julia> q2 = rand(Int8, 64);

julia> hamming_distance(q1, q2);

julia> @be hamming_distance(q1, q2)
Benchmark: 3546 samples with 1000 evaluations
min    23.333 ns
median 23.459 ns
mean   24.461 ns
max    59.542 ns

julia> @be hamming_distance(q1, q2)
Benchmark: 4653 samples with 749 evaluations
min    23.308 ns
median 25.088 ns
mean   26.595 ns
max    62.473 ns
```

Still, it's good to know in a perfect environment, such as an isolated machine, you could make really good use of the hardware.

Let's say though each comparison takes 50ns, then we can search over 20M rows in a second, if we utilized 4 cores then 80M rows.

1M rows should take 12ms over 4 cores. And Wikipedia at 3.4M ~42.5ms.

In practice it turns out it's faster than this:

```julia
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
```

1M rows benchmark:

```julia

julia> X1 = [rand(Int8, 64) for _ in 1:(10^6)];

julia> k_closest(X1, q1, 1)
1-element Vector{Pair{Int64, Int64}}:
 204 => 144609

julia> @be k_closest(X1, q1, 1)
Benchmark: 7 samples with 1 evaluation
min    14.025 ms (2 allocs: 80 bytes)
median 14.538 ms (2 allocs: 80 bytes)
mean   14.828 ms (2 allocs: 80 bytes)
max    16.799 ms (2 allocs: 80 bytes)

julia> k_closest_parallel(X1, q1, 1)
1-element Vector{Pair{Int64, Int64}}:
 204 => 144609

julia> @be k_closest_parallel(X1, q1, 1)
Benchmark: 23 samples with 1 evaluation
min    4.024 ms (50 allocs: 3.297 KiB)
median 4.082 ms (50 allocs: 3.297 KiB)
mean   4.361 ms (50 allocs: 3.297 KiB)
max    5.338 ms (50 allocs: 3.297 KiB)
```

It seems each vector comparison is more like ~15ns rather than ~50ns.

We can actually do even better though by using `StaticArrays`:

```julia
using StaticArrays

julia> q1 = SVector{64, Int8}(rand(Int8, 64));

julia> X1 = [SVector{64, Int8}(rand(Int8, 64)) for _ in 1:(10^6)];

julia> @be k_closest(X1, q1, 1)
Benchmark: 9 samples with 1 evaluation
min    10.426 ms (2 allocs: 80 bytes)
median 10.569 ms (2 allocs: 80 bytes)
mean   11.249 ms (2 allocs: 80 bytes)
max    14.098 ms (2 allocs: 80 bytes)

julia> @be k_closest_parallel(X1, q1, 1)
Benchmark: 31 samples with 1 evaluation
min    2.815 ms (50 allocs: 3.547 KiB)
median 2.855 ms (50 allocs: 3.547 KiB)
mean   3.120 ms (50 allocs: 3.547 KiB)
max    5.084 ms (50 allocs: 3.547 KiB)
```

Now each vector comparison is taking ~11ns!

However, we don't want just closest vector but the "k" closest ones, so let's set "k" to something more realistic:

```
julia> @be k_closest(X1, q1, 20)
Benchmark: 6 samples with 1 evaluation
min    16.300 ms (2 allocs: 400 bytes)
median 16.404 ms (2 allocs: 400 bytes)
mean   16.879 ms (2 allocs: 400 bytes)
max    18.736 ms (2 allocs: 400 bytes)

julia> @be k_closest_parallel(X1, q1, 20)
Benchmark: 21 samples with 1 evaluation
min    4.393 ms (52 allocs: 7.703 KiB)
median 4.460 ms (52 allocs: 7.703 KiB)
mean   4.795 ms (52 allocs: 7.703 KiB)
max    6.494 ms (52 allocs: 7.703 KiB)

julia> @be k_closest(X1, q1, 100)
Benchmark: 2 samples with 1 evaluation
       51.933 ms (2 allocs: 1.625 KiB)
       52.034 ms (2 allocs: 1.625 KiB)

julia> @be k_closest_parallel(X1, q1, 100)
Benchmark: 9 samples with 1 evaluation
min    11.622 ms (54 allocs: 23.781 KiB)
median 11.685 ms (54 allocs: 23.781 KiB)
mean   12.005 ms (54 allocs: 23.781 KiB)
max    13.952 ms (54 allocs: 23.781 KiB)
```

A parallel search over 4 cores is still ~12ms while returning the 100 closest vectors.

Just for fun let's search over 3.4M rows (Wikipedia)

```julia
julia> X1 = [SVector{64, Int8}(rand(Int8, 64)) for _ in 1:(3.4 * 10^6)];

julia> @be k_closest_parallel(X1, q1, 10)
Benchmark: 8 samples with 1 evaluation
min    12.012 ms (52 allocs: 5.500 KiB)
median 12.115 ms (52 allocs: 5.500 KiB)
mean   12.608 ms (52 allocs: 5.500 KiB)
max    14.360 ms (52 allocs: 5.500 KiB)

julia> @be k_closest_parallel(X1, q1, 20)
Benchmark: 7 samples with 1 evaluation
min    14.932 ms (52 allocs: 7.703 KiB)
median 15.068 ms (52 allocs: 7.703 KiB)
mean   15.454 ms (52 allocs: 7.703 KiB)
max    17.015 ms (52 allocs: 7.703 KiB)

julia> @be k_closest_parallel(X1, q1, 30)
Benchmark: 6 samples with 1 evaluation
min    17.996 ms (52 allocs: 9.875 KiB)
median 18.028 ms (52 allocs: 9.875 KiB)
mean   18.402 ms (52 allocs: 9.875 KiB)
max    19.497 ms (52 allocs: 9.875 KiB)
```

And over 100,000 rows:

```julia
julia> X1 = [SVector{64, Int8}(rand(Int8, 64)) for _ in 1:(10^5)];

julia> @be k_closest_parallel(X1, q1, 30)
Benchmark: 154 samples with 1 evaluation
min    542.916 μs (52 allocs: 9.875 KiB)
median 554.500 μs (52 allocs: 9.875 KiB)
mean   594.811 μs (52 allocs: 9.875 KiB)
max    968.583 μs (52 allocs: 9.875 KiB)
```

Under 1ms!

