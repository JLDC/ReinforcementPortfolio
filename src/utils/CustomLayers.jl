# Multiple input layer
struct Join{T, F}
    combine::F
    paths::T
end
Join(combine, paths...) = Join(combine, paths)

Flux.@functor Join

(m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
(m::Join)(xs...) = m(xs)

# Multiple output layer
struct Split
    paths::Tuple
end
Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)