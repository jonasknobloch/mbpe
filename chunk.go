package mbpe

import (
	"iter"
	"sync"
	"unicode/utf8"
)

type Chunk struct {
	src    string
	n      int
	bounds []bool
	morphs []bool
	alpha  float64
}

type Change struct {
	delta  float64
	update bool
	remove bool
}

var InvertWeightFunction = false

var pairWeightPool = sync.Pool{
	New: func() any {
		return make(map[Pair]float64)
	},
}

func NewChunk(src string, n int, splits []string, alpha float64) *Chunk {
	bounds := make([]bool, len(src)+1)

	bounds[0] = true

	i := 0

	for _, r := range src {
		i += utf8.RuneLen(r)

		bounds[i] = true
	}

	chunk := &Chunk{
		src:    src,
		n:      n,
		bounds: bounds,
		morphs: nil,
		alpha:  alpha,
	}

	if len(splits) > 1 {
		chunk.Split(splits)
	}

	return chunk
}

func (c *Chunk) Split(segments []string) {
	if len(segments) == 0 {
		panic("empty segments")
	}

	morphs := make([]bool, len(c.src)+1)

	morphs[0] = true

	i := 0

	for _, sub := range segments[:len(segments)-1] {
		if sub != c.src[i:i+len(sub)] {
			panic("unexpected segment")
		}

		i += len(sub)

		morphs[i] = true
	}

	c.morphs = morphs
}

func (c *Chunk) Alpha(alpha float64) {
	c.alpha = alpha
}

func (c *Chunk) NumPairs() int {
	return c.NumTokens() - 1
}

func (c *Chunk) PairBounds() iter.Seq2[int, [3]int] {
	return func(yield func(int, [3]int) bool) {
		start, end := -1, -1

		for i, bounds := range c.TokensBounds() {
			if i == 0 {
				start = bounds[0]
				end = bounds[1]

				continue
			}

			next := bounds[1]

			if !yield(i-1, [3]int{start, end, next}) {
				return
			}

			start = end
			end = next
		}
	}
}

func (c *Chunk) Pairs() iter.Seq2[int, Pair] {
	return func(yield func(int, Pair) bool) {
		for i, bounds := range c.PairBounds() {
			start := bounds[0]
			end := bounds[1]
			next := bounds[2]

			pair := Pair{
				c.src[start:end],
				c.src[end:next],
			}

			if !yield(i, pair) {
				return
			}
		}
	}
}

func (c *Chunk) PairWeights(weights map[Pair]float64) float64 {
	return c.pairWeights(InvertWeightFunction, weights)
}

func (c *Chunk) pairWeights(inverse bool, weights map[Pair]float64) float64 {
	numPairs := c.NumPairs()

	if numPairs == 0 {
		return 0.0
	}

	n := float64(numPairs)
	k := 0.0

	if c.morphs != nil {
		for _, pair := range c.PairBounds() {
			if c.morphs[pair[1]] != inverse {
				k++
			}
		}
	}

	scale := float64(c.n)

	for _, bounds := range c.PairBounds() {
		left := c.src[bounds[0]:bounds[1]]
		right := c.src[bounds[1]:bounds[2]]

		pair := Pair{
			left,
			right,
		}

		var w float64

		if c.morphs != nil && c.morphs[bounds[1]] != inverse {
			w = (1 - c.alpha) + (c.alpha * (k - 1) / n)
		} else {
			w = 1 + (c.alpha * k / n)
		}

		weights[pair] += w * scale
	}

	epsilon := (c.alpha * k / n) * scale // no merge

	return epsilon
}

func (c *Chunk) MergePairIdx(i int) {
	n := c.NumPairs()

	if i < 0 || i > n {
		panic("merge out of bounds")
	}

	for j, pair := range c.PairBounds() {
		if j != i {
			continue
		}

		c.bounds[pair[1]] = false

		return
	}
}

func (c *Chunk) MergePair(left, right string) {
	for _, pair := range c.PairBounds() {
		start := pair[0]
		end := pair[1]
		next := pair[2]

		if c.src[start:end] == left && c.src[end:next] == right {
			c.bounds[end] = false

			c.MergePair(left, right)

			return
		}
	}
}

func (c *Chunk) TrackedMerge(merge Merge, changes map[Pair]Change) float64 {
	before := pairWeightPool.Get().(map[Pair]float64)
	after := pairWeightPool.Get().(map[Pair]float64)

	defer func() {
		clear(before)
		pairWeightPool.Put(before)
	}()

	defer func() {
		clear(after)
		pairWeightPool.Put(after)
	}()

	epsilonBefore := c.PairWeights(before)

	c.MergePair(merge.pair[0], merge.pair[1])

	epsilonAfter := c.PairWeights(after)

	for pair, weightBefore := range before {
		if weightAfter, ok := after[pair]; ok {
			if weightBefore == weightAfter {
				continue
			}

			changes[pair] = Change{
				delta:  weightAfter - weightBefore,
				update: true,
				remove: false,
			}
		} else {
			changes[pair] = Change{
				delta:  -weightBefore,
				update: false,
				remove: true,
			}
		}
	}

	for pair, weightAfter := range after {
		if _, ok := before[pair]; !ok {
			changes[pair] = Change{
				delta:  weightAfter,
				update: false,
				remove: false,
			}
		}
	}

	return epsilonAfter - epsilonBefore
}

func (c *Chunk) NumTokens() int {
	n := 0

	for _, b := range c.bounds {
		if b {
			n++
		}
	}

	return n - 1
}

func (c *Chunk) TokensBounds() iter.Seq2[int, [2]int] {
	i := 0

	return func(yield func(int, [2]int) bool) {
		start := -1

		for end, b := range c.bounds {
			if !b {
				continue
			}

			if start == -1 {
				start = end

				continue
			}

			if !yield(i, [2]int{start, end}) {
				return
			}

			start = end

			i++
		}
	}
}

func (c *Chunk) Tokens() iter.Seq2[int, string] {
	return func(yield func(int, string) bool) {
		for i, bounds := range c.TokensBounds() {
			start := bounds[0]
			end := bounds[1]

			token := c.src[start:end]

			if !yield(i, token) {
				return
			}
		}
	}
}
