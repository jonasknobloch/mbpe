package mbpe

import "sync"

var pairWeightPool = sync.Pool{
	New: func() any {
		return make(map[Pair]float64)
	},
}

var changePool = sync.Pool{
	New: func() any {
		return make(map[Pair]Change)
	},
}

func GetChanges() map[Pair]Change {
	return changePool.Get().(map[Pair]Change)
}

func ReleaseChanges(m map[Pair]Change) {
	clear(m)
	changePool.Put(m)
}

type Chunk struct {
	src     string
	n       int
	bounds  []int
	morphs  []int
	clashes []bool
	alpha   float64
}

type Change struct {
	delta  float64
	update bool
	remove bool
}

var InvertWeightFunction = false

func NewChunk(src string, n int, splits []string, alpha float64) *Chunk {
	bounds := []int{0}

	for _, r := range src {
		j := bounds[len(bounds)-1] + len(string(r))

		bounds = append(bounds, j)
	}

	var morphs []int

	if len(splits) > 1 {
		morphs = make([]int, 0)

		i := 0

		for _, sub := range splits[:len(splits)-1] {
			i += len(sub)

			morphs = append(morphs, i)
		}
	}

	c := &Chunk{
		src:    src,
		n:      n,
		bounds: bounds,
		morphs: morphs,
		alpha:  alpha,
	}

	c.clashes = c.computeClashes()

	return c
}

func (c *Chunk) computeClashes() []bool {
	n := len(c.bounds) - 2

	if n <= 0 {
		return nil
	}

	clashes := make([]bool, n)

	for i := range clashes {
		lower := c.bounds[i]
		upper := c.bounds[i+2]

		for _, b := range c.morphs {
			if b > lower && b < upper {
				clashes[i] = true
				break
			}
		}
	}

	return clashes
}

func (c *Chunk) Split(segments []string) {
	var morphs []int

	if len(segments) > 1 {
		morphs = make([]int, 0)

		i := 0

		for _, sub := range segments[:len(segments)-1] {
			i += len(sub)

			morphs = append(morphs, i)
		}
	}

	c.morphs = morphs
	c.clashes = c.computeClashes()
}

func (c *Chunk) Alpha(alpha float64) {
	c.alpha = alpha
}

func (c *Chunk) Pairs() []Pair {
	pairs := make([]Pair, len(c.bounds)-2)

	for i := 0; i < len(c.bounds)-2; i++ {
		pairs[i] = Pair{
			c.src[c.bounds[i]:c.bounds[i+1]],
			c.src[c.bounds[i+1]:c.bounds[i+2]],
		}
	}

	return pairs
}

func (c *Chunk) WeightedPairs() ([]Pair, []float64, float64) {
	return c.weightedPairs(InvertWeightFunction)
}

func (c *Chunk) weightedPairs(inverse bool) ([]Pair, []float64, float64) {
	pairs := c.Pairs()

	if len(pairs) == 0 {
		return pairs, []float64{}, 0.0
	}

	weights, epsilon := c.pairWeights(pairs, c.clashes, inverse)

	for i := range weights {
		weights[i] *= float64(c.n)
	}

	epsilon *= float64(c.n)

	return pairs, weights, epsilon
}

func (c *Chunk) pairWeights(pairs []Pair, clashes []bool, inverse bool) ([]float64, float64) {
	weights := make([]float64, len(pairs))

	n := float64(len(weights))
	k := 0.0

	for _, v := range clashes {
		if v != inverse {
			k++
		}
	}

	for i := range pairs {
		var w float64

		if clashes[i] != inverse {
			w = (1 - c.alpha) + (c.alpha * (k - 1) / n)
		} else {
			w = 1 + (c.alpha * k / n)
		}

		weights[i] = w
	}

	epsilon := c.alpha * k / n // no merge

	return weights, epsilon
}

// weightsInto computes weighted pair counts directly into dst, avoiding []Pair
// and []float64 allocations. It relies on the cached c.clashes field.
func (c *Chunk) weightsInto(dst map[Pair]float64) float64 {
	if len(c.clashes) == 0 {
		return 0.0
	}

	inverse := InvertWeightFunction
	n := float64(len(c.clashes))
	k := 0.0

	for _, v := range c.clashes {
		if v != inverse {
			k++
		}
	}

	for i, clash := range c.clashes {
		pair := Pair{
			c.src[c.bounds[i]:c.bounds[i+1]],
			c.src[c.bounds[i+1]:c.bounds[i+2]],
		}

		var w float64

		if clash != inverse {
			w = (1 - c.alpha) + (c.alpha * (k - 1) / n)
		} else {
			w = 1 + (c.alpha * k / n)
		}

		dst[pair] += w * float64(c.n)
	}

	return c.alpha * k / n * float64(c.n)
}

func (c *Chunk) MergePairIdx(i int) {
	if i > len(c.bounds)-2 {
		panic("merge out of bounds")
	}

	if len(c.clashes) > 0 {
		// Pair at i-1 gains a wider right extent: (bounds[i-1], bounds[i+1]) → (bounds[i-1], bounds[i+2]).
		// It can only gain clashes, never lose them. Check the newly covered range [bounds[i+1], bounds[i+2]).
		if i > 0 && !c.clashes[i-1] {
			removed := c.bounds[i+1]
			upper := c.bounds[i+2]

			for _, b := range c.morphs {
				if b >= removed && b < upper {
					c.clashes[i-1] = true
					break
				}
			}
		}

		if i+1 < len(c.clashes) {
			// New pair at i: (merged_token, next_token). Its clash span is (bounds[i], bounds[i+3_old]),
			// covered by old clashes[i] || old clashes[i+1].
			c.clashes[i] = c.clashes[i] || c.clashes[i+1]
			c.clashes = append(c.clashes[:i+1], c.clashes[i+2:]...)
		} else {
			// Merging the last pair — the merged token becomes the last token, no new pair at i.
			c.clashes = c.clashes[:i]
		}
	}

	c.bounds = append(c.bounds[:i+1], c.bounds[i+2:]...)
}

func (c *Chunk) MergePair(left, right string) {
	for i := 0; i < len(c.bounds)-2; i++ {
		l := c.src[c.bounds[i]:c.bounds[i+1]]
		r := c.src[c.bounds[i+1]:c.bounds[i+2]]

		if l == left && r == right {
			c.MergePairIdx(i)
			c.MergePair(left, right)

			return
		}
	}
}

func (c *Chunk) TrackedMerge(merge Merge, changes map[Pair]Change) float64 {
	before := pairWeightPool.Get().(map[Pair]float64)
	epsilonBefore := c.weightsInto(before)

	c.MergePair(merge.pair[0], merge.pair[1])

	after := pairWeightPool.Get().(map[Pair]float64)
	epsilonAfter := c.weightsInto(after)

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

	clear(before)
	pairWeightPool.Put(before)
	clear(after)
	pairWeightPool.Put(after)

	return epsilonAfter - epsilonBefore
}

func (c *Chunk) Tokens() []string {
	r := make([]string, 0, len(c.bounds)-1)

	for i := 0; i < len(c.bounds)-1; i++ {
		r = append(r, c.src[c.bounds[i]:c.bounds[i+1]])
	}

	return r
}

// Deprecated: Use Inverter segmenter instead. Disable MergePrefixWhiteSpace
// to maintain behavior of Invert during training.
func (c *Chunk) Invert() {
	n := len(c.bounds) - len(c.morphs) - 2

	r := make([]int, 0, n)

	for _, b := range c.bounds[1 : len(c.bounds)-1] {
		found := false

		for _, m := range c.morphs {
			if b == m {
				found = true

				break
			}
		}

		if !found {
			r = append(r, b)
		}
	}

	if len(r) != n {
		panic("unexpected number of morphs")
	}

	c.morphs = r
}
