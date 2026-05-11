package morfessor

import (
	"fmt"
	"math"
	"os"
	"unicode/utf8"

	pb "github.com/jonasknobloch/mbpe/morfessor/proto"
	"google.golang.org/protobuf/proto"
)

type Model struct {
	model *pb.BaselineModel
}

func NewModel() *Model {
	return &Model{}
}

func (m *Model) LoadModel(name string) error {
	model, err := decodeModel(name)

	if err != nil {
		return err
	}

	m.model = model

	return nil
}

func (m *Model) Segment(compound string) ([]string, float64) {
	if compound == "" {
		return []string{""}, 0.0
	}

	return viterbiSegment(m.model, compound, 0.0, 30)
}

func decodeModel(name string) (*pb.BaselineModel, error) {
	data, err := os.ReadFile(name)

	if err != nil {
		return nil, err
	}

	var model pb.BaselineModel

	if err := proto.Unmarshal(data, &model); err != nil {
		return nil, err
	}

	return &model, nil
}

func unicodeScalarBounds(message string) []int {
	bounds := make([]int, 0, len(message))

	i := 0

	for _, r := range message {
		i += utf8.RuneLen(r)

		bounds = append(bounds, i)
	}

	return bounds
}

func getCodeLength(lexiconCoding *pb.LexiconEncoding, construction string) float64 {
	l := float64(utf8.RuneCountInString(construction)) + 1.0

	cost := l * math.Log(float64(lexiconCoding.Tokens)+l)

	cost -= math.Log(float64(lexiconCoding.Boundaries) + 1.0)

	for _, atom := range construction {
		count, exists := lexiconCoding.Atoms.Counts[string(atom)] // Lookup atom

		if !exists {
			count = 1
		}

		cost -= math.Log(float64(count))
	}

	return cost
}

func viterbiSegment(model *pb.BaselineModel, compound string, addCount float64, maxLen int) ([]string, float64) {
	bounds := unicodeScalarBounds(compound)

	compoundLength := len(bounds)

	type cell struct {
		cost float64
		prev int
	}

	grid := make([]cell, len(compound)+1)
	grid[0].prev = -1

	corpusTokens := float64(model.XCorpusCoding.Tokens)
	corpusBoundaries := float64(model.XCorpusCoding.Boundaries)

	logTokens := 0.0

	if corpusTokens+corpusBoundaries+addCount > 0 {
		logTokens = math.Log(corpusTokens + corpusBoundaries + addCount)
	}

	badLikelihood := float64(compoundLength)*logTokens + 1.0

	for i := 0; i < compoundLength; i++ {
		t := bounds[i]

		bestPrev := -1
		bestCost := 0.0

		startJ := 0
		if i+1 > maxLen {
			startJ = i + 1 - maxLen
		}

		// j is the rune index of the construction start; rune count = i - j + 1 <= maxLen by construction.
		for j := startJ; j <= i; j++ {
			pt := 0
			if j > 0 {
				pt = bounds[j-1]
			}

			cost := grid[pt].cost
			construction := compound[pt:t]

			if analysis, ok := model.XAnalyses[construction]; ok {
				if len(analysis.Splitloc) == 0 || analysis.Splitloc[0] == 0 {
					if analysis.Count <= 0 {
						panic(fmt.Sprintf("Construction count of '%s' is %d", construction, analysis.Count))
					}

					c := cost + logTokens - math.Log(float64(analysis.Count)+addCount)

					if bestPrev == -1 || c < bestCost {
						bestPrev = pt
						bestCost = c
					}

					continue
				}
			}

			if addCount == 0 {
				if i == j { // single rune: rune count = i - j + 1 = 1
					c := cost + badLikelihood

					if bestPrev == -1 || c < bestCost {
						bestPrev = pt
						bestCost = c
					}
				}

				continue
			}

			if addCount > 0 {
				lexiconCoding := model.XLexiconCoding
				corpusCoding := model.XCorpusCoding

				lexiconBoundaries := float64(lexiconCoding.Boundaries)
				corpusWeight := float64(corpusCoding.Weight)

				var c float64

				if corpusCoding.Tokens == 0 {
					c = cost + addCount*math.Log(addCount) + getCodeLength(lexiconCoding, construction)/corpusWeight
				} else {
					c = cost + logTokens - math.Log(addCount) + (((lexiconBoundaries+addCount)*math.Log(lexiconBoundaries+addCount))-(lexiconBoundaries*math.Log(lexiconBoundaries))+getCodeLength(lexiconCoding, construction))/corpusWeight
				}

				if bestPrev == -1 || c < bestCost {
					bestPrev = pt
					bestCost = c
				}

				continue
			}
		}

		if bestPrev == -1 {
			panic("no best path")
		}

		grid[t] = cell{bestCost, bestPrev}
	}

	if len(grid) != len(compound)+1 {
		panic("invalid grid length")
	}

	var constructions []string

	lastT := len(compound)
	prev := grid[lastT].prev

	for prev != -1 {
		constructions = append(constructions, compound[prev:lastT])
		lastT = prev
		prev = grid[lastT].prev
	}

	for lo, hi := 0, len(constructions)-1; lo < hi; lo, hi = lo+1, hi-1 {
		constructions[lo], constructions[hi] = constructions[hi], constructions[lo]
	}

	cost := grid[len(compound)].cost + math.Log(corpusTokens+corpusBoundaries) - math.Log(corpusBoundaries)

	if len(constructions) == 0 {
		panic("no constructions")
	}

	return constructions, cost
}
