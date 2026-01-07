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
	var bounds []int

	i := 0

	for _, r := range message {
		l := len(string(r))

		bounds = append(bounds, i+l)

		i += l
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
	compoundLength := len(unicodeScalarBounds(compound))

	grid := []struct {
		float64
		*int
	}{
		{0, nil},
	}

	corpusTokens := float64(model.XCorpusCoding.Tokens)
	corpusBoundaries := float64(model.XCorpusCoding.Boundaries)

	logTokens := 0.0

	if corpusTokens+corpusBoundaries+addCount > 0 {
		logTokens = math.Log(corpusTokens + corpusBoundaries + addCount)
	}

	badLikelihood := float64(compoundLength)*logTokens + 1.0

	bounds := unicodeScalarBounds(compound)

	var boundsUpper, boundsLower = make([]int, len(bounds)), make([]int, len(bounds))

	copy(boundsUpper, bounds)
	copy(boundsLower, bounds)

	boundsLower = append([]int{0}, boundsLower[:len(boundsLower)-1]...)

	for _, t := range boundsUpper {
		var bestPath *int
		var bestCost *float64

		evalPath := func(path int, cost float64) {
			if bestCost == nil || cost < *bestCost {
				bestPath = &path
				bestCost = &cost
			}
		}

		for _, pt := range boundsLower {
			if pt >= t {
				break // up to but not including t
			}

			construction := compound[pt:t]

			if utf8.RuneCountInString(construction) > maxLen {
				continue
			}

			cost := grid[pt].float64

			if analysis, ok := model.XAnalyses[construction]; ok {
				if len(analysis.Splitloc) == 0 || analysis.Splitloc[0] == 0 {
					if analysis.Count <= 0 {
						panic(fmt.Sprintf("Construction count of '%s' is %d", construction, analysis.Count))
					}

					cost += logTokens - math.Log(float64(analysis.Count)+addCount)

					evalPath(pt, cost)

					continue
				}
			}

			if addCount == 0 {
				if len(unicodeScalarBounds(construction)) == 1 {
					cost += badLikelihood

					evalPath(pt, cost)
				}

				continue
			}

			if addCount > 0 {
				lexiconCoding := model.XLexiconCoding
				corpusCoding := model.XCorpusCoding

				lexiconBoundaries := float64(lexiconCoding.Boundaries)
				corpusWeight := float64(corpusCoding.Weight)

				if corpusCoding.Tokens == 0 {
					cost += addCount*math.Log(addCount) + getCodeLength(lexiconCoding, construction)/corpusWeight
				} else {
					cost += logTokens - math.Log(addCount) + (((lexiconBoundaries+addCount)*math.Log(lexiconBoundaries+addCount))-(lexiconBoundaries*math.Log(lexiconBoundaries))+getCodeLength(lexiconCoding, construction))/corpusWeight
				}

				evalPath(pt, cost)

				continue
			}
		}

		if bestPath == nil {
			panic("no best path")
		}

		for len(grid) < t {
			grid = append(grid, struct {
				float64
				*int
			}{
				math.NaN(),
				nil,
			})
		}

		grid = append(grid, struct {
			float64
			*int
		}{
			*bestCost,
			bestPath,
		})
	}

	var constructions []string

	if len(grid) != len(compound)+1 {
		panic("invalid grid length")
	}

	cost := grid[len(grid)-1].float64
	path := grid[len(grid)-1].int

	lastT := len(compound)

	for path != nil {
		t := *path
		constructions = append(constructions, compound[t:lastT])
		path = grid[t].int
		lastT = t
	}

	for i, j := 0, len(constructions)-1; i < j; i, j = i+1, j-1 {
		constructions[i], constructions[j] = constructions[j], constructions[i]
	}

	cost += math.Log(corpusTokens+corpusBoundaries) - math.Log(corpusBoundaries)

	if len(constructions) == 0 {
		panic("no constructions")
	}

	return constructions, cost
}
