package mbpe

import (
	"errors"
	"strings"

	"github.com/jonasknobloch/mbpe/bpr"
)

type MergeLayerEvaluator struct {
	gold           [][]string
	addPrefixSpace bool
}

func NewMergeLayerEvaluator() *MergeLayerEvaluator {
	return &MergeLayerEvaluator{
		gold:           make([][]string, 0),
		addPrefixSpace: true,
	}
}

func (ml *MergeLayerEvaluator) LoadSegmentations(name string) error {
	return readTsv(name, func(record []string) error {
		if len(record) != 2 {
			return errors.New("unexpected number of fields")
		}

		compound := record[0]
		segmentation := strings.Split(record[1], " ")

		if ml.addPrefixSpace {
			compound = " " + compound
			segmentation[0] = " " + segmentation[0]
		}

		ml.gold = append(ml.gold, append([]string{compound}, segmentation...))

		return nil
	})
}

func (ml *MergeLayerEvaluator) Eval(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	return ml.evalMorph(tokenizer, maxRank)
}

func (ml *MergeLayerEvaluator) evalMorph(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	sum := 0.0
	total := 0

	for _, split := range ml.gold {
		word := split[0]
		gold := split[1:]

		layers, ok := GetTokenizerSegmentationLayered(tokenizer, word, maxRank)

		if !ok {
			continue
		}

		gV := bpr.BoundaryVector(gold)
		pV := make([][]int, len(layers))

		for i, layer := range layers {
			pV[i] = bpr.BoundaryVector(layer)
		}

		s := 0
		n := 0

		for i, v := range gV {
			if v == 0 {
				continue
			}

			for j, p := range pV {
				if p[i] == 0 {
					s += j

					break
				}

				if j == len(pV)-1 {
					s += len(pV) // bound not destroyed
				}
			}

			n++
		}

		if n == 0 {
			sum += 1
			total++

			continue
		}

		average := float64(s) / float64(n)

		normalized := average / float64(len(pV))

		if normalized > 1 {
			panic("invalid layer")
		}

		sum += normalized

		total++
	}

	return []float64{sum / float64(total)}, nil
}
