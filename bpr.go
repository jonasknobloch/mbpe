package mbpe

import (
	"errors"
	"strings"

	legacy "github.com/jonasknobloch/mbpe/bpr"
)

const (
	MaxPrecision = iota
	MaxRecall
	MaxAccuracy
	MaxF1
)

type BPREvaluator struct {
	skipSingletonGoldSegmentations bool
	skipSingletonTestSegmentations bool
	chooseBestTokenizationLayer    bool
	useLegacyEval                  bool
	gold                           [][]string
	addPrefixSpace                 bool
}

func NewBPREvaluator() *BPREvaluator {
	return &BPREvaluator{
		skipSingletonGoldSegmentations: true,
		skipSingletonTestSegmentations: false,
		chooseBestTokenizationLayer:    false,
		useLegacyEval:                  false,
		gold:                           make([][]string, 0),
		addPrefixSpace:                 true,
	}
}

func (bpr *BPREvaluator) LoadSegmentations(name string) error {
	return readTsv(name, func(record []string) error {
		if len(record) != 2 {
			return errors.New("unexpected number of fields")
		}

		compound := record[0]
		segmentation := strings.Split(record[1], " ")

		if bpr.addPrefixSpace {
			compound = " " + compound
			segmentation[0] = " " + segmentation[0]
		}

		bpr.gold = append(bpr.gold, append([]string{compound}, segmentation...))

		return nil
	})
}

func (bpr *BPREvaluator) Eval(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	if bpr.useLegacyEval {
		return bpr.evalLegacy(tokenizer, maxRank)
	}

	return bpr.eval(tokenizer, maxRank)
}

func (bpr *BPREvaluator) evalLegacy(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	gold := make([][]string, 0)
	pred := make([][]string, 0)

	for _, split := range bpr.gold {
		if bpr.skipSingletonGoldSegmentations && len(split) == 2 {
			continue
		}

		segmentation, ok := GetTokenizerSegmentation(tokenizer, split[0], maxRank)

		if !ok {
			continue
		}

		if bpr.skipSingletonTestSegmentations && len(segmentation) == 1 {
			continue
		}

		gold = append(gold, split[1:])
		pred = append(pred, segmentation)
	}

	precision, recall, f1 := legacy.Eval(gold, pred)

	return []float64{precision, recall, f1}, nil
}

func (bpr *BPREvaluator) eval(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	sumPrecision := 0.0
	sumRecall := 0.0

	total := 0.0

	for _, split := range bpr.gold {
		if bpr.skipSingletonGoldSegmentations && len(split) == 2 {
			continue
		}

		word := split[0]
		gold := split[1:]

		var layers [][]string

		if !bpr.chooseBestTokenizationLayer {
			segmentation, ok := GetTokenizerSegmentation(tokenizer, word, maxRank)

			if !ok {
				continue
			}

			layers = [][]string{segmentation}
		} else {
			segmentation, ok := GetTokenizerSegmentationLayered(tokenizer, word, maxRank)

			if !ok {
				continue
			}

			layers = segmentation
		}

		if bpr.skipSingletonTestSegmentations && len(layers[len(layers)-1]) == 1 {
			continue
		}

		top, counts := evalSegmentations(layers, gold, word, MaxF1)

		tp := counts[0]
		fp := counts[1]
		fn := counts[3]

		p := float64(tp) / float64(tp+fp)
		r := float64(tp) / float64(tp+fn)

		if len(top) == 1 {
			p = 1 // precision 1 of no bounds in prediction
		}

		if len(gold) == 1 {
			r = 1 // recall 1 if no bounds in gold segmentation
		}

		sumPrecision += p
		sumRecall += r

		total++
	}

	precision := sumPrecision / total
	recall := sumRecall / total

	f1 := 2 * precision * recall / (precision + recall)

	return []float64{precision, recall, f1}, nil
}

func evalSegmentations(segmentations [][]string, gold []string, src string, mode int) ([]string, [4]int) {
	results := make([][4]int, len(segmentations))

	boundsSet := func(segmentation []string) map[int]struct{} {
		b := make(map[int]struct{})

		i := 0

		for j, g := range segmentation {
			if j == len(segmentation)-1 {
				break
			}

			l := len(g)

			b[i+l] = struct{}{}
			i += l
		}

		return b
	}

	b := boundsSet(gold)

	for n, segmentation := range segmentations {
		tp := 0
		fp := 0
		tn := 0
		fn := 0

		a := boundsSet(segmentation)

		for i := range src {
			if i == 0 {
				continue
			}

			_, inA := a[i]
			_, inB := b[i]

			if inA {
				if inB {
					tp++
				} else {
					fp++
				}
			} else {
				if inB {
					fn++
				} else {
					tn++
				}
			}
		}

		results[n] = [4]int{tp, fp, tn, fn}
	}

	m := float64(0)
	i := -1

	for j, v := range results {
		tp := v[0]
		fp := v[1]
		tn := v[2]
		fn := v[3]

		precision := float64(tp) / float64(tp+fp)
		recall := float64(tp) / float64(tp+fn)
		accuracy := float64(tp+tn) / float64(tp+tn+fp+fn)
		f1 := 2 * precision * recall / (precision + recall)

		switch mode {
		case MaxPrecision:
			if precision >= m {
				m = precision
				i = j
			}
		case MaxRecall:
			if recall >= m {
				m = recall
				i = j
			}
		case MaxAccuracy:
			if accuracy >= m {
				m = accuracy
				i = j
			}
		case MaxF1:
			if f1 >= m {
				m = f1
				i = j
			}
		}
	}

	if i == -1 {
		i = len(results) - 1
	}

	// fmt.Println(src, ":", gold, ":", i, ":", segmentations)

	return segmentations[i], results[i]
}

func (bpr *BPREvaluator) EvalFile(name string) ([]float64, error) {
	gold := make([][]string, len(bpr.gold))
	pred := make([][]string, len(bpr.gold))

	i := 0

	if err := readTsv(name, func(record []string) error {
		if len(record) != 2 {
			return errors.New("unexpected number of fields")
		}

		if record[0] != bpr.gold[i][0] {
			return errors.New("unexpected compound")
		}

		gold[i] = bpr.gold[i][1:]
		pred[i] = strings.Split(record[1], " ")

		i++

		return nil
	}); err != nil {
		return nil, err
	}

	precision, recall, f1 := legacy.Eval(gold, pred)

	return []float64{precision, recall, f1}, nil
}
