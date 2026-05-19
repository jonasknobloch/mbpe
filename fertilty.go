package mbpe

import (
	"errors"

	"go.jknobloc.com/x/dataset"
)

type FertilityEvaluator struct {
	dict *Dict
}

func NewFertilityEvaluator() *FertilityEvaluator {
	return &FertilityEvaluator{
		dict: NewDict(),
	}
}

func (f *FertilityEvaluator) InitDict(reader dataset.Reader) error {
	return f.dict.ProcessTexts(reader.Texts())
}

func (f *FertilityEvaluator) LoadDict(name string) error {
	return f.dict.Load(name)
}

func (f *FertilityEvaluator) Eval(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	numTokens := 0
	numChunks := 0

	m, ok := tokenizer.model.(*MBPE)

	if !ok {
		return nil, errors.New("unexpected model type")
	}

	for _, chunk := range f.dict.Items() {
		var ids []int

		func() {
			defer func() {
				if r := recover(); r != nil {
					ids = nil
				}
			}()

			ids = m.tokenize(chunk.src, nil, maxRank)
		}()

		numTokens += len(ids) * chunk.n
		numChunks += chunk.n
	}

	fertility := float64(numTokens) / float64(numChunks)

	return []float64{fertility}, nil
}
