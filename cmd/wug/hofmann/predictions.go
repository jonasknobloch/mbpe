package main

import (
	"errors"
	"math"
	"strings"

	"github.com/jonasknobloch/mbpe"
)

type Entry struct {
	text     string
	tokens   []string
	logProbs []float64
}

type Predictions = map[string]map[string][][2]Entry

func processPredictions(name string) (Predictions, error) {
	var data map[string]interface{}

	if err := mbpe.FromJSON(name, &data); err != nil {
		return nil, err
	}

	results := make(Predictions)

	for i, t := range types {
		results[t] = make(map[string][][2]Entry)

		for _, instruction := range instructions[i] {
			results[t][instruction] = make([][2]Entry, 200)
		}
	}

	for i, t := range types {
		var typesData map[string]interface{}

		if d, ok := data[types[i]].(map[string]interface{}); ok {
			typesData = d
		} else {
			return nil, errors.New("unexpected data format")
		}

		for j, instruction := range instructions[i] {
			var instructionData map[string]interface{}

			if d, ok := typesData[instructions[i][j]].(map[string]interface{}); ok {
				instructionData = d
			} else {
				return nil, errors.New("unexpected data format")
			}

			keys := getKeys(instructionData)

			for k, key := range keys {
				pairData := instructionData[key].(map[string]interface{})

				stims := strings.Split(key, "_")

				left := pairData[stims[0]].([]interface{})
				right := pairData[stims[1]].([]interface{})

				tokensLeft := left[0].([]interface{})
				logProbsLeft := left[1].([]interface{})

				tokensRight := right[0].([]interface{})
				logProbsRight := right[1].([]interface{})

				pair := [2]Entry{
					{
						text:     stims[0],
						tokens:   make([]string, len(tokensLeft)),
						logProbs: make([]float64, len(logProbsLeft)),
					},
					{
						text:     stims[1],
						tokens:   make([]string, len(tokensRight)),
						logProbs: make([]float64, len(logProbsRight)),
					},
				}

				for pos, token := range tokensLeft {
					pair[0].tokens[pos] = token.(string)
					pair[0].logProbs[pos] = logProbsLeft[pos].(float64)
				}

				for pos, token := range tokensRight {
					pair[1].tokens[pos] = token.(string)
					pair[1].logProbs[pos] = logProbsRight[pos].(float64)
				}

				results[t][instruction][k] = pair
			}
		}
	}

	return results, nil
}

func predictionsToSlice(predictions Predictions) [][][2]Entry {
	r := make([][][2]Entry, 200)

	for i := range r {
		r[i] = make([][2]Entry, 12)
	}

	for i, t := range types {
		for j, instruction := range instructions[i] {
			promptIndex := (i * 4) + j // 4 prompts per type

			for k, pair := range predictions[t][instruction] {
				r[k][promptIndex] = pair
			}
		}
	}

	return r
}

func flattenPredictions(predictions [][][2]Entry) [][2]Entry {
	r := make([][2]Entry, 0, 200*12)

	for _, pairs := range predictions {
		for _, pair := range pairs {
			r = append(r, pair)
		}
	}

	return r
}

func cumulatePredictions(predictions [][2]Entry, groupSize int) ([]float64, []bool) {
	if len(predictions)%groupSize != 0 {
		panic("unexpected group size")
	}

	n := len(predictions) / groupSize

	ratios := make([]float64, n)
	binary := make([]bool, n)

	for i := 0; i < n; i++ {
		sumLeft := 0.0
		sumRight := 0.0

		left := 0
		right := 0

		for j := 0; j < groupSize; j++ {
			pair := predictions[i*groupSize+j]

			a, b := evalPair(pair)

			sumLeft += a
			sumRight += b

			if a > b { // smaller absolute (neg log prob)
				left++
			} else {
				right++
			}
		}

		sumLeftNorm := sumLeft / float64(groupSize)
		sumRightNorm := sumRight / float64(groupSize)

		ratios[i] = math.Exp(sumLeftNorm) / (math.Exp(sumLeftNorm) + math.Exp(sumRightNorm))
		binary[i] = sumLeftNorm > sumRightNorm // binary[i] = left > right
	}

	return ratios, binary
}

func evalPair(pair [2]Entry) (float64, float64) {
	sumLeft := 0.0
	sumRight := 0.0

	for i := 0; i < len(pair[0].tokens); i++ {
		sumLeft += pair[0].logProbs[i]
	}

	for i := 0; i < len(pair[1].tokens); i++ {
		sumRight += pair[1].logProbs[i]
	}

	return sumLeft, sumRight
}

func evalPairAvg(pair [2]Entry) (float64, float64) {
	sumLeft := 0.0
	sumRight := 0.0

	m := 0
	n := 0

	for i := 0; i < len(pair[0].tokens); i++ {
		sumLeft += pair[0].logProbs[i]
		m++
	}

	for i := 0; i < len(pair[1].tokens); i++ {
		sumRight += pair[1].logProbs[i]
		n++
	}

	return sumLeft / float64(m), sumRight / float64(n)
}

func evalPredictions(predictions Predictions, groupSize int) ([]float64, []bool) {
	return cumulatePredictions(flattenPredictions(predictionsToSlice(predictions)), groupSize)
}
