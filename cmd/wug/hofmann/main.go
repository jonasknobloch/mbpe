package main

import (
	"fmt"
	"log"
	"math"
	"slices"

	"github.com/jonasknobloch/mbpe"
	"github.com/jonasknobloch/mbpe/cmd/wug/nonce"
)

func main() {
	paper()
	// babyLM()
	// babyLM2()
}

func paper() {
	fmt.Println("Table 4:")
	table4()
	fmt.Println("\nFigure 5a:")
	figure5a()
	fmt.Println("\nFigure 5b:")
	figure5b()
}

func babyLM() {
	paths, stubs := mbpe.WalkResultsStatic("data/wug_results/out/gpt2_%d_%s%s_babylm_v2_ity_ness_nonce.json")

	fmt.Printf("vocab,prefix,alpha,suffix,accuracy,error\n")

	for i, path := range paths {
		results, deviations := againstGold(path, [][]string{nonce.Able[:], nonce.Ish[:], nonce.Ive[:], nonce.Ous[:], nonce.All}, []float64{-1}, 12) // set group size 12 to average across prompts per nonce adjective

		for j, values := range results {
			a := values[0]
			b := deviations[j][0]

			if math.IsNaN(a) {
				a = 0
			}

			if math.IsNaN(b) {
				b = 0
			}

			s := ""

			switch j {
			case 0:
				s = "able"
			case 1:
				s = "ish"
			case 2:
				s = "ive"
			case 3:
				s = "ous"
			default:
				s = "all"
			}

			fmt.Printf("%s,%s,%s,%s,%.4f,%.4f\n", stubs[i][0], stubs[i][1], stubs[i][2], s, a, b)
		}
	}
}

func babyLM2() {
	paths, stubs := mbpe.WalkResultsStatic("data/wug_results/out/gpt2_%d_%s%s_babylm_v2_ity_ness_nonce.json")

	ratios, _, _, _ := surveyResponses("data/wug_results/survey_responses.json")

	columns := getKeys(toSet(ratios))

	slices.Reverse(columns)

	columns = append(columns, -1)

	fmt.Printf("vocab,prefix,alpha,")

	for _, c := range columns {
		if c == -1 {
			fmt.Printf("avg,")

			continue
		}

		fmt.Printf("%.2f,", c)
	}

	for _, c := range columns {
		if c == -1 {
			fmt.Println("avg_err")

			continue
		}

		fmt.Printf("%.2f_err,", c)
	}

	for i, path := range paths {
		fmt.Printf("%s,%s,%s", stubs[i][0], stubs[i][1], stubs[i][2])

		results, deviation := againstGold(path, [][]string{nonce.All}, columns, 1) // set group size 12 to average across prompts per nonce adjective

		for _, v := range results[len(results)-1] {
			if math.IsNaN(v) {
				v = 0
			}

			fmt.Printf(",%.2f", v)
		}

		for _, v := range deviation[len(deviation)-1] {
			if math.IsNaN(v) {
				v = 0
			}

			fmt.Printf(",%.2f", v)
		}

		fmt.Println()
	}
}

func againstGold(name string, adjectives [][]string, ratios []float64, groupSize int) ([][]float64, [][]float64) {
	ratiosGold, binaryGold, keys, err := surveyResponses("data/wug_results/survey_responses.json")

	if err != nil {
		log.Fatal(err)
	}

	predictions, err := processPredictions(name)

	if err != nil {
		log.Fatal(err)
	}

	ratiosPred, binaryPred := evalPredictions(predictions, groupSize)

	groups := len(binaryPred) / len(binaryGold)

	r := make([][]float64, 0)
	e := make([][]float64, 0)

	for _, adj := range adjectives {
		acc := make([]float64, 0)
		dev := make([]float64, 0)

		allowed := toSet(adj)

		for _, ratio := range ratios {
			p := 0
			n := 0

			totalError := 0.0

			for i, key := range keys {
				if key != nonce.All[i] {
					panic("unexpected nonce adjective: " + key)
				}

				if _, ok := allowed[key]; !ok {
					continue
				}

				if ratio != -1 && ratiosGold[i] != ratio {
					continue
				}

				for j := 0; j < groups; j++ {
					if binaryPred[(i*groups)+j] == binaryGold[i] {
						p += 1
					} else {
						n += 1
					}

					totalError += math.Abs(ratiosPred[(i*groups)+j] - ratiosGold[i])
				}
			}

			acc = append(acc, float64(p)/float64(p+n))
			dev = append(dev, totalError/float64(p+n))
		}

		r = append(r, acc)
		e = append(e, dev)
	}

	return r, e
}

func table4() {
	acc, _ := againstGold("data/wug_results/gptj_predictions_nonce.json", [][]string{nonce.Able[:], nonce.Ish[:], nonce.Ive[:], nonce.Ous[:]}, []float64{-1}, 1)

	for _, v := range acc {
		fmt.Printf("%.3f\n", v[0])
	}

	return
}

func figure5a() {
	predictions, err := processPredictions("data/wug_results/gptj_predictions_nonce.json")

	if err != nil {
		log.Fatal(err)
	}

	_, binary := evalPredictions(predictions, 1)

	groups := len(binary) / len(nonce.All)

	for _, adj := range [][]string{nonce.Able[:], nonce.Ish[:], nonce.Ive[:], nonce.Ous[:]} {
		allowed := toSet(adj)

		p := 0
		n := 0

		for i, key := range nonce.All {
			if _, ok := allowed[key]; !ok {
				continue
			}

			for j := 0; j < groups; j++ {
				if binary[(i*groups)+j] {
					p++
				} else {
					n++
				}
			}
		}

		fmt.Printf("%.3f\n", 1-(float64(p)/float64(p+n)))
	}
}

func figure5b() {
	_, binaryGold, keys, _ := surveyResponses("data/wug_results/survey_responses.json")

	for _, adj := range [][]string{nonce.Able[:], nonce.Ish[:], nonce.Ive[:], nonce.Ous[:]} {
		allowed := toSet(adj)

		p := 0
		n := 0

		for i, key := range keys {
			if _, ok := allowed[key]; !ok {
				continue
			}

			if binaryGold[i] {
				p++
			} else {
				n++
			}
		}

		fmt.Printf("%.3f\n", 1-(float64(p)/float64(p+n)))
	}
}
