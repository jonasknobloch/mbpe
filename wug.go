package mbpe

import "github.com/jonasknobloch/mbpe/cmd/wug/nonce"

type AdjectiveSuffixEvaluator struct {
	able          [50]string
	ish           [50]string
	ive           [50]string
	ous           [50]string
	reportAverage bool
}

func NewAdjectiveSuffixEvaluator() *AdjectiveSuffixEvaluator {
	able := nonce.Able
	ish := nonce.Ish
	ive := nonce.Ive
	ous := nonce.Ous

	return &AdjectiveSuffixEvaluator{
		able:          able,
		ish:           ish,
		ive:           ive,
		ous:           ous,
		reportAverage: false,
	}
}

func (a *AdjectiveSuffixEvaluator) SetReportAverage(reportAverage bool) {
	a.reportAverage = reportAverage
}

func (a *AdjectiveSuffixEvaluator) Eval(tokenizer Tokenizer, maxRank int) ([]float64, error) {
	m, ok := tokenizer.model.(*MBPE)

	if !ok {
		panic("unexpected model type")
	}

	eval := func(dict [50]string, suffix string) float64 {
		n := 0

		for _, v := range dict {
			t := m.tokenize("Ġ"+v, nil, maxRank)

			r := m.ToString(t)

			if r[len(r)-1] == suffix {
				n++
			}
		}

		return float64(n) / float64(len(dict))
	}

	able := eval(a.able, "able")
	ive := eval(a.ive, "ive")
	ish := eval(a.ish, "ish")
	ous := eval(a.ous, "ous")

	if a.reportAverage {
		return []float64{(able + ive + ish + ous) / 4.0}, nil
	}

	return []float64{able, ive, ish, ous}, nil
}
