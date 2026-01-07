package mbpe

import (
	"math"
	"unicode/utf8"

	"github.com/jonasknobloch/mbpe/morfessor"
)

type Morfessor struct {
	model *morfessor.Model
	alpha float64
}

func NewMorfessor(alpha float64) *Morfessor {
	if alpha < 0 || alpha > 1 {
		panic("alpha must be in [0, 1]")
	}

	return &Morfessor{
		model: morfessor.NewModel(),
		alpha: alpha,
	}
}

func (m *Morfessor) LoadModel(name string) error {
	return m.model.LoadModel(name)
}

func (m *Morfessor) Segment(compound string) ([]string, float64) {
	substrings, count := m.model.Segment(compound)

	singles := 0

	for _, s := range substrings {
		if utf8.RuneCountInString(s) == 1 {
			singles++
		}

		if singles == 2 {
			return []string{compound}, 0
		}
	}

	if count == math.NaN() || count < 0 {
		return []string{compound}, 0
	}

	return substrings, m.alpha
}
