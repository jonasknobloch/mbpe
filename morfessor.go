package mbpe

import (
	"math"
	"unicode/utf8"

	"github.com/jonasknobloch/mbpe/morfessor"
)

type Morfessor struct {
	model *morfessor.Model
}

func NewMorfessor() *Morfessor {
	return &Morfessor{
		model: morfessor.NewModel(),
	}
}

func (m *Morfessor) LoadModel(name string) error {
	return m.model.LoadModel(name)
}

func (m *Morfessor) Segment(compound string) ([]string, bool) {
	substrings, count := m.model.Segment(compound)

	singles := 0

	for _, s := range substrings {
		if utf8.RuneCountInString(s) == 1 {
			singles++
		}

		if singles == 2 {
			return []string{compound}, false
		}
	}

	if count == math.NaN() || count < 0 {
		return []string{compound}, false
	}

	return substrings, true
}
