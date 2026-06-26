package mbpe

import "strings"

type Inverter struct {
	segmenter Segmenter
}

func NewInverter(segmenter Segmenter) *Inverter {
	return &Inverter{
		segmenter: segmenter,
	}
}

func (i *Inverter) Segment(text string) ([]string, bool) {
	template, ok := i.segmenter.Segment(text)

	result := make([]string, 0)

	for _, t := range template {
		substrings := strings.Split(t, "")

		if len(result) == 0 {
			result = append(result, substrings...)

			continue
		}

		result[len(result)-1] = result[len(result)-1] + substrings[0]

		result = append(result, substrings[1:]...)
	}

	return result, ok
}
