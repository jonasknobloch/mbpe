package mbpe

import (
	"errors"
	"strings"
)

type Static struct {
	dict map[string][]string
}

func NewStatic() *Static {
	return &Static{
		dict: make(map[string][]string),
	}
}

func (c *Static) LoadDict(name string) error {
	return readTsv(name, func(record []string) error {
		if len(record) != 2 {
			return errors.New("unexpected number of fields")
		}

		c.dict[record[0]] = strings.Split(record[1], " ")

		return nil
	})
}

func (c *Static) Segment(text string) ([]string, bool) {
	substrings, ok := c.dict[text]

	if !ok {
		return []string{text}, false
	}

	return substrings, ok
}
