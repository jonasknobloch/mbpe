package mbpe

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	RuneUnicode32 = iota
	RuneWhitespaceNotUnicode32
	RuneLetter
	RuneNumber
	RuneOther
	StateInitial
	StateU32
	StateWhitespaceNotUnicode32
	StateLetter
	StateNumber
	StateOther
	StateWhitespaceLookAhead
)

type FSA struct {
	state  int
	input  []rune
	static []string
}

func NewFSA() *FSA {
	return &FSA{
		state:  StateInitial,
		input:  make([]rune, 0),
		static: []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"},
	}
}

func (f *FSA) Reset() {
	f.state = StateInitial
	f.input = make([]rune, 0)
}

func (f *FSA) Read(next rune) bool {
	var r int

	if next == 32 {
		r = RuneUnicode32
	} else if unicode.IsSpace(next) {
		r = RuneWhitespaceNotUnicode32
	} else if unicode.IsLetter(next) {
		r = RuneLetter
	} else if unicode.IsNumber(next) {
		r = RuneNumber
	} else {
		r = RuneOther
	}

	if f.state == StateInitial {
		switch r {
		case RuneUnicode32:
			f.input = append(f.input, next)
			f.state = StateU32

			break
		case RuneWhitespaceNotUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceNotUnicode32

			break
		case RuneLetter:
			f.input = append(f.input, next)
			f.state = StateLetter

			break
		case RuneNumber:
			f.input = append(f.input, next)
			f.state = StateNumber

			break
		default:
			f.input = append(f.input, next)
			f.state = StateOther
		}
	} else if f.state == StateU32 {
		switch r {
		case RuneUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceLookAhead

			break
		case RuneWhitespaceNotUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceLookAhead

			break
		case RuneLetter:
			f.input = append(f.input, next)
			f.state = StateLetter

			break
		case RuneNumber:
			f.input = append(f.input, next)
			f.state = StateNumber

			break
		default:
			f.input = append(f.input, next)
			f.state = StateOther
		}
	} else if f.state == StateWhitespaceNotUnicode32 {
		switch r {
		case RuneUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceLookAhead

			break
		case RuneWhitespaceNotUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceLookAhead

			break
		default:
			return false
		}
	} else if f.state == StateNumber {
		switch r {
		case RuneNumber:
			f.input = append(f.input, next)
			f.state = StateNumber

			break
		default:
			return false
		}
	} else if f.state == StateLetter {
		switch r {
		case RuneLetter:
			f.input = append(f.input, next)
			f.state = StateLetter

			break
		default:
			return false
		}
	} else if f.state == StateOther {
		switch r {
		case RuneOther:
			f.input = append(f.input, next)
			f.state = StateOther

			break
		default:
			return false
		}
	} else if f.state == StateWhitespaceLookAhead {
		switch r {
		case RuneUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceLookAhead

			break
		case RuneWhitespaceNotUnicode32:
			f.input = append(f.input, next)
			f.state = StateWhitespaceLookAhead

			break
		default:
			return false
		}
	} else {
		panic("invalid state")
	}

	return true
}

func (f *FSA) FindAll(s string) []string {
	var findAll func(runes []rune, matches []string) []string

	findAll = func(runes []rune, matches []string) []string {
		s = string(runes)

		for _, v := range f.static {
			if strings.HasPrefix(s, v) {
				matches = append(matches, v)
				runes = runes[utf8.RuneCountInString(v):]

				if len(runes) == 0 {
					return matches
				}

				return findAll(runes, matches)
			}
		}

		for i, r := range runes {
			ok := f.Read(r)

			if !ok {
				if f.state == StateInitial {
					return matches
				}

				if f.state == StateWhitespaceLookAhead {
					matches = append(matches, string(f.input[:len(f.input)-1]))

					f.Reset()

					return findAll(runes[i-1:], matches)
				}

				matches = append(matches, string(f.input))

				f.Reset()

				return findAll(runes[i:], matches)
			}
		}

		if len(f.input) > 0 {
			matches = append(matches, string(f.input))
		}

		return matches
	}

	defer f.Reset()

	return findAll([]rune(s), make([]string, 0))
}
