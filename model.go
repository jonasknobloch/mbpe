package mbpe

import (
	"bufio"
	"fmt"
	"strings"

	"github.com/jonasknobloch/mbpe/hf"
)

type Model interface {
	Tokenize(string) []int
	Save(string, string) error
	Load(string, string) error
}

type MBPE struct {
	vocab  []string
	atoi   map[string]int
	itoa   map[int]string
	merges [][2]string
	ranks  map[[2]string]int
}

func NewMBPE() *MBPE {
	return &MBPE{}
}

func (m *MBPE) Vocab() []string {
	return m.vocab
}

func (m *MBPE) Merges() [][2]string {
	return m.merges
}

func (m *MBPE) InitVocab(n int) {
	m.vocab = make([]string, 0, n)

	m.atoi = make(map[string]int, n)
	m.itoa = make(map[int]string, n)
}

func (m *MBPE) InitMerges(n int) {
	m.merges = make([][2]string, 0, n)

	m.ranks = make(map[[2]string]int, n)
}

func (m *MBPE) Len() int {
	return len(m.vocab)
}

func (m *MBPE) Cap() int {
	return cap(m.vocab)
}

func (m *MBPE) Alphabet() []string {
	offset := len(m.vocab) - len(m.merges)

	if offset < 0 {
		panic("unexpected vocabulary size")
	}

	return m.vocab[:len(m.vocab)-len(m.merges)]
}

func (m *MBPE) AddToken(token string) {
	if _, ok := m.atoi[token]; ok {
		panic("token already exists")
	}

	id := len(m.vocab)

	m.vocab = append(m.vocab, token)

	m.atoi[token] = id
	m.itoa[id] = token
}

func (m *MBPE) AddMerge(left, right string) {
	if _, ok := m.ranks[[2]string{left, right}]; ok {
		panic("merge already exists")
	}

	id := len(m.merges)

	m.merges = append(m.merges, [2]string{left, right})

	m.ranks[[2]string{left, right}] = id
}

func (m *MBPE) TokenizeLayered(phrase string, maxRank int) [][]int {
	merges := make([]int, 0)

	m.tokenize(phrase, &merges, maxRank)

	chunk := NewChunk(phrase, 1, nil, 0)

	layers := make([][]int, len(merges)+1)

	toIDs := func(tokens []string) []int {
		layer := make([]int, len(tokens))

		for i, token := range tokens {
			id, ok := m.atoi[token]

			if !ok {
				panic("unknown token")
			}

			layer[i] = id
		}

		return layer
	}

	layers[0] = toIDs(chunk.Tokens())

	for i, pos := range merges {
		chunk.MergePairIdx(pos)

		layers[i+1] = toIDs(chunk.Tokens())
	}

	return layers
}

func (m *MBPE) Tokenize(phrase string) []int {
	return m.tokenize(phrase, nil, -1)
}

func (m *MBPE) tokenize(phrase string, merges *[]int, maxRank int) []int {
	c := NewChunk(phrase, 1, nil, 0)

	for {
		pairs := c.Pairs()

		if len(pairs) == 0 {
			break
		}

		idx := -1
		rank := -1

		for i, pair := range pairs {
			r, ok := m.ranks[pair]

			if !ok {
				continue
			}

			if maxRank > -1 && r > maxRank-1 {
				continue
			}

			if idx == -1 || r < rank {
				idx = i
				rank = r
			}
		}

		if idx == -1 {
			break
		}

		if merges != nil {
			*merges = append(*merges, idx)
		}

		c.MergePairIdx(idx)
	}

	r := make([]int, len(c.bounds)-1)

	for i, token := range c.Tokens() {
		id, ok := m.atoi[token]

		if !ok {
			panic("unknown token")
		}

		r[i] = id
	}

	return r
}

func (m *MBPE) ToString(ids []int) []string {
	result := make([]string, len(ids))

	for i, id := range ids {
		token, ok := m.itoa[id]

		if !ok {
			panic("unknown token identifier")
		}

		result[i] = token
	}

	return result
}

func (m *MBPE) Save(vocab, merges string) error {
	if err := toJSON(vocab, hf.Vocab(m.atoi)); err != nil {
		return err
	}

	if err := toFile(merges, func(writer *bufio.Writer) error {
		for _, merge := range m.merges {
			if _, err := writer.WriteString(merge[0] + " " + merge[1] + "\n"); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return err
	}

	return nil
}

func (m *MBPE) Load(vocab, merges string) error {
	atoi := make(map[string]int)

	if err := fromJSON(vocab, &atoi); err != nil {
		return err
	}

	itoa := make(map[int]string, len(atoi))

	for token, id := range atoi {
		itoa[id] = token
	}

	vs := make([]string, len(itoa))

	for i := range len(itoa) {
		vs[i] = itoa[i]
	}

	ms := make([][2]string, 0) // unknown number of ms

	first := true

	if err := fromFile(merges, func(scanner *bufio.Scanner) error {
		for scanner.Scan() {
			line := scanner.Text()

			if first {
				first = false

				if strings.HasPrefix(line, "#version:") {
					continue
				}
			}

			if err := scanner.Err(); err != nil {
				return err
			}

			var merge [2]string

			if _, err := fmt.Sscanf(line, "%s %s", &merge[0], &merge[1]); err != nil {
				return err
			}

			ms = append(ms, merge)
		}

		return nil
	}); err != nil {
		return err
	}

	ranks := make(map[[2]string]int, len(ms))

	for i, merge := range ms {
		ranks[merge] = i
	}

	m.vocab = vs
	m.atoi = atoi
	m.itoa = itoa
	m.merges = ms
	m.ranks = ranks

	return nil
}

func (m *MBPE) Trim(n int) {
	nVocab := len(m.Vocab())
	nAlphabet := len(m.Alphabet())
	nMerges := len(m.Merges())

	if n > nVocab {
		panic("new vocab size exceeds current vocab size")
	}

	if n < nAlphabet {
		panic("alphabet exceeds new vocab size")
	}

	r := n - nAlphabet

	for i := n; i < nVocab; i++ {
		delete(m.atoi, m.itoa[i])
		delete(m.itoa, i)
	}

	for j := r; j < nMerges; j++ {
		delete(m.ranks, m.merges[j])
	}

	m.vocab = m.vocab[:n]
	m.merges = m.merges[:r]
}
