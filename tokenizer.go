package mbpe

type Tokenizer struct {
	preTokenizer PreTokenizer
	model        Model
	decoder      Decoder
}

func NewTokenizer(model Model) *Tokenizer {
	return &Tokenizer{
		model: model,
	}
}

func (t *Tokenizer) PreTokenizer() PreTokenizer {
	return t.preTokenizer
}

func (t *Tokenizer) Decoder() Decoder {
	return t.decoder
}

func (t *Tokenizer) SetPreTokenizer(preTokenizer PreTokenizer) {
	t.preTokenizer = preTokenizer
}

func (t *Tokenizer) SetDecoder(decoder Decoder) {
	t.decoder = decoder
}

func (t *Tokenizer) Model() Model {
	return t.model
}

func (t *Tokenizer) Tokenize(phrase string) []int {
	chunks := t.preTokenizer.PreTokenize(phrase)

	r := make([]int, 0)

	for _, chunk := range chunks {
		r = append(r, t.model.Tokenize(chunk)...)
	}

	return r
}
