package mbpe

import (
	"os"
	"slices"
	"testing"

	"go.jknobloc.com/x/shelf"
)

func TestMain(m *testing.M) {
	shelf.Root = "../.shelf"

	os.Exit(m.Run())
}

func BenchmarkMBPE_Tokenize(b *testing.B) {
	model := NewMBPE()

	vocab := shelf.Abs("models/gpt2/vocab.json")
	merges := shelf.Abs("models/gpt2/merges.txt")

	if err := model.Load(vocab, merges); err != nil {
		b.Fatal(err)
	}

	tokenizer := NewTokenizer(model)

	byteLevel := NewByteLevel(true)

	tokenizer.SetPreTokenizer(byteLevel)
	tokenizer.SetDecoder(byteLevel)

	for i := 0; i < b.N; i++ {
		tokenizer.Tokenize("To infinity and beyond!")
	}
}

func TestMBPE_Tokenize(t *testing.T) {
	model := NewMBPE()

	vocab := shelf.Abs("models/gpt2/vocab.json")
	merges := shelf.Abs("models/gpt2/merges.txt")

	if err := model.Load(vocab, merges); err != nil {
		t.Fatal(err)
	}

	tokenizer := NewTokenizer(model)

	byteLevel := NewByteLevel(true)

	tokenizer.SetPreTokenizer(byteLevel)
	tokenizer.SetDecoder(byteLevel)

	text := "The quick brown fox jumps over the lazy dog."

	ids := tokenizer.Tokenize(text)

	expected := []int{383, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13}

	if !slices.Equal(ids, expected) {
		t.Errorf("expected %v but got %v", expected, ids)
	}
}
