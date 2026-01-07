package mbpe

import (
	"testing"

	"github.com/dlclark/regexp2"
)

func regexp2FindAllString(re *regexp2.Regexp, s string) []string {
	var matches []string

	m, _ := re.FindStringMatch(s)

	for m != nil {
		matches = append(matches, m.String())

		m, _ = re.FindNextMatch(m)
	}

	return matches
}

var re = regexp2.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+(?!\S)|\s+`, 0)

func TestFSA_FindAll(t *testing.T) {
	fsa := NewFSA()

	expected := []string{
		"Theirs",
		" for",
		" their",
		" style",
		" I",
		"'ll",
		" read",
		",",
		" his",
		" for",
		" his",
		" love",
		"'.",
		"\n",
	}

	out := fsa.FindAll("Theirs for their style I'll read, his for his love'.\n")

	for i, s := range out {
		if s != expected[i] {
			t.Errorf("expected [%s] but got [%s]", expected[i], s)
		}
	}
}

func BenchmarkFSA_FindAll(b *testing.B) {
	fsa := NewFSA()

	for i := 0; i < b.N; i++ {
		fsa.FindAll("Theirs for their style I'll read, his for his love'.\n")
	}
}

func BenchmarkRegexp2_FindAll(b *testing.B) {
	for i := 0; i < b.N; i++ {
		regexp2FindAllString(re, "Theirs for their style I'll read, his for his love'.\n")
	}
}

func FuzzFSA_FindAll(f *testing.F) {
	fsa := NewFSA()

	f.Add("foo")
	f.Add(" ")
	f.Add("\n")
	f.Add("   bar")

	f.Fuzz(func(t *testing.T, s string) {
		out := fsa.FindAll(s)
		ref := regexp2FindAllString(re, s)

		if len(out) != len(ref) {
			t.Fatalf("expected %s but got %s", ref, out)
		}

		for i, m := range out {
			if m != ref[i] {
				t.Errorf("expected [%s] but got [%s]", ref[i], m)
			}
		}
	})
}
