package mbpe

import (
	"reflect"
	"testing"
)

type baseInverter struct{}

func (s *baseInverter) Segment(text string) ([]string, bool) {
	return []string{"foo", "bar"}, true
}

func TestInverter_Segment(t *testing.T) {
	i := NewInverter(&baseInverter{})

	segmentation, _ := i.Segment("foobar")

	expected := []string{"f", "o", "ob", "a", "r"}

	if !reflect.DeepEqual(segmentation, expected) {
		t.Errorf("expected %v but got %v", expected, segmentation)
	}
}
