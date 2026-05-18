package mbpe

import (
	"reflect"
	"testing"
)

type baseInverter struct{}

func (s *baseInverter) Segment(text string) []string {
	return []string{"foo", "bar"}
}

func TestInverter_Segment(t *testing.T) {
	i := NewInverter(&baseInverter{})

	segmentation := i.Segment("foobar")

	expected := []string{"f", "o", "ob", "a", "r"}

	if !reflect.DeepEqual(segmentation, expected) {
		t.Errorf("expected %v but got %v", expected, segmentation)
	}
}
