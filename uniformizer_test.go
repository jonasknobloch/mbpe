package mbpe

import (
	"reflect"
	"testing"
)

type baseUniformizer struct{}

func (s *baseUniformizer) Segment(text string) ([]string, bool) {
	return []string{"foo", "bar"}, true
}

func TestUniformizer_Segment(t *testing.T) {
	u := NewUniformizer(&baseUniformizer{})

	segmentation, _ := u.Segment("foobarbaz")

	expected := []string{"foob", "arbaz"}

	if !reflect.DeepEqual(segmentation, expected) {
		t.Errorf("expected %v but got %v", expected, segmentation)
	}
}
