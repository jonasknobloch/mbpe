package mbpe

import (
	"reflect"
	"testing"
)

type baseUniformizer struct{}

func (s *baseUniformizer) Segment(text string) []string {
	return []string{"foo", "bar"}
}

func TestUniformizer_Segment(t *testing.T) {
	u := NewUniformizer(&baseUniformizer{})

	segmentation := u.Segment("foobarbaz")

	expected := []string{"foob", "arbaz"}

	if !reflect.DeepEqual(segmentation, expected) {
		t.Errorf("expected %v but got %v", expected, segmentation)
	}
}
