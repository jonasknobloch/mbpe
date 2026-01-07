package morfessor

import (
	"reflect"
	"testing"

	pb "github.com/jonasknobloch/mbpe/morfessor/proto"
)

var model *pb.BaselineModel

func init() {
	m, err := decodeModel("../data/morfessor/unsup_model.proto")

	if err != nil {
		panic(err)
	}

	model = m
}

func TestGetCodeLengthComposed(t *testing.T) {
	cost := getCodeLength(model.XLexiconCoding, "\u00E9")
	expected := 14.375011301554892

	if cost != expected {
		t.Errorf("expected %v but got %v", expected, cost)
	}
}

func TestGetCodeLengthDecomposed(t *testing.T) {
	cost := getCodeLength(model.XLexiconCoding, "\u0065\u0301")
	expected := 16.653371479972932 // 16.65337147997293

	if cost != expected {
		t.Errorf("expected %v but got %v", expected, cost)
	}
}

func TestViterbiSegment(t *testing.T) {
	segments, score := viterbiSegment(model, "unfoobared", 0.0, 30)

	expectedSegments := []string{"un", "foo", "bar", "ed"}
	expectedScore := 32.684465337620665

	if !reflect.DeepEqual(segments, expectedSegments) || score != expectedScore {
		t.Errorf("unexpected result: %v, %v", segments, score)
	}
}

func TestViterbiSegmentComposed(t *testing.T) {
	segments, score := viterbiSegment(model, "brul\u00E9e", 0.0, 30)

	expectedSegments := []string{"bru", "l", "\u00E9", "e"}
	expectedScore := 109.47779723820601

	if !reflect.DeepEqual(segments, expectedSegments) {
		t.Errorf("unexpected segments: expected %v, got %v", expectedSegments, segments)
	}

	if score != expectedScore {
		t.Errorf("unexpected score: expected %v, got %v", expectedScore, score)
	}
}

func TestViterbiSegmentDecomposed(t *testing.T) {
	segments, score := viterbiSegment(model, "brul\u0065\u0301e", 0.0, 30)

	expectedSegments := []string{"brul\u0065", "\u0301", "e"}
	expectedScore := 118.92118396646775

	if !reflect.DeepEqual(segments, expectedSegments) {
		t.Errorf("unexpected segments: expected %v, got %v", expectedSegments, segments)
	}

	if score != expectedScore {
		t.Errorf("unexpected score: expected %v, got %v", expectedScore, score)
	}
}

func TestViterbiSegmentMaxLen(t *testing.T) {
	segments, score := viterbiSegment(model, "unsupervised", 0.0, 5)

	expectedSegments := []string{"un", "super", "vis", "ed"}
	expectedScore := 29.684031672881893

	if !reflect.DeepEqual(segments, expectedSegments) {
		t.Errorf("unexpected segments: expected %v, got %v", expectedSegments, segments)
	}

	if score != expectedScore {
		t.Errorf("unexpected score: expected %v, got %v", expectedScore, score)
	}
}

func TestUnicodeScalarBounds(t *testing.T) {
	bounds := unicodeScalarBounds("foo")
	expected := []int{1, 2, 3}

	if !reflect.DeepEqual(bounds, expected) {
		t.Errorf("expected %v but got %v", expected, bounds)
	}

	bounds = unicodeScalarBounds("l\u00E9l")
	expected = []int{1, 3, 4}

	if !reflect.DeepEqual(bounds, expected) {
		t.Errorf("expected %v but got %v", expected, bounds)
	}

	bounds = unicodeScalarBounds("l\u0065\u0301l")
	expected = []int{1, 2, 4, 5}

	if !reflect.DeepEqual(bounds, expected) {
		t.Errorf("expected %v but got %v", expected, bounds)
	}
}
