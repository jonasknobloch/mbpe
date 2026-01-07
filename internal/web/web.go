//go:build wasm

package web

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"syscall/js"

	"github.com/jonasknobloch/mbpe"
)

type WToken struct {
	Token string
	ID    int
}

type WResult struct {
	Raw           string
	Segmentations [][]WToken
	Mermaid       string
}

//go:embed en-m000.gob
var m000 []byte

//go:embed en-m010.gob
var m010 []byte

//go:embed en-m020.gob
var m020 []byte

//go:embed en-m030.gob
var m030 []byte

//go:embed en-m040.gob
var m040 []byte

//go:embed en-m050.gob
var m050 []byte

//go:embed en-m060.gob
var m060 []byte

//go:embed en-m070.gob
var m070 []byte

//go:embed en-m080.gob
var m080 []byte

//go:embed en-m090.gob
var m090 []byte

//go:embed en-m100.gob
var m100 []byte

func WrapTokenizeWeb() js.Func {
	return js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) < 3 {
			log.Fatal("not enough arguments")
		}

		input := args[0].String()
		modelChoice := args[1].String()
		vocabSize := args[2].Int()

		return WTokenize(input, modelChoice, vocabSize)
	})
}

func WTokenize(input string, modelChoice string, vocabSize int) js.Value {
	modelMapping := map[string][]byte{
		"m000": m000,
		"m010": m010,
		"m020": m020,
		"m030": m030,
		"m040": m040,
		"m050": m050,
		"m060": m060,
		"m070": m070,
		"m080": m080,
		"m090": m090,
		"m100": m100,
	}

	serialized, ok := modelMapping[modelChoice]

	if !ok {
		log.Fatal("invalid model choice")
	}

	result, err := WTokenizeWithSerialized(input, serialized, vocabSize)

	if err != nil {
		log.Fatal(err)
	}

	jsonData, err := json.Marshal(result)

	if err != nil {
		log.Fatal(err)
	}

	return js.ValueOf(string(jsonData))
}

func WTokenizeWithSerialized(input string, serialized []byte, vocabSize int) ([]WResult, error) {
	var model *mbpe.MBPE

	if m, err := mbpe.DeserializeModel(serialized); err != nil {
		return nil, err
	} else {
		model = m
	}

	tokenizer := mbpe.NewTokenizer(model)

	byteLevel := mbpe.NewByteLevel(true)

	tokenizer.SetPreTokenizer(byteLevel)
	tokenizer.SetDecoder(byteLevel)

	chunks := tokenizer.PreTokenizer().PreTokenize(input)

	result := make([]WResult, len(chunks))

	maxRank := -1

	if vocabSize > -1 {
		alphabet := model.Alphabet()

		if !(vocabSize < len(alphabet)) && !(vocabSize > len(model.Vocab())) {
			maxRank = vocabSize - len(alphabet)
		}
	}

	for i, chunk := range chunks {
		layers := model.TokenizeLayered(chunk, maxRank)

		result[i] = WResult{
			Raw:           chunk,
			Segmentations: WLayersToSegmentations(layers, model),
			Mermaid:       WLayersToMermaid(layers, model),
		}
	}

	return result, nil
}

func WLayersToSegmentations(layers [][]int, model *mbpe.MBPE) [][]WToken {
	segmentations := make([][]WToken, len(layers))

	for i, layer := range layers {
		segmentations[i] = make([]WToken, len(layer))

		tokens := model.ToString(layer)

		for j, id := range layer {
			segmentations[i][j] = WToken{
				Token: tokens[j],
				ID:    id,
			}
		}
	}

	return segmentations
}

func WLayersToMermaid(layers [][]int, model *mbpe.MBPE) string {
	nodes := make([][2]int, 0)
	edges := make([][2][2]int, 0)

	positions := make([][][2]int, len(layers))

	for i, layer := range layers {
		positions[i] = make([][2]int, 0)

		if i == 0 {
			for j := range layer {
				positions[i] = append(positions[i], [2]int{i, j})
				nodes = append(nodes, [2]int{i, j})
			}

			continue
		}

		for j, token := range layer {
			if token != layers[i-1][j] {
				break
			}

			positions[i] = append(positions[i], positions[i-1][j])
		}

		k := len(positions[i])

		positions[i] = append(positions[i], [2]int{i, k})

		nodes = append(nodes, [2]int{i, k})

		edges = append(edges, [2][2]int{positions[i-1][k], positions[i][k]})
		edges = append(edges, [2][2]int{positions[i-1][k+1], positions[i][k]})

		if len(positions[i]) < len(positions[i-1])-1 {
			positions[i] = append(positions[i], positions[i-1][k+2:]...)
		}
	}

	root := [2]int{len(positions), 0}

	nodes = append(nodes, root)

	for _, foo := range positions[len(positions)-1] {
		edges = append(edges, [2][2]int{foo, root})
	}

	var sb strings.Builder

	sb.WriteString("graph TD\n")

	for _, node := range nodes {
		label := "ROOT"

		if node[0] < len(layers) {
			label = model.ToString([]int{layers[node[0]][node[1]]})[0]
		}

		sb.WriteString(fmt.Sprintf("%d+%d[%s]\n", node[0], node[1], label))
	}

	for _, edge := range edges {
		sb.WriteString(fmt.Sprintf("%d+%d-%s>%d+%d\n", edge[1][0], edge[1][1], strings.Repeat("-", edge[1][0]-edge[0][0]), edge[0][0], edge[0][1]))
	}

	return sb.String()
}
