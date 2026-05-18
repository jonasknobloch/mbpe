package mbpe

import (
	"bufio"
	"container/heap"
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"go.jknobloc.com/x/dataset"
	"go.jknobloc.com/x/tui"
)

type MBPETrainer struct {
	preTokenizer    PreTokenizer
	segmenter       Segmenter
	segments        sync.Map
	alpha           float64
	model           *MBPE
	vocabSize       int
	dict            *Dict
	initialAlphabet map[string]struct{}
}

var changesPool = sync.Pool{
	New: func() any {
		return make(map[Pair]Change)
	},
}

func NewMBPETrainer(preTokenizer PreTokenizer, segmenter Segmenter, alpha float64, model *MBPE, vocabSize int, initialAlphabet map[string]struct{}) *MBPETrainer {
	if alpha < 0 || alpha > 1 {
		panic("alpha must be in [0, 1]")
	}

	return &MBPETrainer{
		preTokenizer:    preTokenizer,
		segmenter:       segmenter,
		segments:        sync.Map{},
		alpha:           alpha,
		model:           model,
		vocabSize:       vocabSize,
		dict:            NewDict(),
		initialAlphabet: initialAlphabet,
	}
}

func (t *MBPETrainer) Model() *MBPE {
	return t.model
}

func (t *MBPETrainer) Dict() *Dict {
	return t.dict
}

func (t *MBPETrainer) InitDict(reader dataset.Reader) error {
	var total int

	if n, err := reader.Num(); err != nil {
		return err
	} else {
		total = n
	}

	var completed atomic.Int64

	pb := tui.NewProgressBar("Pre-process files", 20, total, time.Now())

	texts := func(yield func(int, string) bool) {
		last := -1

		for k, v := range reader.Texts() {
			if last != -1 && k != last {
				completed.Add(1)
			}

			last = k

			if !yield(k, v) {
				return
			}
		}
	}

	pb.Start(1*time.Second, func() int {
		return int(completed.Load())
	})

	defer pb.Close()

	if err := t.dict.ProcessTexts(texts); err != nil {
		return err
	}

	if err := reader.Err(); err != nil {
		return err
	}

	completed.Add(1) // final document

	return nil
}

func (t *MBPETrainer) LoadDict(name string) error {
	return t.dict.Load(name)
}

func (t *MBPETrainer) SaveSegments(w io.Writer) error {
	m := make(map[string][]string)

	t.segments.Range(func(key, value any) bool {
		k := key.(string)
		v := value.([]string)

		m[k] = v

		return true
	})

	enc := gob.NewEncoder(w)

	if err := enc.Encode(m); err != nil {
		return err
	}

	return nil
}

func (t *MBPETrainer) LoadSegments(r io.Reader) error {
	m := make(map[string][]string)

	dec := gob.NewDecoder(r)

	if err := dec.Decode(&m); err != nil {
		return err
	}

	for k, v := range m {
		t.segments.Store(k, v)
	}

	return nil
}

func (t *MBPETrainer) AddToken(left, right string) {
	t.model.AddToken(left + right)
	t.model.AddMerge(left, right)
}

func (t *MBPETrainer) Train() {
	t.model.InitVocab(t.vocabSize)

	t.InitVocab()

	k := t.model.Cap() - t.model.Len()

	t.model.InitMerges(k)

	chunks := t.dict.Items()

	if t.segmenter != nil {
		pbSplit := tui.NewProgressBar("Segment chunks", 20, len(chunks), time.Now())

		pbSplit.Start(1*time.Second, func() int {
			return pbSplit.Completed()
		})

		var wg sync.WaitGroup

		sem := make(chan struct{}, runtime.NumCPU())

		for i := range chunks {
			wg.Add(1)

			sem <- struct{}{}

			go func(i int) {
				defer wg.Done()

				defer func() {
					<-sem
				}()

				var segments []string

				if v, ok := t.segments.Load(chunks[i].src); ok {
					segments = v.([]string)
				} else {
					segments = SegmentWithoutPrefixWhitespace(chunks[i].src, t.segmenter)

					t.segments.Store(chunks[i].src, segments)
				}

				chunks[i].Split(segments)
				chunks[i].Alpha(t.alpha)

				pbSplit.Add(1)
			}(i)
		}

		wg.Wait()

		pbSplit.Close()
	}

	var mergeWeights = make(map[Pair]float64) // pair_counts
	var pairPositions = make(map[Pair][]int)  // where_to_update

	pbPairs := NewProgressBar("Initialize pairs", 20, len(chunks), time.Now())

	epsilon := float64(0)

	weights := make(map[Pair]float64)

	for i, chunk := range chunks {
		clear(weights)

		epsilon += chunk.PairWeights(weights)

		for pair, w := range weights {
			mergeWeights[pair] += w
			pairPositions[pair] = append(pairPositions[pair], i)
		}

		pbPairs.Increment()
		pbPairs.Print()
	}

	pbPairs.Finish()

	queue := NewQueue(make([]Merge, 0))

	for pair, positions := range pairPositions {
		ids := [2]int{
			t.model.atoi[pair[0]],
			t.model.atoi[pair[1]],
		}

		heap.Push(queue, Merge{
			pair:      pair,
			ids:       ids,
			weight:    mergeWeights[pair],
			positions: positions,
		})

		delete(pairPositions, pair)
	}

	pbMerges := NewProgressBar("Compute merges", 20, t.vocabSize-t.model.Len(), time.Now())

	for t.model.Len() < t.vocabSize && queue.Len() != 0 {
		top := heap.Pop(queue).(Merge)

		if top.weight != mergeWeights[top.pair] {
			top.weight = mergeWeights[top.pair]

			queue.Push(top)

			continue
		}

		// dumpQueue(*queue, top, mergeWeights)

		if top.weight < 1 {
			break
		}

		t.AddToken(top.pair[0], top.pair[1])

		changes := changesPool.Get().(map[Pair]Change)

		for _, position := range top.positions {
			clear(changes)

			delta := chunks[position].TrackedMerge(top, changes)

			epsilon += delta

			for pair, change := range changes {
				mergeWeights[pair] += change.delta

				// After updating counts (weights), we need to collect postions for all potential merges.
				// With alpha set to zero (reference), checking if delta > 0 is sufficient, since pair counts
				// can only decrease - meaning only new pairs have a positive count.
				// In fact the Hugging Face implementation checks for count > 0.

				// Alpha > 0 enables positive weight updates, as well as the removal and addition of new pairs
				// with a zero count. Sticking with checking for count > 0 could then lead to queueing duplicated
				// merge operations.

				if change.remove || change.update {
					continue // don't queue removals and weight updates
				}

				pairPositions[pair] = append(pairPositions[pair], position)
			}
		}

		for pair := range pairPositions {
			id := [2]int{
				t.model.atoi[pair[0]],
				t.model.atoi[pair[1]],
			}

			merge := Merge{
				pair:      pair,
				ids:       id,
				weight:    mergeWeights[pair],
				positions: pairPositions[pair],
			}

			heap.Push(queue, merge)

			delete(pairPositions, pair)
		}

		pbMerges.Increment()
		pbMerges.Print()
	}

	pbMerges.Finish()

	fmt.Println("\nepsilon", epsilon)
}

func (t *MBPETrainer) InitVocab() {
	tokens := make(map[string]int)

	// TODO side effects of byte replacements on strings.Split(..., "")

	for _, chunk := range t.dict.items {
		for _, token := range chunk.Tokens() {
			if _, ok := tokens[token]; !ok {
				tokens[token] = 0

				// fmt.Printf("discovered new token %s %02x \n", token, []byte(token))
			}

			tokens[token]++
		}
	}

	if t.initialAlphabet != nil {
		for token := range t.initialAlphabet {
			tokens[token] = 0
		}
	}

	alphabet := make([]string, 0, len(tokens))

	for token := range tokens {
		alphabet = append(alphabet, token)
	}

	sort.Strings(alphabet)

	for _, token := range alphabet {
		t.model.AddToken(token)
	}
}

func saveMergeFrame(name string, frame []Merge) error {
	if _, err := os.Stat("merge_frames"); os.IsNotExist(err) {
		_ = os.Mkdir("merge_frames", os.ModePerm)
	}

	if err := toFile(name, func(writer *bufio.Writer) error {
		for _, merge := range frame {
			if _, err := writer.WriteString(fmt.Sprintf("%s (%d) %s (%d) %f\n", merge.pair[0], merge.ids[0], merge.pair[1], merge.ids[1], merge.weight)); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return err
	}

	return nil
}

func dumpQueue(queue Queue, top Merge, mergeWeights map[Pair]float64) {
	list := make([]Merge, queue.Len())

	copy(list, queue)

	list = append(list, top) // TODO just use heap helpers instead of sorting the underlying slice

	for _, merge := range list {
		if merge.weight != mergeWeights[merge.pair] {
			merge.weight = mergeWeights[merge.pair]
		}
	}

	sort.Slice(list, func(i, j int) bool {
		return list[j].Less(list[i])
	})

	fmt.Println()

	// _ = saveMergeFrame(fmt.Sprintf("merge_frames/frame_%d.txt", time.Now().UnixNano()), list[0:10])

	for i, merge := range list {
		if i == 10 {
			break
		}

		fmt.Printf("%s (%d) %s (%d) %f\n", merge.pair[0], merge.ids[0], merge.pair[1], merge.ids[1], merge.weight)
	}
}
