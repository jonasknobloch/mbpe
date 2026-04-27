package mbpe

import (
	"bufio"
	"fmt"
	"iter"
	"path"
	"runtime"
	"sort"
	"sync"

	"go.jknobloc.com/x/dataset"
)

type Dict struct {
	mutex sync.RWMutex
	items []Chunk
	lut   map[string]int
	lines int
}

func NewDict() *Dict {
	return &Dict{
		items: make([]Chunk, 0),
		lut:   make(map[string]int),
	}
}

func (d *Dict) Items() []Chunk {
	return d.items
}

func (d *Dict) Lines() int {
	return d.lines
}

func (d *Dict) ProcessTexts(texts iter.Seq2[int, string]) error {
	jobs := make(chan string)

	var wg sync.WaitGroup

	maxWorkers := runtime.NumCPU()

	for w := 0; w < maxWorkers; w++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for line := range jobs {
				preTokenizer := NewByteLevel(true)

				chunks := preTokenizer.PreTokenize(line)
				results := make([]Chunk, 0, len(chunks))

				for _, chunk := range chunks {
					results = append(results, *NewChunk(chunk, 1, nil, 0))
				}

				d.mutex.Lock()

				for _, chunk := range results {
					i, ok := d.lut[chunk.src]

					if !ok {
						d.lut[chunk.src] = len(d.items)
						d.items = append(d.items, chunk)

						continue
					}

					d.items[i].n += chunk.n
				}

				d.lines++

				d.mutex.Unlock()
			}
		}()
	}

	for _, text := range texts {
		jobs <- text
	}

	close(jobs)

	wg.Wait()

	return nil
}

// Deprecated: Use ProcessTexts with an appropriate reader instead.
func (d *Dict) ProcessFiles(names ...string) error {
	for _, name := range names {
		dir := path.Dir(name)
		base := path.Base(name)

		var reader dataset.Reader

		if r, err := dataset.NewFileReader(dir, base); err != nil {
			return err
		} else {
			reader = r
		}

		if err := d.ProcessTexts(reader.Texts()); err != nil {
			return err
		}
	}

	return nil
}

func (d *Dict) Save(name string) error {
	items := make([]Chunk, len(d.items))

	copy(items, d.items)

	sort.Slice(items, func(i, j int) bool {
		if items[i].n != items[j].n {
			return items[i].n > items[j].n
		}

		return items[i].src < items[j].src
	})

	if err := toFile(name, func(writer *bufio.Writer) error {
		for _, chunk := range items {
			if _, err := writer.WriteString(fmt.Sprintf("%s %d\n", chunk.src, chunk.n)); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return err
	}

	return nil
}

func (d *Dict) Load(name string) error {
	return fromFile(name, func(scanner *bufio.Scanner) error {
		for scanner.Scan() {
			line := scanner.Text()

			if err := scanner.Err(); err != nil {
				return err
			}

			var s string
			var n int

			if _, err := fmt.Sscanf(line, "%s %d", &s, &n); err != nil {
				return err
			}

			chunk := NewChunk(s, n, nil, 0)

			d.lut[chunk.src] = len(d.items)
			d.items = append(d.items, *chunk)
		}

		return nil
	})
}
