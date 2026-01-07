package main

import (
	"bufio"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/jonasknobloch/mbpe"
	"golang.org/x/image/colornames"
)

func main() {
	// tokenize()
	// eval()
	// train()
	// server()
	// serialize()
}

func eval() {
	base := "out/morfessor"

	var tokenizers []string

	if paths, err := subDirs(base); err != nil {
		log.Fatal(err)
	} else {
		tokenizers = paths
	}

	initTokenizer := func(vocab, merges string) *mbpe.Tokenizer {
		model := mbpe.NewMBPE()

		if err := model.Load(vocab, merges); err != nil {
			log.Fatal(err)
		}

		tokenizer := mbpe.NewTokenizer(model)

		byteLevel := mbpe.NewByteLevel(true)

		tokenizer.SetPreTokenizer(byteLevel)
		tokenizer.SetDecoder(byteLevel)

		return tokenizer
	}

	runner := mbpe.NewRunner()

	runner.SetFormat("%.4f")

	for _, name := range tokenizers {
		runner.AddTokenizer(*initTokenizer(filepath.Join(name, "vocab.json"), filepath.Join(name, "merges.txt")), filepath.Base(name))
	}

	bpr := func() mbpe.Evaluator {
		bprEval := mbpe.NewBPREvaluator()

		if err := bprEval.LoadSegmentations("data/mbpe/goldstd_trainset.segmentation.eng.tsv"); err != nil {
			log.Fatal(err)
		}

		return bprEval
	}()

	ml := func() mbpe.Evaluator {
		mlEval := mbpe.NewMergeLayerEvaluator()

		if err := mlEval.LoadSegmentations("data/mbpe/goldstd_trainset.segmentation.eng.tsv"); err != nil {
			log.Fatal(err)
		}

		return mlEval
	}()

	fert := func() mbpe.Evaluator {
		fertilityEval := mbpe.NewFertilityEvaluator()

		if err := fertilityEval.InitDict("data/babyllm/test/all.test"); err != nil {
			log.Fatal(err)
		}

		return fertilityEval
	}()

	ref := func() mbpe.Evaluator {
		refEval := mbpe.NewReferenceEvaluator()

		if err := refEval.LoadModel(filepath.Join(tokenizers[0], "vocab.json"), filepath.Join(tokenizers[0], "merges.txt")); err != nil {
			log.Fatal(err)
		}

		return refEval
	}()

	wug := func() mbpe.Evaluator {
		wugEval := mbpe.NewAdjectiveSuffixEvaluator()

		// wugEval.SetReportAverage(true)

		return wugEval
	}

	runner.AddEvaluator(bpr, "Boundary Precision Recall")
	runner.AddEvaluator(ml, "Merge Layer")
	runner.AddEvaluator(fert, "Fertility")
	runner.AddEvaluator(ref, "Reference Overlap")
	runner.AddEvaluator(wug(), "Nonce Adjectives")

	baseline := mbpe.NewRunner()

	baseline.AddTokenizer(*initTokenizer(filepath.Join(tokenizers[0], "vocab.json"), filepath.Join(tokenizers[0], "merges.txt")), filepath.Base(tokenizers[0]))

	baseline.AddEvaluator(ml, "Merge Layer")
	baseline.AddEvaluator(fert, "Fertility")

	md00, raw00 := runner.RunAll(100512)
	md01, raw01 := runner.RunAll(50256)
	md02, raw02 := runner.RunAll(32768)
	md03, raw03 := runner.RunAll(16384)
	md04, raw04 := runner.RunAll(8192)

	_, rawBase := baseline.RunAll(105000, 100512, 90000, 80000, 70000, 60000, 50000, 40000, 32768, 20000, 16384, 8192, 5000)

	s00 := mbpe.NewPlotData(mbpe.SelectColumn(raw00[2], 0), mbpe.SelectColumn(raw00[1], 0), true, false, "100512", colornames.Red)
	s01 := mbpe.NewPlotData(mbpe.SelectColumn(raw01[2], 0), mbpe.SelectColumn(raw01[1], 0), true, false, "50256", colornames.Green)
	s02 := mbpe.NewPlotData(mbpe.SelectColumn(raw02[2], 0), mbpe.SelectColumn(raw02[1], 0), true, false, "32768", colornames.Blue)
	s03 := mbpe.NewPlotData(mbpe.SelectColumn(raw03[2], 0), mbpe.SelectColumn(raw03[1], 0), true, false, "16384", colornames.Purple)
	s04 := mbpe.NewPlotData(mbpe.SelectColumn(raw04[2], 0), mbpe.SelectColumn(raw04[1], 0), true, false, "8192", colornames.Brown)

	sBase := mbpe.NewPlotData(mbpe.SelectColumn(rawBase[1], 0), mbpe.SelectColumn(rawBase[0], 0), false, true, "baseline", colornames.Black)

	data := []mbpe.PlotData{s00, s01, s02, s03, s04, sBase}

	mbpe.Plot(data, [2]float64{1.00, 1.2}, [2]float64{0.75, 0.90}, "Fertility", "Merge Layer")

	fmt.Printf(md00)
	fmt.Println()
	fmt.Printf(md01)
	fmt.Println()
	fmt.Printf(md02)
	fmt.Println()
	fmt.Println(md03)
	fmt.Println()
	fmt.Println(md04)
}

func tokenize() {
	model := mbpe.NewMBPE()

	// err := model.Load("out-m100/00-en-m000/vocab.json", "out-m100/00-en-m000/merges.txt")
	// err := model.Load("out-m100/01-en-m010/vocab.json", "out-m100/01-en-m010/merges.txt")
	// err := model.Load("out-m100/02-en-m020/vocab.json", "out-m100/02-en-m020/merges.txt")
	// err := model.Load("out-m100/03-en-m030/vocab.json", "out-m100/03-en-m030/merges.txt")
	// err := model.Load("out-m100/04-en-m040/vocab.json", "out-m100/04-en-m040/merges.txt")
	// err := model.Load("out-m100/05-en-m050/vocab.json", "out-m100/05-en-m050/merges.txt")
	// err := model.Load("out-m100/06-en-m060/vocab.json", "out-m100/06-en-m060/merges.txt")
	// err := model.Load("out-m100/07-en-m070/vocab.json", "out-m100/07-en-m070/merges.txt")
	// err := model.Load("out-m100/08-en-m080/vocab.json", "out-m100/08-en-m080/merges.txt")
	// err := model.Load("out-m100/09-en-m090/vocab.json", "out-m100/09-en-m090/merges.txt")
	// err := model.Load("out-m100/10-en-m100/vocab.json", "out-m100/10-en-m100/merges.txt")

	err := model.Load("out-m100/05-en-m050/vocab.json", "out-m100/05-en-m050/merges.txt")
	// err := model.Load("out-m100/10-en-m100/vocab.json", "out-m100/10-en-m100/merges.txt")

	if err != nil {
		log.Fatal(err)
	}
	//
	// if err := SerializeModel(model, "en-m100.gob"); err != nil {
	// 	log.Fatal(err)
	// }

	// var model *MBPE
	//
	// if m, err := DeserializeModel(m000); err != nil {
	// 	log.Fatal(err)
	// } else {
	// 	model = m
	// }

	tokenizer := mbpe.NewTokenizer(model)

	byteLevel := mbpe.NewByteLevel(true)

	tokenizer.SetPreTokenizer(byteLevel)
	tokenizer.SetDecoder(byteLevel)

	ids := tokenizer.Tokenize(" airsickness")
	tokens := model.ToString(ids)

	fmt.Println(ids)
	fmt.Println(tokens)

	fmt.Println(tokenizer.Decoder().Decode(tokens))
}

func segmentFile(name string, vocabSize int) {
	model := mbpe.NewMBPE()

	tokenizer := mbpe.NewTokenizer(model)

	byteLevel := mbpe.NewByteLevel(true)

	tokenizer.SetPreTokenizer(byteLevel)
	tokenizer.SetDecoder(byteLevel)

	if err := model.Load("out/00-en-base/vocab.json", "out/00-en-base/merges.txt"); err != nil {
		log.Fatal(err)
	}

	compounds := make([]string, 0)

	if err := mbpe.ReadTsv(name, func(record []string) error {
		if len(record) == 0 {
			return errors.New("unexpected number of fields")
		}

		compounds = append(compounds, " "+strings.TrimLeft(record[0], " "))

		return nil
	}); err != nil {
		log.Fatal(err)
	}

	segmentations := make([][]string, len(compounds))

	maxRank := -1

	if vocabSize > -1 {
		maxRank = vocabSize - len(model.Alphabet())
	}

	for i, compound := range compounds {
		segmentation, ok := mbpe.GetTokenizerSegmentation(*tokenizer, compound, maxRank)

		if !ok {
			continue
		}

		segmentations[i] = segmentation
	}

	if err := mbpe.ToFile("segmentations.txt", func(writer *bufio.Writer) error {
		for i, segmentation := range segmentations {
			if _, err := writer.WriteString(fmt.Sprintf("%s\t%s\n", strings.TrimLeft(compounds[i], " "), strings.TrimLeft(strings.Join(segmentation, " "), " "))); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		log.Fatal(err)
	}
}

func train() {
	out := "out/morfessor"

	morfessor := func(alpha float64) mbpe.Segmenter {
		m := mbpe.NewMorfessor(alpha)

		if err := m.LoadModel("data/morfessor/semisup_model.proto"); err != nil {
			log.Fatal(err)
		}

		return m
	}

	// mbpe.InvertWeightFunction = true

	m000 := morfessor(0.0)
	m010 := morfessor(0.1)
	m020 := morfessor(0.2)
	m030 := morfessor(0.3)
	m040 := morfessor(0.4)
	m050 := morfessor(0.5)
	m060 := morfessor(0.6)
	m070 := morfessor(0.7)
	m080 := morfessor(0.8)
	m090 := morfessor(0.9)
	m100 := morfessor(1.0)

	newTrainer := func(segmenter mbpe.Segmenter) *mbpe.MBPETrainer {
		return mbpe.NewMBPETrainer(mbpe.NewByteLevel(true), segmenter, mbpe.NewMBPE(), 1<<17)
	}

	trainers := []struct {
		*mbpe.MBPETrainer
		string
	}{
		{newTrainer(m000), "m000_babylm"},
		{newTrainer(m010), "m010_babylm"},
		{newTrainer(m020), "m020_babylm"},
		{newTrainer(m030), "m030_babylm"},
		{newTrainer(m040), "m040_babylm"},
		{newTrainer(m050), "m050_babylm"},
		{newTrainer(m060), "m060_babylm"},
		{newTrainer(m070), "m070_babylm"},
		{newTrainer(m080), "m080_babylm"},
		{newTrainer(m090), "m090_babylm"},
		{newTrainer(m100), "m100_babylm"},
	}

	for i, t := range trainers {
		dict := filepath.Join(out, "dict.txt")

		raw := []string{
			"data/babyllm/train_100M/bnc_spoken.train",
			"data/babyllm/train_100M/childes.train",
			"data/babyllm/train_100M/gutenberg.train",
			"data/babyllm/train_100M/open_subtitles.train",
			"data/babyllm/train_100M/simple_wiki.train",
			"data/babyllm/train_100M/switchboard.train",
		}

		if err := t.LoadDict(dict); err != nil {
			if err := t.InitDict(raw...); err != nil {
				log.Fatal(err)
			}

			if err := t.Dict().Save(dict); err != nil {
				log.Fatal(err)
			}
		}

		if i > 0 {
			fmt.Println()
		}

		fmt.Printf("%s\n\n", t.string)

		t.Train()

		dir := filepath.Join(out, t.string)

		if err := os.Mkdir(dir, 0755); err != nil && !errors.Is(err, os.ErrExist) {
			log.Fatal(err)
		}

		if err := t.Model().Save(filepath.Join(dir, "vocab.json"), filepath.Join(dir, "merges.txt")); err != nil {
			log.Fatal(err)
		}
	}
}

func serialize() {
	base := "out/morfessor"

	var paths []string

	if ps, err := subDirs(base); err != nil {
		log.Fatal(err)
	} else {
		paths = ps
	}

	steps := []int{100512, 50256, 32768, 16384, 8192}

	for _, step := range steps {
		for _, path := range paths {
			model := mbpe.NewMBPE()

			if err := model.Load(filepath.Join(path, "vocab.json"), filepath.Join(path, "merges.txt")); err != nil {
				log.Fatal(err)
			}

			model.Trim(step)

			dir := fmt.Sprintf("tokenizer_gpt2_%d_%s_v2", step, filepath.Base(path))

			out := filepath.Join("tokenizers", dir)

			if err := os.Mkdir(out, os.ModePerm); err != nil {
				log.Fatal(err)
			}

			if err := model.Save(filepath.Join(out, "vocab.json"), filepath.Join(out, "merges.txt")); err != nil {
				log.Fatal(err)
			}

			var config string
			var special string

			if bs, err := os.ReadFile("scripts/saved_tokenizer_gpt2/tokenizer_config.json"); err != nil {
				log.Fatal(bs)
			} else {
				config = string(bs)
			}

			if bs, err := os.ReadFile("scripts/saved_tokenizer_gpt2/special_tokens_map.json"); err != nil {
				log.Fatal(err)
			} else {
				special = string(bs)
			}

			config = strings.Replace(config, "50256", fmt.Sprintf("%d", step), -1)

			if err := os.WriteFile(filepath.Join(out, "tokenizer_config.json"), []byte(config), os.ModePerm); err != nil {
				log.Fatal(err)
			}

			if err := os.WriteFile(filepath.Join(out, "special_tokens_map.json"), []byte(special), os.ModePerm); err != nil {
				log.Fatal(err)
			}
		}
	}
}

func subDirs(base string) ([]string, error) {
	paths := make([]string, 0)

	err := filepath.WalkDir(base, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(base, path)

		if err != nil {
			return err
		}

		depth := strings.Count(rel, string(os.PathSeparator))

		if d.IsDir() && rel != "." && depth == 0 {
			paths = append(paths, path)
		}

		return nil
	})

	return paths, err
}
