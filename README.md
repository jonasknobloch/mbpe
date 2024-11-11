# Morphologically Biased Byte-Pair Encoding

mBPE acts as an extension to the [huggingface/tokenizers](https://github.com/huggingface/tokenizers) library and is
designed to enhance segmentations produced by the byte-pair encoding tokenization algorithm[^1]. Byte-pair encoding has
been shown to poorly approximate morphological boundaries[^2], which is especially problematic for morphologically rich
language. By incorporating morphological knowledge into the pre-tokenization process, we aim to improve the quality of
produced segmentations through an induced bias towards morphologically motivated sub-word boundaries.

[^1]: [Neural Machine Translation of Rare Words with Subword Units](https://doi.org/10.48550/arXiv.1508.07909)

[^2]: [Byte Pair Encoding is Suboptimal for Language Model Pretraining](https://doi.org/10.48550/arXiv.2004.03720)

Pre-trained tokenizers and models are available on [Hugging Face](https://huggingface.co/jonasknobloch).

* [gpt2_cx-en_00000-00000_50k](https://huggingface.co/jonasknobloch/gpt2_cx-cs_00000-00019_50k)
* [gpt2+ts_cx-en_00000-00000_50k](https://huggingface.co/jonasknobloch/gpt2-ts_cx-en_00000-00009_50k)
* [gpt2+morf_u0-30-50-x_cx-en_00000-00000_50k](https://huggingface.co/jonasknobloch/gpt2-morf_u0-30-50-x_cx-en_00000-00009_50k)
* [gpt2+morf_s0-30-x-2_cx-en_00000-00000_50k](https://huggingface.co/jonasknobloch/gpt2-morf_s0-30-x-2_cx-en_00000-00009_50k)

## Pre-Tokenizers

### External

The external pre-tokenizer enables the integration custom pre-tokenization algorithms via a socket connection.
Tokenization parallelism should be disabled by setting `TOKENIZERS_PARALLELISM=true`. Note that disabling parallelism
will slow down tokenization significantly. See [jonasknobloch/unimorph](https://github.com/jonasknobloch/unimorph)
for a reference server implementation.

### Tree-Split

The tree-split pre-tokenizer introduces additional boundaries by clustering inflected word forms retrieved from
[UniMorph](https://unimorph.github.io)[^3] dictionaries. Form clusters are aligned by constructing a suffix tree for each
cluster. New boundaries are then introduced by traversing the trees and introducing boundaries at nodes with multiple children.

[^3]: [UniMorph 4.0: Universal Morphology](https://doi.org/10.48550/arXiv.2205.03608)

### Morfessor

The Morfessor pre-tokenizer introduces additional boundaries retrieved using an arbitrary
[Morfessor](http://morpho.aalto.fi/projects/morpho/morfessor2.shtml)[^4][^5] model. Trained Morfessor models need to be
converted using the provided protobuf definition and conversion script

[^4]: [Unsupervised Discovery of Morphemes](https://doi.org/10.48550/arXiv.cs/0205057)

[^5]: [Morfessor 2.0: Python Implementation and Extensions for Morfessor Baseline](https://urn.fi/URN:ISBN:978-952-60-5501-5)

## Intrinsic Metrics

### Tokenizer Fertility

| tokenizer                                  | compounds | fertility |
|--------------------------------------------|-----------|-----------|
| gpt2_cx-en_00000-00000_50k                 | 4992469   | **1.32**  |
| gpt2+ts_cx-en_00000-00000_50k              | 4923123   | 1.40      |
| gpt2+morf_u0-30-50-x_cx-en_00000-00000_50k | 3630703   | 1.42      |
| gpt2+morf_s0-30-x-2_cx-en_00000-00000_50k  | 99191     | 1.69      |

### Boundary Precision and Recall

| tokenizer                                  | P        | R        | F1       |
|--------------------------------------------|----------|----------|----------|
| gpt2_cx-en_00000-00000_50k                 | 0.33     | 0.56     | 0.42     |
| gpt2+ts_cx-en_00000-00000_50k              | 0.40     | 0.58     | 0.47     |
| gpt2+morf_u0-30-50-x_cx-en_00000-00000_50k | 0.45     | **0.61** | 0.52     |
| gpt2+morf_s0-30-x-2_cx-en_00000-00000_50k  | **0.56** | 0.59     | **0.57** |
