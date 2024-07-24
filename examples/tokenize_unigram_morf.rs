use tokenizers::models::unigram::Unigram;
use tokenizers::models::unigram::UnigramTrainerBuilder;
use tokenizers::{AddedToken, DecoderWrapper, Model, ModelWrapper, NormalizerWrapper, PostProcessorWrapper, Result, TokenizerBuilder};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use mbpe::pre_tokenizers::{morfessor, PreTokenizerWrapper};
use mbpe::pre_tokenizers::morfessor::MorfessorConfig;
use mbpe::pre_tokenizers::sequence::Sequence;

fn main() -> Result<()> {
    let morfessor = morfessor::new_pre_tokenizer(
        true,
        true,
        "scripts/unsup_model.proto",
        MorfessorConfig::default(),
    );

    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::from(morfessor),
        PreTokenizerWrapper::from(ByteLevel::new(false, true, false)),
    ]);

    let mut tokenizer = TokenizerBuilder::<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
        .with_model(ModelWrapper::from(Unigram::default()))
        .with_pre_tokenizer(Some(PreTokenizerWrapper::from(pre_tokenizer)))
        .with_post_processor(Some(PostProcessorWrapper::from(ByteLevel::default())))
        .with_decoder(Some(DecoderWrapper::from(ByteLevel::default())))
        .build()?;

    let mut trainer = TrainerWrapper::from(
        UnigramTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(50256)
            .build()?
    );

    tokenizer.train_from_files(&mut trainer, vec!["data/tiny_shakespeare.txt".to_string()])?;

    let end_of_text = AddedToken::from(String::from("<|endoftext|>"), true);

    tokenizer.add_special_tokens(&[end_of_text]);

    tokenizer.save("tokenizer_unigram+morf_tiny_shakespeare_50k.json", false)?;

    Ok(())
}