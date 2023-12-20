use morphy::pre_tokenizers::PreTokenizerWrapper;
use morphy::pre_tokenizers::external::External;
use morphy::pre_tokenizers::sequence::Sequence;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

pub fn main() {
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::from(ByteLevel::default()),
        PreTokenizerWrapper::from(External {
            path_to_socket: "/tmp/unimorph.sock".to_string(),
            ignored_prefix: "Ġ".to_string(),
            split_delimiter: '#',
        }),
    ]);

    let mut pre_tokenized = PreTokenizedString::from("That's some impressive retrofitting!");

    let _ = pre_tokenizer.pre_tokenize(&mut pre_tokenized);

    println!(
        "{:?}",
        pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte)
    );
}
