//! Compression algorithms.

pub mod crc32;
pub mod deflate;
pub mod adler32;
pub mod huffman;
pub mod lz77;

pub use crc32::crc32;
pub use deflate::deflate;
pub use adler32::adler32;
