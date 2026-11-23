# Third-party dataset notices

The experiments use public vector embeddings from the [VIBE dataset](https://huggingface.co/datasets/vector-index-bench/vibe), revision `07b387891a221b7b073b83d2f752b76462e5fa03`, distributed by vector-index-bench under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The VIBE code is [MIT licensed](https://github.com/vector-index-bench/vibe/blob/main/LICENSE). This project selects subsets, groups exact duplicate vectors, and splits the corpus into training, validation, and test sets; it does not modify vector values. The source files and generated data are not committed.

- `arxiv_nomic`: embeddings of arXiv descriptive metadata, including abstracts. [arXiv metadata terms](https://info.arxiv.org/help/api/tou.html) identify the descriptive metadata as CC0. VIBE used [Nomic Embed Text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), Apache-2.0.
- `landmark_nomic`: embeddings of the Google Landmarks index set. [Google Landmarks](https://github.com/cvdfoundation/google-landmark#index-image-licenses) describes the index images as CC0/public domain. VIBE used [Nomic Embed Vision v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5), Apache-2.0.

The embedding datasets' CC BY terms apply to the distributed embeddings; the upstream source and model references above supply the attribution and origin. The fixed file sizes and SHA-256 values are recorded in `configs/experiments/e_means/sources.yaml` and each prepared dataset manifest.
