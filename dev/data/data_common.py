"""
Common utilities for the datasets
"""

import requests
from tqdm import tqdm
import numpy as np


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def write_evalfile(filename, datas, model_desc="gpt-2"):
    """
    Saves eval data as a .bin file, for reading in C.
    Used for multiple-choice style evals, e.g. HellaSwag and MMLU
    - First comes a header with 256 int32s
    - The examples follow, each example is a stream of uint16_t (gpt-2) or uint32_t (llama 3):
        - <START_EXAMPLE> delimiter of 2**bits - 1, i.e. 65,535 for gpt2
        - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        - <EXAMPLE_INDEX>, the index of the example in the dataset
        - <LABEL>, the index of the correct completion
        - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
        - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
    """
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    version = {
        "gpt-2": 1,
        "llama-3": 2
    }[model_desc]
    bits = {
        "gpt-2": 16,
        "llama-3": 32
    }[model_desc]
    dtype = {
        "gpt-2": np.uint16,
        "llama-3": np.uint32
    }[model_desc]
    max_token_id = 2**bits - 1
    max_length = 2**bits-1
    start_example_id = max_token_id

    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522 # magic
    header[1] = version # version
    header[2] = len(datas) # number of examples
    header[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_bytes = 0 # in units of uint16s
    full_stream = [] # the stream of uint16s, we'll write a single time at the end
    assert len(datas) < max_length, "too many examples?"
    for idx, data in enumerate(datas):
        stream = []
        # header of the example
        stream.append(start_example_id) # <START_EXAMPLE>
        stream.append(0) # <EXAMPLE_BYTES> (fill in later)
        stream.append(idx) # <EXAMPLE_INDEX>
        stream.append(data["label"]) # <LABEL>
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "expected 4 completions for now? can relax later"
        stream.append(len(ending_tokens)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_tokens = data["ctx_tokens"]
        assert all(0 <= t < max_token_id for t in ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        # the completion tokens
        for end_tokens in ending_tokens:
            assert all(0 <= t < max_token_id for t in end_tokens), "bad completion token"
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        # write to full stream
        nbytes = len(stream)*bits//8 # 2/4 bytes per uint16/unit32
        assert nbytes <= max_length, "example too large?"
        stream[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    # construct the numpy array
    stream_np = np.array(full_stream, dtype=dtype)
    # fill in the longest_example field
    assert 0 < longest_example_bytes < max_length, f"bad longest_example"
    header[3] = longest_example_bytes
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())
