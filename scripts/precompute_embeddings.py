import os
import hydra
import pandas as pd
from omegaconf import DictConfig
import bend.io.sequtils as sequtils


# load config
@hydra.main(config_path="../conf/embedding/", config_name="embed", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a embedding of nucleotide sequences.
    This function is called by hydra.
    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    """
    print("Embedding data for", cfg.task)
    if "chunk_size" in cfg[cfg.model]:
        cfg.chunk_size = cfg[cfg.model]["chunk_size"]
    # read the bed file and get the splits :
    if "splits" not in cfg or cfg.splits is None:
        splits = sequtils.get_splits(cfg[cfg.task].bed)
    else:
        splits = cfg.splits
    print("Embedding with", cfg.model)
    # instatiante model
    embedder = hydra.utils.instantiate(cfg[cfg.model])
    for split in splits:
        print(f"Embedding split: {split}")
        output_dir = f"{cfg.data_dir}/{cfg.task}/{cfg.model}/"

        os.makedirs(output_dir, exist_ok=True)

        # embed in chunks
        # get length of bed file and divide by chunk size, if a spcific chunk is not set1
        df = pd.read_csv(cfg[cfg.task].bed, sep="\t", low_memory=False)
        df = df[df.iloc[:, -1] == split] if split is not None else df
        possible_chunks = list(range(int(len(df) / cfg.chunk_size) + 1))
        if cfg.chunk is None:
            cfg.chunk = possible_chunks
        cfg.chunk = [cfg.chunk] if isinstance(cfg.chunk, int) else cfg.chunk
        # embed in chunks
        for n, chunk in enumerate(cfg.chunk):
            if chunk not in possible_chunks:
                print(f"{chunk} is not a valid chunk id. {split} chunk ids are {possible_chunks}")
                continue

            output_file = f"{output_dir}/{split}_{chunk}.tar.gz"
            if os.path.exists(output_file):
                print(f"\t Skipping chunk {chunk} for split {split} — {output_file} already exists.")
                continue

            print(f"\t Embedding chunk {chunk} ({chunk +1}/{len(possible_chunks)})")
            sequtils.embed_from_bed(
                **cfg[cfg.task],
                embedder=embedder,
                output_path=output_file,
                split=split,
                chunk=chunk,
                chunk_size=cfg.chunk_size,
                upsample_embeddings=(
                    cfg[cfg.model]["upsample_embeddings"] if "upsample_embeddings" in cfg[cfg.model] else False
                ),
            )


if __name__ == "__main__":

    print("Run Embedding")

    run_experiment()
