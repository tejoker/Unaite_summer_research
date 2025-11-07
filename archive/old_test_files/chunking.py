import csv
import os
import glob
import itertools

def split_csv(
    input_file: str,
    output_prefix: str,            # put this inside your target folder; dirname() is used as output dir
    n: int = 1000,
    start_line: int | None = None, # 1-based (header is line 1); >=2 if provided
    make_first_n: bool = True,
    encoding: str = "utf-8",
):
    """
    - Writes chunks named output_of_the_{i}th_chunk.csv into the folder containing `output_prefix`.
    - If make_first_n=True: chunk #1 is header + first `n` data rows (no special filename).
    - Then continues chunking from `start_line` (1-based incl. header) if provided,
      otherwise right after the first-N slice.
    - Header is repeated in every chunk.
    - Old files matching output_of_the_*th_chunk.csv in that folder are deleted before writing.
    """
    output_dir = os.path.dirname(output_prefix)
    if not output_dir:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Clean old outputs from previous runs for this folder/pattern
    for old in glob.glob(os.path.join(output_dir, "output_of_the_*th_chunk.csv")):
        try:
            os.remove(old)
        except OSError:
            pass

    chunk_paths = []
    first_n_path = None

    with open(input_file, "r", newline="", encoding=encoding) as infile:
        reader = csv.reader(infile)
        header = next(reader)

        data_rows_consumed = 0  # how many data rows have been read so far

        chunk_idx = 0

        # 1) Optional first-N chunk (always named as a normal chunk)
        if make_first_n:
            chunk_idx = 1
            first_n_path = os.path.join(output_dir, f"output_of_the_{chunk_idx}th_chunk.csv")
            with open(first_n_path, "w", newline="", encoding=encoding) as out_first:
                w = csv.writer(out_first)
                w.writerow(header)
                for row in itertools.islice(reader, 0, n):
                    w.writerow(row)
                    data_rows_consumed += 1
            chunk_paths.append(first_n_path)

        # Decide where the *next* chunking starts (in 0-based data-row coordinates)
        if start_line is None:
            effective_start_data_idx = data_rows_consumed
        else:
            if start_line < 2:
                raise ValueError("start_line must be >= 2 (line 1 is the header).")
            # We can't rewind the reader; if start_line points before what we've already consumed,
            # start where we actually are.
            effective_start_data_idx = max(start_line - 1, data_rows_consumed)

        # 2) Chunk the remainder (or from requested start_line)
        outfile = None
        writer = None

        for i, row in enumerate(reader, start=data_rows_consumed):
            if i < effective_start_data_idx:
                continue

            rel = i - effective_start_data_idx  # 0,1,2,...

            if rel % n == 0:
                # start a new chunk
                if outfile:
                    outfile.close()
                chunk_idx += 1
                path = os.path.join(output_dir, f"output_of_the_{chunk_idx}th_chunk.csv")
                outfile = open(path, "w", newline="", encoding=encoding)
                writer = csv.writer(outfile)
                writer.writerow(header)
                chunk_paths.append(path)

            writer.writerow(row)

        if outfile:
            outfile.close()

    # Return: first_n_path is the same as chunk_paths[0] when make_first_n=True, else None
    return first_n_path, chunk_paths


if __name__ == "__main__":
    split_csv(
        input_file=r"C:\Users\Home\Documents\REZEL\program_internship_paul_wurth\data\cleaned_dataset.csv",
        # only the DIRECTORY of this matters now; files will be named output_of_the_{i}th_chunk.csv
        output_prefix=r"C:\Users\Home\Documents\REZEL\program_internship_paul_wurth\data\data_test",
        n=1000,
        start_line=2
    )
