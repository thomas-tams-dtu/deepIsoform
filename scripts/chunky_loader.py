#!/usr/bin/env python3

import gzip
import pandas as pd
from torch.utils.data import Dataset
from io import StringIO
import time

class ReadGzip():
    """
    Class for opening a gzip file and lines_per_chunk amount of lines from the file
    each time load_next_chunk() is called

    The class should be opened first with open_gzip and close with close_gzip when done reading
    """
    def __init__(self, file_path, skip_lines):
        self.file_path = file_path
        self.gzip_file = None
        self.skip_lines = skip_lines

    def open_gzip(self, skip_header=True):
        self.gzip_file = gzip.open(self.file_path, 'rt')
        if skip_header:
            next(self.gzip_file)
        
        for _ in range(self.skip_lines):
            next(self.gzip_file)

    def close_gzip(self):
        self.gzip_file.close()

    def read_chunk(self, lines_per_chunk):
        #start = time.time()
        chunk = []
        for _ in range(lines_per_chunk):
            line = next(self.gzip_file, None)
            if line is None:
                break
            chunk.append(line)

        #end = time.time()
        #print('chunk time', end-start)
        return "".join(chunk)

class ChunkyDataset(Dataset):
    """
    Class used for to load in chunks of data from a gzipped file and return the features and class
    to a pytorch DataLoader module
    """
    def __init__(self, file_path, nrows, lines_per_chunk, skip_lines, got_header=True):
        self.file_path = file_path
        self.skip_lines = skip_lines
        self.lines_per_chunk = lines_per_chunk

        self.gzip_reader = ReadGzip(self.file_path, skip_lines=self.skip_lines)
        self.got_header = got_header
        self.nrows = nrows - self.got_header

    def __len__(self):
        return self.nrows

    def __getitem__(self, index):
        if index == 0:
            self.gzip_reader.open_gzip(skip_header=self.got_header)

        if index % self.lines_per_chunk == 0:
            chunk = self.gzip_reader.read_chunk(self.lines_per_chunk)
            if not chunk:
                self.gzip_reader.close_gzip()
                return None

            self.df = pd.read_csv(StringIO(chunk), sep='\t', header=None)
            self.sampleid = self.df.iloc[:, 0].values
            self.data = self.df.iloc[:, 1:].values

        return self.data[index % self.lines_per_chunk], self.sampleid[index % self.lines_per_chunk]