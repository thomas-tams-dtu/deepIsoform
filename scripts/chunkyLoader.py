#!/usr/bin/env python3

import gzip
import pandas as pd
from torch.utils.data import Dataset
from io import StringIO


class ReadGzip():
    """
    Class for opening a gzip file and lines_per_chunk amount of lines from the file
      each time load_next_chunk() is called

      The class should be opened first with open_gzip and close with close_gzip when done reading
    """
    def __init__(self, file_path):
      self.file_path = file_path
      self.gzip_file = None
      self.chunk_string = str()

    def open_gzip(self, skip_header=True):
      self.gzip_file = gzip.open(self.file_path, 'rt')
      if skip_header:
        next(self.gzip_file)

    def close_gzip(self):
      self.gzip_file.close()

    def load_next_chunk(self, lines_per_chunk):
      self.chunk_string = str()
      for _ in range(lines_per_chunk):
        self.chunk_string += self.gzip_file.readline()

      return self.chunk_string


class ChunkyDataset(Dataset):    
    """
    Class used for to load in chunks of data from a gzipped file and return the features and class
    to a pytorch DataLoader module
    """
    def __init__(self, file_path, nrows = 500, lines_per_chunk=10, got_header=True):
        self.file_path = file_path
        self.gzip_reader = ReadGzip(self.file_path)
        self.lines_per_chunk = lines_per_chunk
        self.chunk_lines = str()
        self.got_header = got_header
        self.nrows = nrows - self.got_header
        self.lines_returned_from_chunk = 0
        self.index_corrector = 0
        self.df = pd.DataFrame.empty
        self.sampleid = None
        self.data = None


    def loadNextChunk(self):
        # Load in next chunk of lines
        self.chunk_lines = self.gzip_reader.load_next_chunk(lines_per_chunk=self.lines_per_chunk)

        # Load in values for next chunk
        self.df = pd.read_csv(StringIO(self.chunk_lines), sep='\t', header=None)
        self.sampleid = self.df.iloc[:,0].values
        self.data = self.df.iloc[:,1:].values

    def __len__(self):
        return self.nrows

    def __getitem__(self, index):
        # If no data is loaded yet
        if index == 0:
          self.gzip_reader.open_gzip(skip_header=self.got_header)
          self.loadNextChunk()

        # Return all lines normally if lines loaded from current chunk is lower than the lines per chunk
        if self.lines_returned_from_chunk < self.lines_per_chunk:
            self.lines_returned_from_chunk += 1
            #print('index', index, 'loading', self.sampleid[index - self.index_corrector], 'corrected idx', index - self.index_corrector, 'lines_returned_from_chunk', self.lines_returned_from_chunk)

            # Close file if last line has been loaded
            if index == self.nrows - 1:
              self.gzip_reader.close_gzip()
              print('close gzipfile at idx', index)

            return self.data[index - self.index_corrector], self.sampleid[index - self.index_corrector]

        # If lines returned has reached the number of lines, load new chunk and return first line
        else:
          # Update index corrector and chunk
          self.index_corrector += self.lines_returned_from_chunk
          self.loadNextChunk()

          # Also send first line of chunk and update chunk counter
          self.lines_returned_from_chunk = 1

          # Close file if last line has been loaded
          if index == self.nrows - 1:
            self.gzip_reader.close_gzip()
            print('close gzipfile at idx', index)

          return self.data[index - self.index_corrector], self.sampleid[index - self.index_corrector]
