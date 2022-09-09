"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved."""

from torchtext.data.iterator import Iterator
from torchtext.data.batch import Batch


class CustomIterator(Iterator):
    """
    Custom modification of the torchtext.data.iterator Iterator class.
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None):

        super().__init__(dataset, batch_size, sort_key, device, batch_size_fn, train, repeat, shuffle, sort,
                         sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield minibatch, Batch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return
