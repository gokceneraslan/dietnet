import threading, queue
from multiprocessing.pool import ThreadPool

import bcolz

class BcolzSequence:
    def __init__(self, x_file, y_file, batch_size):
        self.X = bcolz.open(x_file)
        self.y = bcolz.open(y_file)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):

        batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

        return batch_x, batch_y

# OrderedEnqueue class from Keras master branch
# copied here to remove dependency to Keras master branch

class OrderedEnqueuer:
    """Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        scheduling: Sequential querying of datas if 'sequential', random otherwise.
    """

    def __init__(self, sequence,
                 use_multiprocessing=False,
                 scheduling='sequential'):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing
        self.scheduling = scheduling
        self.workers = 0
        self.executor = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor = multiprocessing.Pool(workers)
        else:
            self.executor = ThreadPool(workers)
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _run(self):
        """Function to submit request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        while True:
            if self.scheduling is not 'sequential':
                random.shuffle(sequence)
            for i in sequence:
                if self.stop_signal.is_set():
                    return
                self.queue.put(
                    self.executor.apply_async(get_index,
                                              (self.sequence, i)), block=True)

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            raise StopIteration(e)

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.executor.close()
        self.executor.join()
        self.run_thread.join(timeout)
