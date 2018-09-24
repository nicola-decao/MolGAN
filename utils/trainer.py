import time
import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from datetime import timedelta
from utils.progress_bar import ProgressBar
from collections import defaultdict
import pprint


class Trainer:

    def __init__(self, model, optimizer, session):
        self.model, self.optimizer, self.session, self.print = model, optimizer, session, defaultdict(list)

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def save(self, directory):
        saver = tf.train.Saver()

        dirs = directory.split('/')
        dirs = ['/'.join(dirs[:i]) for i in range(1, len(dirs) + 1)]
        mkdirs = [d for d in dirs if not os.path.exists(d)]

        for d in mkdirs:
            os.makedirs(d)

        saver.save(self.session, '{}/{}.ckpt'.format(directory, 'model'))
        pickle.dump(self.print, open('{}/{}.pkl'.format(directory, 'trainer'), 'wb'))
        self.log('Model saved in {}!'.format(directory))

    def load(self, directory):
        saver = tf.train.Saver()
        saver.restore(self.session, '{}/{}.ckpt'.format(directory, 'model'))
        self.print = pickle.load(open('{}/{}.pkl'.format(directory, 'trainer'), 'rb'))
        self.log('Model loaded from {}!'.format(directory))

    def train(self, batch_dim, epochs, steps,
              train_fetch_dict, train_feed_dict,
              eval_fetch_dict, eval_feed_dict,
              test_fetch_dict, test_feed_dict,
              _train_step=None, _eval_step=None, _test_step=None,
              _train_update=None, _eval_update=None, _test_update=None,
              eval_batch=None, test_batch=None,
              best_fn=None, min_epochs=None, look_ahead=None,
              save_every=None, directory=None,
              skip_first_eval=False):

        if _train_step is None:
            def _train_step(step, steps, epoch, epochs, min_epochs, model, optimizer, batch_dim):
                return self.session.run(train_fetch_dict(step, steps, epoch, epochs, min_epochs, model, optimizer),
                                        feed_dict=train_feed_dict(step, steps, epoch, epochs, min_epochs, model,
                                                                  optimizer, batch_dim))

        if _eval_step is None:
            def _eval_step(epoch, epochs, min_epochs, model, optimizer, batch_dim, eval_batch, start_time,
                           last_epoch_start_time, _eval_update):
                from_start = timedelta(seconds=int((time.time() - start_time)))
                last_epoch = timedelta(seconds=int((time.time() - last_epoch_start_time)))
                eta = timedelta(seconds=int((time.time() - start_time) * (epochs - epoch) / epoch)
                                ) if (time.time() - start_time) > 1 else '-:--:-'

                self.log(
                    'Epochs {:10}/{} in {} (last epoch in {}), ETA: {}'.format(epoch, epochs, from_start, last_epoch,
                                                                               eta))

                if eval_batch is not None:
                    pr = ProgressBar(80, eval_batch)
                    output = defaultdict(list)

                    for i in range(eval_batch):
                        for k, v in self.session.run(eval_fetch_dict(epoch, epochs, min_epochs, model, optimizer),
                                                     feed_dict=eval_feed_dict(epoch, epochs, min_epochs, model,
                                                                              optimizer, batch_dim)).items():
                            output[k].append(v)
                        pr.update(i + 1)

                    self.log(date=False)
                    output = {k: np.mean(v) for k, v in output.items()}
                else:
                    output = self.session.run(eval_fetch_dict(epoch, epochs, min_epochs, model, optimizer),
                                              feed_dict=eval_feed_dict(epoch, epochs, min_epochs, model,
                                                                       optimizer, batch_dim))

                if _eval_update is not None:
                    output.update(_eval_update(epoch, epochs, min_epochs, model, optimizer, batch_dim, eval_batch))

                p = pprint.PrettyPrinter(indent=1, width=80)
                self.log('Validation --> {}'.format(p.pformat(output)))

                for k in output:
                    self.print[k].append(output[k])

                return output

        if _test_step is None:
            def _test_step(model, optimizer, batch_dim, test_batch, start_time, _test_update):
                self.load(directory)
                from_start = timedelta(seconds=int((time.time() - start_time)))
                self.log('End of training ({} epochs) in {}'.format(epochs, from_start))

                if test_batch is not None:
                    pr = ProgressBar(80, test_batch)
                    output = defaultdict(list)

                    for i in range(test_batch):
                        for k, v in self.session.run(test_fetch_dict(model, optimizer),
                                                     feed_dict=test_feed_dict(model, optimizer, batch_dim)).items():
                            output[k].append(v)
                        pr.update(i + 1)

                    self.log(date=False)
                    output = {k: np.mean(v) for k, v in output.items()}
                else:
                    output = self.session.run(test_fetch_dict(model, optimizer),
                                              feed_dict=test_feed_dict(model, optimizer, batch_dim))

                if _test_update is not None:
                    output.update(_test_update(model, optimizer, batch_dim, test_batch))

                p = pprint.PrettyPrinter(indent=1, width=80)
                self.log('Test --> {}'.format(p.pformat(output)))

                for k in output:
                    self.print['Test ' + k].append(output[k])

                return output

        best_model_value = None
        no_improvements = 0
        start_time = time.time()
        last_epoch_start_time = time.time()

        for epoch in range(epochs + 1):

            if not (skip_first_eval and epoch == 0):

                result = _eval_step(epoch, epochs, min_epochs, self.model, self.optimizer, batch_dim, eval_batch,
                                    start_time, last_epoch_start_time, _eval_update)

                if best_fn is not None and (True if best_model_value is None else best_fn(result) > best_model_value):
                    self.save(directory)
                    best_model_value = best_fn(result)
                    no_improvements = 0
                elif look_ahead is not None and no_improvements < look_ahead:
                    no_improvements += 1
                    self.load(directory)
                elif min_epochs is not None and epoch >= min_epochs:
                    self.log('No improvements after {} epochs!'.format(no_improvements))
                    break

                if save_every is not None and epoch % save_every == 0:
                    self.save(directory)

            if epoch < epochs:
                last_epoch_start_time = time.time()
                pr = ProgressBar(80, steps)
                for step in range(steps):
                    _train_step(steps * epoch + step, steps, epoch, epochs, min_epochs, self.model, self.optimizer,
                                batch_dim)
                    pr.update(step + 1)

                self.log(date=False)

        _test_step(self.model, self.optimizer, batch_dim, eval_batch, start_time, _test_update)
