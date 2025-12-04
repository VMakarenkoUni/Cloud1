from Pyro4 import expose
import time
import random


class Solver:
    _worker_cache = {}

    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers

    def solve(self):
        total_start = time.time()

        paragraphs = self.read_input()
        max_k = min(5, len(paragraphs) - 1)
        max_iter = 3

        print("Paragraphs: %d, Workers: %d" % (len(paragraphs), len(self.workers)))

        vocab_start = time.time()
        word_counts = {}
        for para in paragraphs:
            words = para.lower().split()
            for word in words:
                word = word.strip('.,!?";:()[]')
                if word and len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1

        global_vocab = {}
        for word, count in word_counts.items():
            if count >= 10:
                global_vocab[word] = len(global_vocab)

        self.idx_to_word = {v: k for k, v in global_vocab.items()}
        vocab_time = time.time() - vocab_start
        print("Vocab build: %.3fs (%d words, filtered from %d)" % (vocab_time, len(global_vocab), len(word_counts)))

        init_start = time.time()
        init_futures = []
        for i in range(len(self.workers)):
            init_futures.append(self.workers[i].init_worker(paragraphs, global_vocab, i))
        for f in init_futures:
            f.value
        init_time = time.time() - init_start
        print("Workers init: %.3fs" % init_time)

        print("\n=== ELBOW METHOD (parallel) ===")
        elbow_start = time.time()

        k_values = list(range(max_k, 1, -1))

        import threading
        k_lock = threading.Lock()
        results_lock = threading.Lock()
        elbow_results = []

        def worker_loop(wid, worker):
            while True:
                with k_lock:
                    if not k_values:
                        return
                    k = k_values.pop(0)

                result = worker.run_elbow_k([k], max_iter, wid).value[0]

                with results_lock:
                    elbow_results.append(result)

        threads = []
        for i in range(len(self.workers)):
            t = threading.Thread(target=worker_loop, args=(i, self.workers[i]))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        elbow_results.sort(key=lambda x: x['k'])
        elbow_time = time.time() - elbow_start

        for r in elbow_results:
            print("k=%2d | Inertia: %8.4f | Iters: %2d | Time: %.2fs" %
                  (r['k'], r['inertia'], r['iterations'], r['time']))

        optimal_k = self.find_elbow(elbow_results)
        print("\n>>> Optimal k: %d <<<\n" % optimal_k)

        n = len(paragraphs)
        step = n // len(self.workers)
        chunks = []
        for i in range(len(self.workers)):
            start_idx = i * step
            end_idx = n if i == len(self.workers) - 1 else (i + 1) * step
            chunks.append(paragraphs[start_idx:end_idx])

        chunk_futures = []
        for i in range(len(self.workers)):
            chunk_futures.append(self.workers[i].init_chunks(chunks[i], i))
        for f in chunk_futures:
            f.value

        print("=== FINAL CLUSTERING (k=%d) ===" % optimal_k)
        final_start = time.time()
        labels, inertia, iters, comm_time, calc_time = self.run_kmeans(paragraphs, chunks, optimal_k, max_iter)
        final_time = time.time() - final_start

        total_time = time.time() - total_start

        print("\n=== TIMING SUMMARY ===")
        print("Vocab build:     %.3fs" % vocab_time)
        print("Workers init:    %.3fs" % init_time)
        print("Elbow method:    %.3fs" % elbow_time)
        print("Final cluster:   %.3fs" % final_time)
        print("Total:           %.3fs" % total_time)

        self.write_output(paragraphs, labels, optimal_k, elbow_results,
                          iters, inertia, total_time, len(self.workers),
                          vocab_time, init_time, elbow_time, final_time)

    def run_kmeans(self, paragraphs, chunks, n_clusters, max_iter):
        n = len(paragraphs)

        random.seed(n_clusters)
        centroid_indices = random.sample(range(n), n_clusters)
        centroid_texts = [paragraphs[i] for i in centroid_indices]

        labels = [0] * n
        total_comm_time = 0
        total_calc_time = 0
        iteration = 0

        while iteration < max_iter:
            comm_start = time.time()

            mapped = []
            for i in range(len(self.workers)):
                mapped.append(self.workers[i].find_nearest(centroid_texts, n_clusters, i))

            new_labels = []
            inertia_parts = []
            all_cluster_sums = [{} for _ in range(n_clusters)]
            all_cluster_counts = [0] * n_clusters

            for m in mapped:
                result = m.value
                new_labels.extend(result['labels'])
                inertia_parts.append(result['inertia'])
                for k in range(n_clusters):
                    all_cluster_counts[k] += result['cluster_counts'][k]
                    if result['cluster_sums'][k]:
                        for part in result['cluster_sums'][k].split(','):
                            if ':' not in part:
                                continue
                            sep = part.rfind(':')
                            idx = int(part[:sep])
                            val = int(part[sep + 1:])
                            all_cluster_sums[k][idx] = all_cluster_sums[k].get(idx, 0) + val

            comm_time = time.time() - comm_start
            total_comm_time += comm_time

            calc_start = time.time()
            changed = sum(1 for i in range(n) if labels[i] != new_labels[i])
            labels = new_labels
            iteration += 1

            if changed == 0:
                total_calc_time += time.time() - calc_start
                break

            new_centroid_texts = []
            for k in range(n_clusters):
                if all_cluster_counts[k] > 0:
                    words = []
                    for idx, val in all_cluster_sums[k].items():
                        word = self.idx_to_word[idx]
                        words.extend([word] * int(val))
                    new_centroid_texts.append(' '.join(words))
                else:
                    new_centroid_texts.append(centroid_texts[k])
            centroid_texts = new_centroid_texts
            total_calc_time += time.time() - calc_start

        total_inertia = sum(inertia_parts)
        return labels, total_inertia, iteration, total_comm_time, total_calc_time

    def find_elbow(self, results):
        if len(results) < 3:
            return results[0]['k']

        max_diff = 0
        elbow_k = results[0]['k']

        for i in range(1, len(results) - 1):
            prev_drop = results[i - 1]['inertia'] - results[i]['inertia']
            next_drop = results[i]['inertia'] - results[i + 1]['inertia']
            diff = prev_drop - next_drop

            if diff > max_diff:
                max_diff = diff
                elbow_k = results[i]['k']

        return elbow_k

    @staticmethod
    @expose
    def init_worker(paragraphs, global_vocab, worker_id):
        vectors = []
        norms = []
        for para in paragraphs:
            vec = {}
            words = para.lower().split()
            for word in words:
                word = word.strip('.,!?";:()[]')
                if word in global_vocab:
                    idx = global_vocab[word]
                    vec[idx] = vec.get(idx, 0) + 1
            norm = sum(v ** 2 for v in vec.values()) ** 0.5
            vectors.append(vec)
            norms.append(norm)

        Solver._worker_cache[worker_id] = {
            'vocab': global_vocab,
            'vectors': vectors,
            'norms': norms,
            'paragraphs': paragraphs,
            'mode': 'full'
        }
        return {'status': 'ok', 'paragraphs': len(paragraphs), 'vocab_size': len(global_vocab)}

    @staticmethod
    @expose
    def init_chunks(chunk, worker_id):
        cache = Solver._worker_cache[worker_id]
        vocab = cache['vocab']
        full_paragraphs = cache['paragraphs']

        start_idx = 0
        for i, p in enumerate(full_paragraphs):
            if p == chunk[0]:
                start_idx = i
                break

        chunk_vectors = cache['vectors'][start_idx:start_idx + len(chunk)]
        chunk_norms = cache['norms'][start_idx:start_idx + len(chunk)]

        Solver._worker_cache[worker_id] = {
            'vocab': vocab,
            'vectors': chunk_vectors,
            'norms': chunk_norms,
            'paragraphs': chunk,
            'mode': 'chunk'
        }
        return {'status': 'ok', 'paragraphs': len(chunk)}

    @staticmethod
    @expose
    def run_elbow_k(k_list, max_iter, worker_id):
        cache = Solver._worker_cache[worker_id]
        vocab = cache['vocab']
        vectors = cache['vectors']
        norms = cache['norms']
        paragraphs = cache['paragraphs']
        n = len(paragraphs)

        results = []

        for n_clusters in k_list:
            k_start = time.time()

            random.seed(n_clusters)
            centroid_indices = random.sample(range(n), n_clusters)

            centroid_vectors = [dict(vectors[i]) for i in centroid_indices]
            centroid_norms = [norms[i] for i in centroid_indices]

            labels = [0] * n
            iteration = 0

            while iteration < max_iter:
                new_labels = []
                inertia = 0

                for pi, v in enumerate(vectors):
                    min_dist = float('inf')
                    min_idx = 0
                    norm_v = norms[pi]
                    if norm_v == 0:
                        new_labels.append(0)
                        inertia += 1
                        continue
                    for i, c in enumerate(centroid_vectors):
                        norm_c = centroid_norms[i]
                        if norm_c == 0:
                            continue
                        dot = sum(v.get(idx, 0) * val for idx, val in c.items())
                        dist = 1 - dot / (norm_v * norm_c)
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = i
                    new_labels.append(min_idx)
                    inertia += min_dist

                changed = sum(1 for i in range(n) if labels[i] != new_labels[i])
                labels = new_labels
                iteration += 1

                if changed == 0:
                    break

                cluster_sums = [{} for _ in range(n_clusters)]
                cluster_counts = [0] * n_clusters
                for pi, label in enumerate(labels):
                    cluster_counts[label] += 1
                    for idx, val in vectors[pi].items():
                        cluster_sums[label][idx] = cluster_sums[label].get(idx, 0) + val

                centroid_vectors = []
                centroid_norms = []
                for k in range(n_clusters):
                    if cluster_counts[k] > 0:
                        centroid_vectors.append(cluster_sums[k])
                        norm = sum(v ** 2 for v in cluster_sums[k].values()) ** 0.5
                        centroid_norms.append(norm)
                    else:
                        centroid_vectors.append({})
                        centroid_norms.append(0)

            k_time = time.time() - k_start
            results.append({
                'k': n_clusters,
                'inertia': inertia,
                'iterations': iteration,
                'time': k_time
            })

        return results

    @staticmethod
    @expose
    def find_nearest(centroid_texts, n_clusters, worker_id):
        cache = Solver._worker_cache[worker_id]
        vocab = cache['vocab']
        para_vectors = cache['vectors']
        para_norms = cache['norms']

        centroid_vectors = []
        centroid_norms = []
        for text in centroid_texts:
            vec = {}
            words = text.lower().split()
            for word in words:
                word = word.strip('.,!?";:()[]')
                if word in vocab:
                    idx = vocab[word]
                    vec[idx] = vec.get(idx, 0) + 1
            norm = sum(v ** 2 for v in vec.values()) ** 0.5
            centroid_vectors.append(vec)
            centroid_norms.append(norm)

        labels = []
        inertia = 0
        for pi, v in enumerate(para_vectors):
            min_dist = float('inf')
            min_idx = 0
            norm_v = para_norms[pi]
            if norm_v == 0:
                labels.append(0)
                inertia += 1
                continue
            for i, c in enumerate(centroid_vectors):
                norm_c = centroid_norms[i]
                if norm_c == 0:
                    continue
                dot = sum(v.get(idx, 0) * val for idx, val in c.items())
                dist = 1 - dot / (norm_v * norm_c)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            labels.append(min_idx)
            inertia += min_dist

        cluster_sums = [{} for _ in range(n_clusters)]
        cluster_counts = [0] * n_clusters
        for pi, label in enumerate(labels):
            cluster_counts[label] += 1
            for idx, val in para_vectors[pi].items():
                cluster_sums[label][idx] = cluster_sums[label].get(idx, 0) + val

        cluster_sums_str = []
        for k in range(n_clusters):
            parts = ["%d:%d" % (idx, v) for idx, v in cluster_sums[k].items()]
            cluster_sums_str.append(','.join(parts))

        return {
            'labels': labels,
            'inertia': inertia,
            'cluster_sums': cluster_sums_str,
            'cluster_counts': cluster_counts
        }

    def read_input(self):
        f = open(self.input_file_name, 'r')
        text = f.read()
        f.close()
        return [p.strip() for p in text.split('\n') if len(p.strip()) > 50]

    def write_output(self, paragraphs, labels, optimal_k, elbow_results,
                     iterations, inertia, total_time, n_workers,
                     vocab_time, init_time, elbow_time, final_time):
        f = open(self.output_file_name, 'w')

        f.write('=' * 60 + '\n')
        f.write('K-MEANS TEXT CLUSTERING WITH ELBOW METHOD\n')
        f.write('=' * 60 + '\n\n')

        f.write('=== TIMING ===\n')
        f.write('Vocab build:     %.3fs\n' % vocab_time)
        f.write('Workers init:    %.3fs\n' % init_time)
        f.write('Elbow method:    %.3fs\n' % elbow_time)
        f.write('Final cluster:   %.3fs\n' % final_time)
        f.write('Total:           %.3fs\n\n' % total_time)

        f.write('=== SUMMARY ===\n')
        f.write('Workers: %d\n' % n_workers)
        f.write('Paragraphs: %d\n' % len(paragraphs))
        f.write('Optimal k: %d\n' % optimal_k)
        f.write('Final iterations: %d\n' % iterations)
        f.write('Final inertia: %.4f\n\n' % inertia)

        f.write('=== ELBOW METHOD RESULTS ===\n')
        f.write('%-5s %-12s %-6s %-10s\n' % ('K', 'Inertia', 'Iters', 'Time'))
        f.write('-' * 40 + '\n')
        for r in elbow_results:
            f.write('%-5d %-12.4f %-6d %-10.2fs\n' %
                    (r['k'], r['inertia'], r['iterations'], r['time']))

        f.write('\n=== CLUSTERS ===\n')
        for k in range(optimal_k):
            cluster_indices = [i for i in range(len(labels)) if labels[i] == k]
            f.write('\nCluster %d: %d paragraphs (%.1f%%)\n' %
                    (k, len(cluster_indices), 100 * len(cluster_indices) / len(paragraphs)))

            for idx in cluster_indices[:3]:
                preview = paragraphs[idx][:100] + '...' if len(paragraphs[idx]) > 100 else paragraphs[idx]
                f.write('  - %s\n' % preview)
            if len(cluster_indices) > 3:
                f.write('  ... +%d more\n' % (len(cluster_indices) - 3))

        f.close()
