from Pyro4 import expose
import time
import random
import math


class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers

    def solve(self):
        start_time = time.time()

        text = self.read_input()
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]

        step = len(paragraphs) // len(self.workers)

        map_start = time.time()
        mapped = []
        chunks = []
        for i in range(len(self.workers)):
            start_idx = i * step
            end_idx = len(paragraphs) if i == len(self.workers) - 1 else (i + 1) * step
            chunk = paragraphs[start_idx:end_idx]
            chunks.append(chunk)
            mapped.append(self.workers[i].cluster_chunk(chunk, i))
        map_end = time.time()

        reduce_start = time.time()
        results = []
        worker_stats = []
        for m in mapped:
            result = m.value
            results.append(result)
            worker_stats.append(result['stats'])
        reduce_end = time.time()

        end_time = time.time()

        timing = {
            'total': end_time - start_time,
            'map_phase': map_end - map_start,
            'reduce_phase': reduce_end - reduce_start,
            'workers_used': len(self.workers)
        }

        self.write_output(chunks, results, worker_stats, timing)

    @staticmethod
    @expose
    def cluster_chunk(paragraphs, worker_id):
        random.seed(worker_id)

        if len(paragraphs) == 0:
            return {
                'labels': [],
                'inertia': 0,
                'stats': {
                    'worker_id': worker_id,
                    'paragraphs': 0,
                    'vocab_size': 0,
                    'clusters': 0,
                    'inertia': 0
                }
            }

        vocab = {}
        for para in paragraphs:
            words = para.lower().split()
            for word in words:
                word = word.strip('.,!?";:')
                if word and len(word) > 2:
                    if word not in vocab:
                        vocab[word] = len(vocab)

        if len(vocab) == 0:
            return {
                'labels': [0] * len(paragraphs),
                'inertia': 0,
                'stats': {
                    'worker_id': worker_id,
                    'paragraphs': len(paragraphs),
                    'vocab_size': 0,
                    'clusters': 1,
                    'inertia': 0
                }
            }

        vectors = []
        for para in paragraphs:
            vec = [0] * len(vocab)
            words = para.lower().split()
            for word in words:
                word = word.strip('.,!?";:')
                if word in vocab:
                    vec[vocab[word]] += 1
            total = sum(vec)
            if total > 0:
                vec = [v / total for v in vec]
            vectors.append(vec)

        k = min(5, len(vectors))
        n = len(vectors)
        dim = len(vocab)

        centroids = random.sample(vectors, k)

        for iteration in range(50):
            labels = []
            for v in vectors:
                distances = []
                for c in centroids:
                    dist = sum((v[i] - c[i]) ** 2 for i in range(dim))
                    distances.append(math.sqrt(dist))
                labels.append(distances.index(min(distances)))

            new_centroids = []
            for ki in range(k):
                cluster_points = [vectors[i] for i in range(n) if labels[i] == ki]
                if cluster_points:
                    centroid = [sum(p[i] for p in cluster_points) / len(cluster_points) for i in range(dim)]
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(centroids[ki])

            converged = True
            for i in range(k):
                for j in range(dim):
                    if abs(centroids[i][j] - new_centroids[i][j]) > 0.0001:
                        converged = False
                        break
                if not converged:
                    break

            centroids = new_centroids
            if converged:
                break

        inertia = 0
        for i, v in enumerate(vectors):
            c = centroids[labels[i]]
            inertia += sum((v[j] - c[j]) ** 2 for j in range(dim))

        return {
            'labels': labels,
            'inertia': inertia,
            'stats': {
                'worker_id': worker_id,
                'paragraphs': len(paragraphs),
                'vocab_size': len(vocab),
                'clusters': k,
                'inertia': inertia
            }
        }

    def read_input(self):
        f = open(self.input_file_name, 'r')
        return f.read()

    def write_output(self, chunks, results, stats, timing):
        f = open(self.output_file_name, 'w')

        f.write('=== TIMING ===\n')
        f.write('Total time: %.3f seconds\n' % timing['total'])
        f.write('Map phase: %.3f seconds\n' % timing['map_phase'])
        f.write('Reduce phase: %.3f seconds\n' % timing['reduce_phase'])
        f.write('Workers used: %d\n\n' % timing['workers_used'])

        f.write('=== WORKER STATISTICS ===\n')
        for s in stats:
            f.write('Worker %d: %d paragraphs, %d vocab, %d clusters, inertia %.4f\n' %
                    (s['worker_id'], s['paragraphs'], s['vocab_size'], s['clusters'], s['inertia']))

        f.write('\n=== CLUSTERING RESULTS ===\n')
        for i, result in enumerate(results):
            if len(result['labels']) == 0:
                continue

            f.write('\n--- Worker %d ---\n' % i)
            labels = result['labels']
            paragraphs = chunks[i]
            n_clusters = max(labels) + 1

            for k in range(n_clusters):
                cluster_paras = [paragraphs[j] for j in range(len(paragraphs)) if labels[j] == k]
                f.write('Cluster %d: %d paragraphs\n' % (k, len(cluster_paras)))
                for idx, para in enumerate(cluster_paras[:2]):
                    preview = para[:80] + '...' if len(para) > 80 else para
                    f.write('  %s\n' % preview)
                if len(cluster_paras) > 2:
                    f.write('  ... %d more\n' % (len(cluster_paras) - 2))

        f.close()