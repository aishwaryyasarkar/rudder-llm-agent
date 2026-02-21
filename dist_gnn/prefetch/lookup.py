import numba
from numba import jit, prange, njit, get_num_threads, config, threading_layer
import numpy as np

@jit(nopython=True, parallel=True)
def lookup(input_nodes_array, prefetch_ids, buffer_length, eviction_score, decay):
    """
    Performs parallel lookup operation on the input_nodes_array using the prefetch_ids.

    Args:
        input_nodes_array (numpy.ndarray): An array of input node IDs.
        prefetch_ids (numpy.ndarray): An array of prefetch IDs.
        buffer_length (int): The length of the buffer.
        eviction_score (float): The eviction score.
        decay (float): The decay factor.

    Returns:
        tuple: A tuple containing the following arrays:
            - hit_minibatch_idx (numpy.ndarray): An array of indices of the input_nodes_array that were found in the prefetch_ids.
            - missed_minibatch_idx (numpy.ndarray): An array of indices of the input_nodes_array that were not found in the prefetch_ids.
            - hit_buffer_idx (numpy.ndarray): An array of indices of the prefetch_ids where the input_nodes_array elements were found.
            - eviction_score (float): The updated eviction score.
    """
    # Initially mark all scores for decay
    eviction_score *= decay

    n = len(input_nodes_array)
    hit_minibatch_idx = np.full(n, -1, dtype=np.int32)
    missed_minibatch_idx = np.full(n, -1, dtype=np.int32)
    hit_buffer_idx = np.full(n, -1, dtype=np.int32)

    for i in prange(n):
        node_id = input_nodes_array[i]
        idx = np.searchsorted(prefetch_ids, node_id, side='left')
        if idx < buffer_length and idx < len(prefetch_ids) and prefetch_ids[idx] == node_id:
            hit_minibatch_idx[i] = i
            hit_buffer_idx[i] = idx
            # Restore the eviction score for this hit
            eviction_score[idx] /= decay
        else:
            missed_minibatch_idx[i] = i

    hit_minibatch_idx = hit_minibatch_idx[hit_minibatch_idx != -1]
    missed_minibatch_idx = missed_minibatch_idx[missed_minibatch_idx != -1]
    hit_buffer_idx = hit_buffer_idx[hit_buffer_idx != -1]

    return hit_minibatch_idx, missed_minibatch_idx, hit_buffer_idx, eviction_score

@jit(nopython=True, parallel=True)
def update_normal_scores(halo_nodes, unique_missed_minibatch_nodes, normal_scores):
    """
    Updates the normal scores for the halo nodes based on missed minibatch nodes.

    Args:
        halo_nodes (numpy.ndarray): An array of halo nodes.
        unique_missed_minibatch_nodes (numpy.ndarray): An array of unique missed minibatch nodes.
        normal_scores (numpy.ndarray): An array of normal scores.

    Returns:
        numpy.ndarray: An array of updated normal scores.
    """
    n = len(unique_missed_minibatch_nodes)
    l = len(halo_nodes)
    for i in prange(n):
        node_id = unique_missed_minibatch_nodes[i]
        idx = np.searchsorted(halo_nodes, node_id, side='left')
        if idx < l and halo_nodes[idx] == node_id:
            normal_scores[idx] += 1
    return normal_scores


@jit(nopython=True, parallel=True)
def update_normal_score_of_evicted_nodes(halo_nodes_rank, eviction_candidates, normal_scores, eviction_scores):
    """
    Update the normal scores of evicted nodes based on their eviction scores.

    Args:
        halo_nodes_rank (numpy.ndarray): Array of halo node ranks.
        eviction_candidates (numpy.ndarray): Array of node IDs of the eviction candidates.
        normal_scores (numpy.ndarray): Array of normal scores.
        eviction_scores (numpy.ndarray): Array of eviction scores.

    Returns:
        numpy.ndarray: Updated array of normal scores.
    """
    n = len(eviction_candidates)
    for i in prange(n):
        node_id = eviction_candidates[i]
        idx = np.searchsorted(halo_nodes_rank, node_id, side='left')
        if idx < len(halo_nodes_rank) and halo_nodes_rank[idx] == node_id:
            normal_scores[idx] = eviction_scores[i]
    return normal_scores