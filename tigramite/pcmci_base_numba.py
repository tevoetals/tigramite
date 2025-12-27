"""Tigramite causal discovery for time series - Numba Optimized Version."""

from __future__ import print_function
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import scipy.stats
import math
from joblib import Parallel, delayed
from numba import njit, prange, types
from numba.typed import Dict as NumbaDict
import numba

# ============================================================================
# NUMBA-OPTIMIZED KERNELS (Funções numéricas puras)
# ============================================================================

@njit(cache=True)
def _ecdf_numba(x):
    """Optimized empirical CDF for FDR correction."""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


@njit(cache=True)
def _fdr_bh_correction_numba(pvals_sorted):
    """Benjamini-Hochberg FDR correction - core numerical loop."""
    n = len(pvals_sorted)
    ecdffactor = np.arange(1, n + 1) / float(n)
    
    # pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected_raw = np.empty(n, dtype=np.float64)
    for i in range(n):
        pvals_corrected_raw[i] = pvals_sorted[i] / ecdffactor[i]
    
    # Reverse cumulative minimum
    pvals_corrected = np.empty(n, dtype=np.float64)
    pvals_corrected[n-1] = pvals_corrected_raw[n-1]
    for i in range(n - 2, -1, -1):
        pvals_corrected[i] = min(pvals_corrected[i + 1], pvals_corrected_raw[i])
    
    # Clip to [0, 1]
    for i in range(n):
        if pvals_corrected[i] > 1.0:
            pvals_corrected[i] = 1.0
    
    return pvals_corrected


@njit(cache=True)
def _apply_fdr_correction_numba(p_matrix_flat, mask_flat, pvals_sortind, pvals_corrected):
    """Apply corrected p-values back to matrix positions."""
    n = len(pvals_sortind)
    pvals_corrected_reordered = np.empty(n, dtype=np.float64)
    for i in range(n):
        pvals_corrected_reordered[pvals_sortind[i]] = pvals_corrected[i]
    
    result = p_matrix_flat.copy()
    mask_indices = np.where(mask_flat)[0]
    for i in range(n):
        result[mask_indices[i]] = pvals_corrected_reordered[i]
    
    return result


@njit(cache=True)
def _dict_to_matrix_kernel(val_keys_i, val_keys_j, val_keys_tau, val_values, 
                           n_vars, tau_max_plus1, default):
    """Convert flattened dict representation to matrix."""
    matrix = np.full((n_vars, n_vars, tau_max_plus1), default, dtype=np.float64)
    
    n_entries = len(val_keys_i)
    for idx in range(n_entries):
        k = val_keys_i[idx]
        j = val_keys_j[idx]
        tau = val_keys_tau[idx]
        val = val_values[idx]
        
        if tau == 0:
            matrix[k, j, 0] = val
            matrix[j, k, 0] = val
        else:
            matrix[k, j, tau] = val
    
    return matrix


@njit(cache=True, parallel=True)
def _symmetrize_matrices_kernel(p_matrix, val_matrix, 
                                 link_i, link_j, link_types,
                                 N):
    """Symmetrize p_matrix and val_matrix based on link assumptions.
    
    link_types encoding: 0 = not present, 1 = o-o/o?o, 2 = -->/->
    """
    p_out = p_matrix.copy()
    val_out = val_matrix.copy()
    
    n_links = len(link_i)
    for idx in prange(n_links):
        i = link_i[idx]
        j = link_j[idx]
        ltype = link_types[idx]
        
        if ltype == 1:  # o-o or o?o - use max p-value
            if p_matrix[i, j, 0] >= p_matrix[j, i, 0]:
                p_out[j, i, 0] = p_matrix[i, j, 0]
                val_out[j, i, 0] = val_matrix[i, j, 0]
        elif ltype == 2:  # --> or -?>
            p_out[j, i, 0] = p_matrix[i, j, 0]
            val_out[j, i, 0] = val_matrix[i, j, 0]
    
    return p_out, val_out


@njit(cache=True)
def _count_link_frequencies(graphs_encoded, n_results, N, tau_max_plus1, n_link_types):
    """Count frequency of each link type across bootstrap/window results.
    
    graphs_encoded: (n_results, N, N, tau_max+1) array of integers encoding link types
    Returns counts array of shape (N, N, tau_max+1, n_link_types)
    """
    counts = np.zeros((N, N, tau_max_plus1, n_link_types), dtype=np.int64)
    
    for b in range(n_results):
        for i in range(N):
            for j in range(N):
                for tau in range(tau_max_plus1):
                    link_type = graphs_encoded[b, i, j, tau]
                    counts[i, j, tau, link_type] += 1
    
    return counts


@njit(cache=True, parallel=True)
def _find_most_frequent_links(counts, n_results, N, tau_max_plus1):
    """Find most frequent link type for each position.
    
    Returns:
        most_freq: (N, N, tau_max+1) array of most frequent link type indices
        frequencies: (N, N, tau_max+1) array of frequencies
    """
    most_freq = np.zeros((N, N, tau_max_plus1), dtype=np.int64)
    frequencies = np.zeros((N, N, tau_max_plus1), dtype=np.float64)
    
    n_link_types = counts.shape[3]
    
    for i in prange(N):
        for j in range(N):
            for tau in range(tau_max_plus1):
                max_count = 0
                max_idx = 0
                for lt in range(n_link_types):
                    if counts[i, j, tau, lt] > max_count:
                        max_count = counts[i, j, tau, lt]
                        max_idx = lt
                
                most_freq[i, j, tau] = max_idx
                frequencies[i, j, tau] = max_count / float(n_results)
    
    return most_freq, frequencies


@njit(cache=True)
def _compute_percentiles(val_matrix_stack, q_low, q_high):
    """Compute percentiles along first axis efficiently."""
    n_results, N1, N2, tau_max_plus1 = val_matrix_stack.shape
    
    result_low = np.zeros((N1, N2, tau_max_plus1), dtype=np.float64)
    result_high = np.zeros((N1, N2, tau_max_plus1), dtype=np.float64)
    
    for i in range(N1):
        for j in range(N2):
            for tau in range(tau_max_plus1):
                vals = np.sort(val_matrix_stack[:, i, j, tau])
                
                # Percentile calculation
                idx_low = int(q_low * (n_results - 1))
                idx_high = int(q_high * (n_results - 1))
                
                result_low[i, j, tau] = vals[idx_low]
                result_high[i, j, tau] = vals[idx_high]
    
    return result_low, result_high


@njit(cache=True)
def _check_cyclic_kernel(adj_matrix, n_vars):
    """Check for cycles in adjacency matrix using DFS.
    
    adj_matrix[i,j] = 1 if there's a directed edge i -> j at lag 0
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = np.zeros(n_vars, dtype=np.int32)
    
    for start in range(n_vars):
        if color[start] == WHITE:
            # DFS stack: (node, neighbor_index)
            stack = [(start, 0)]
            color[start] = GRAY
            
            while len(stack) > 0:
                node, next_neighbor = stack[-1]
                
                # Find next unvisited neighbor
                found = False
                for neighbor in range(next_neighbor, n_vars):
                    if adj_matrix[node, neighbor] == 1:
                        stack[-1] = (node, neighbor + 1)
                        
                        if color[neighbor] == GRAY:
                            return True  # Cycle found
                        elif color[neighbor] == WHITE:
                            color[neighbor] = GRAY
                            stack.append((neighbor, 0))
                            found = True
                            break
                
                if not found:
                    color[node] = BLACK
                    stack.pop()
    
    return False


# ============================================================================
# LINK TYPE ENCODING (para usar com Numba)
# ============================================================================

# Mapping de strings para inteiros (usado fora do Numba)
LINK_TYPE_TO_INT = {
    '': 0,
    'o-o': 1,
    'o?o': 2,
    '-->': 3,
    '-?>': 4,
    '<--': 5,
    '<?-': 6,
    'x-x': 7,
    'x?x': 8,
}

INT_TO_LINK_TYPE = {v: k for k, v in LINK_TYPE_TO_INT.items()}
N_LINK_TYPES = len(LINK_TYPE_TO_INT)


def encode_graph(graph_str):
    """Convert string graph to integer encoding."""
    shape = graph_str.shape
    result = np.zeros(shape, dtype=np.int64)
    for idx in np.ndindex(shape):
        result[idx] = LINK_TYPE_TO_INT.get(graph_str[idx], 0)
    return result


def decode_graph(graph_int):
    """Convert integer encoding back to string graph."""
    shape = graph_int.shape
    result = np.empty(shape, dtype='<U3')
    for idx in np.ndindex(shape):
        result[idx] = INT_TO_LINK_TYPE.get(graph_int[idx], '')
    return result


# ============================================================================
# OPTIMIZED PCMCIbase CLASS
# ============================================================================

class PCMCIbase():
    r"""PCMCI base class - Numba Optimized Version.

    Parameters
    ----------
    dataframe : data object
        This is the Tigramite dataframe object.
    cond_ind_test : conditional independence test object
        This can be ParCorr or other classes from tigramite.independence_tests.
    verbosity : int, optional (default: 0)
        Verbose levels 0, 1, ...
    """

    def __init__(self, dataframe, cond_ind_test, verbosity=0):
        self.dataframe = dataframe
        self.cond_ind_test = deepcopy(cond_ind_test)
        if isinstance(self.cond_ind_test, type):
            raise ValueError("PCMCI requires that cond_ind_test is instantiated.")
        self.cond_ind_test.set_dataframe(self.dataframe)
        self.verbosity = verbosity
        self.var_names = self.dataframe.var_names
        self.T = self.dataframe.T
        self.N = self.dataframe.N
        
        # Pre-compile Numba functions on first use
        self._warmup_numba()
    
    def _warmup_numba(self):
        """Warm up Numba JIT compilation with small arrays."""
        # Trigger compilation with minimal data
        _small = np.array([0.1, 0.2, 0.3])
        _ = _ecdf_numba(_small)
        _ = _fdr_bh_correction_numba(_small)

    def _reverse_link(self, link):
        """Reverse a given link."""
        if link == "":
            return ""
        left_mark = "<" if link[2] == ">" else link[2]
        right_mark = ">" if link[0] == "<" else link[0]
        return left_mark + link[1] + right_mark

    def _check_cyclic(self, link_dict):
        """Return True if the link_dict has a contemporaneous cycle.
        
        Optimized version using Numba kernel.
        """
        n_vars = len(link_dict)
        
        # Build adjacency matrix for lag-0 directed edges
        adj_matrix = np.zeros((n_vars, n_vars), dtype=np.int32)
        
        for vertex in link_dict:
            for itaui in link_dict.get(vertex, ()):
                i, taui = itaui
                link_type = link_dict[vertex][itaui]
                if taui == 0 and link_type in ['-->', '-?>']:
                    adj_matrix[vertex, i] = 1
        
        return _check_cyclic_kernel(adj_matrix, n_vars)

    def _set_link_assumptions(self, link_assumptions, tau_min, tau_max,
                              remove_contemp=False):
        """Helper function to set and check the link_assumptions argument."""
        _int_link_assumptions = deepcopy(link_assumptions)
        _vars = list(range(self.N))
        _lags = list(range(-(tau_max), -tau_min + 1, 1))
        
        if _int_link_assumptions is None:
            _int_link_assumptions = {}
            for j in _vars:
                _int_link_assumptions[j] = {}
                for i in _vars:
                    for lag in range(tau_min, tau_max + 1):
                        if not (i == j and lag == 0):
                            if lag == 0:
                                _int_link_assumptions[j][(i, 0)] = 'o?o'
                            else:
                                _int_link_assumptions[j][(i, -lag)] = '-?>'
        else:
            if remove_contemp:
                for j in _int_link_assumptions.keys():
                    _int_link_assumptions[j] = {
                        link: _int_link_assumptions[j][link]
                        for link in _int_link_assumptions[j]
                        if link[1] != 0
                    }

        # Make contemporaneous assumptions consistent
        for j in _vars:
            for link in _int_link_assumptions[j]:
                i, tau = link
                link_type = _int_link_assumptions[j][link]
                if tau == 0:
                    if (j, 0) in _int_link_assumptions[i]:
                        if _int_link_assumptions[j][link] != self._reverse_link(
                                _int_link_assumptions[i][(j, 0)]):
                            raise ValueError(
                                f"Inconsistent link assumptions for indices {i} - {j}")
                    else:
                        _int_link_assumptions[i][(j, 0)] = self._reverse_link(
                            _int_link_assumptions[j][link])
                else:
                    new_link_type = '-' + link_type[1] + '>'
                    _int_link_assumptions[j][link] = new_link_type

        # Validation
        _key_set = set(_int_link_assumptions.keys())
        valid_entries = _key_set == set(range(self.N))
        valid_types = ['o-o', 'o?o', '-->', '-?>', '<--', '<?-']

        for links in _int_link_assumptions.values():
            if isinstance(links, dict) and len(links) == 0:
                continue
            for var, lag in links:
                if var not in _vars or lag not in _lags:
                    valid_entries = False
                if links[(var, lag)] not in valid_types:
                    valid_entries = False

        if not valid_entries:
            raise ValueError(
                "link_assumptions must be dictionary with keys for all [0,...,N-1] "
                f"variables and contain only valid link types: {valid_types}")

        if self._check_cyclic(_int_link_assumptions):
            raise ValueError("link_assumptions has contemporaneous cycle(s).")

        return _int_link_assumptions

    def _dict_to_matrix(self, val_dict, tau_max, n_vars, default=1):
        """Helper function to convert dictionary to matrix format.
        
        Optimized with Numba kernel.
        """
        # Flatten dictionary to arrays for Numba
        keys_i = []
        keys_j = []
        keys_tau = []
        values = []
        
        for j in val_dict.keys():
            for link in val_dict[j].keys():
                k, tau = link
                keys_i.append(k)
                keys_j.append(j)
                keys_tau.append(abs(tau))
                values.append(val_dict[j][link])
        
        if len(keys_i) == 0:
            return np.full((n_vars, n_vars, tau_max + 1), default, dtype=np.float64)
        
        return _dict_to_matrix_kernel(
            np.array(keys_i, dtype=np.int64),
            np.array(keys_j, dtype=np.int64),
            np.array(keys_tau, dtype=np.int64),
            np.array(values, dtype=np.float64),
            n_vars, tau_max + 1, float(default)
        )

    def get_corrected_pvalues(self, p_matrix, fdr_method='fdr_bh',
                              exclude_contemporaneous=True,
                              tau_min=0, tau_max=1, link_assumptions=None):
        """Returns p-values corrected for multiple testing.
        
        Optimized with Numba FDR correction.
        """
        _, N, tau_max_plusone = p_matrix.shape
        self._check_tau_limits(tau_min, tau_max)
        
        # Build mask
        if link_assumptions is not None:
            mask = np.zeros((N, N, tau_max_plusone), dtype=np.bool_)
            _int_link_assumptions = self._set_link_assumptions(
                link_assumptions, tau_min, tau_max)
            for j, links_ in _int_link_assumptions.items():
                for link in links_:
                    i, lag = link
                    if _int_link_assumptions[j][link] not in ["<--", "<?-"]:
                        mask[i, j, abs(lag)] = True
        else:
            mask = np.ones((N, N, tau_max_plusone), dtype=np.bool_)
        
        # Ignore diagonal at lag 0
        mask[range(N), range(N), 0] = False
        if exclude_contemporaneous:
            mask[:, :, 0] = False
        
        q_matrix = np.array(p_matrix)
        
        if fdr_method is None or fdr_method == 'none':
            return q_matrix
        elif fdr_method == 'fdr_bh':
            # Extract p-values under mask
            pvs = p_matrix[mask]
            if len(pvs) == 0:
                return q_matrix
            
            # Sort and apply FDR correction with Numba
            pvals_sortind = np.argsort(pvs)
            pvals_sorted = pvs[pvals_sortind]
            
            # Numba-optimized FDR correction
            pvals_corrected = _fdr_bh_correction_numba(pvals_sorted)
            
            # Reorder corrected p-values
            pvals_corrected_reordered = np.empty_like(pvals_corrected)
            pvals_corrected_reordered[pvals_sortind] = pvals_corrected
            
            q_matrix[mask] = pvals_corrected_reordered
        else:
            raise ValueError('Only FDR method fdr_bh implemented')
        
        return q_matrix

    def _get_adj_time_series(self, graph, include_conflicts=True, sort_by=None):
        """Helper function that returns dictionary of adjacencies from graph."""
        N, _, tau_max_plusone = graph.shape
        adjt = {}
        
        if include_conflicts:
            for j in range(N):
                where = np.where(graph[:, j, :] != "")
                adjt[j] = list(zip(*(where[0], -where[1])))
        else:
            for j in range(N):
                where = np.where(np.logical_and.reduce((
                    graph[:, j, :] != "",
                    graph[:, j, :] != "x-x",
                    graph[:, j, :] != "x?x"
                )))
                adjt[j] = list(zip(*(where[0], -where[1])))

        if sort_by is not None:
            for j in range(N):
                abs_values = {k: np.abs(sort_by[j][k]) for k in list(sort_by[j])
                              if k in adjt[j]}
                adjt[j] = sorted(abs_values, key=abs_values.get, reverse=True)

        return adjt

    def _get_adj_time_series_contemp(self, graph, include_conflicts=True, sort_by=None):
        """Helper function for contemporaneous adjacencies."""
        N = graph.shape[0]
        adjt = self._get_adj_time_series(graph, include_conflicts, sort_by)
        for j in range(N):
            adjt[j] = [a for a in adjt[j] if a[1] == 0]
        return adjt

    def _get_simplicial_node(self, circle_cpdag, variable_order):
        """Find simplicial nodes in circle component CPDAG."""
        for j in variable_order:
            adj_j = np.where(np.logical_or(
                circle_cpdag[:, j, 0] == "o-o",
                circle_cpdag[:, j, 0] == "o?o"
            ))[0].tolist()

            all_adjacent = len(adj_j) > 0

            if len(adj_j) == 1:
                return (j, adj_j)
            else:
                for (var1, var2) in itertools.combinations(adj_j, 2):
                    if circle_cpdag[var1, var2, 0] == "":
                        all_adjacent = False
                        break

                if all_adjacent:
                    return (j, adj_j)

        return None

    def _get_dag_from_cpdag(self, cpdag_graph, variable_order):
        """Yields one member of the Markov equivalence class of a CPDAG."""
        dag = np.copy(cpdag_graph)
        circle_cpdag = np.copy(cpdag_graph)
        circle_cpdag[:, :, 1:] = ""
        circle_cpdag[circle_cpdag == "x-x"] = ""
        
        for i, j, tau in zip(*np.where(circle_cpdag != "")):
            if circle_cpdag[i, j, 0][1] == '?':
                raise ValueError("Invalid middle mark.")
            if circle_cpdag[i, j, 0] == "-->":
                circle_cpdag[i, j, 0] = ""

        simplicial_node = self._get_simplicial_node(circle_cpdag, variable_order)
        while simplicial_node is not None:
            (j, adj_j) = simplicial_node
            for var in adj_j:
                dag[var, j, 0] = "-->"
                dag[j, var, 0] = "<--"
                circle_cpdag[var, j, 0] = circle_cpdag[j, var, 0] = ""
            simplicial_node = self._get_simplicial_node(circle_cpdag, variable_order)

        return dag

    def convert_to_string_graph(self, graph_bool):
        """Converts the 0,1-based graph to string array with links '-->'."""
        graph = np.zeros(graph_bool.shape, dtype='<U3')
        graph[:] = ""
        graph[:, :, 1:][graph_bool[:, :, 1:] == 1] = "-->"
        graph[:, :, 0][np.logical_and(graph_bool[:, :, 0] == 1,
                                       graph_bool[:, :, 0].T == 1)] = "o-o"
        graph[:, :, 0][np.logical_and(graph_bool[:, :, 0] == 2,
                                       graph_bool[:, :, 0].T == 2)] = "x-x"
        
        for (i, j) in zip(*np.where(
                np.logical_and(graph_bool[:, :, 0] == 1, graph_bool[:, :, 0].T == 0))):
            graph[i, j, 0] = "-->"
            graph[j, i, 0] = "<--"

        return graph

    def symmetrize_p_and_val_matrix(self, p_matrix, val_matrix, link_assumptions,
                                     conf_matrix=None):
        """Symmetrizes matrices based on link_assumptions.
        
        Optimized version with Numba kernel for numerical operations.
        """
        # Prepare arrays for Numba kernel
        link_i = []
        link_j = []
        link_types = []  # 0=not present, 1=o-o/o?o, 2=-->/->
        
        for i in range(self.N):
            for j in range(self.N):
                if (i, 0) in link_assumptions[j]:
                    ltype = link_assumptions[j][(i, 0)]
                    if ltype in ["o-o", 'o?o']:
                        link_i.append(i)
                        link_j.append(j)
                        link_types.append(1)
                    elif ltype in ["-->", '-?>']:
                        link_i.append(i)
                        link_j.append(j)
                        link_types.append(2)
        
        if len(link_i) > 0:
            p_out, val_out = _symmetrize_matrices_kernel(
                p_matrix, val_matrix,
                np.array(link_i, dtype=np.int64),
                np.array(link_j, dtype=np.int64),
                np.array(link_types, dtype=np.int64),
                self.N
            )
        else:
            p_out = p_matrix
            val_out = val_matrix
        
        # Handle conf_matrix separately (less critical path)
        conf_out = conf_matrix
        if conf_matrix is not None:
            conf_out = conf_matrix.copy()
            for i in range(self.N):
                for j in range(self.N):
                    if (i, 0) in link_assumptions[j]:
                        if link_assumptions[j][(i, 0)] in ["o-o", 'o?o']:
                            if p_matrix[i, j, 0] >= p_matrix[j, i, 0]:
                                conf_out[j, i, 0] = conf_matrix[i, j, 0]
                        elif link_assumptions[j][(i, 0)] in ["-->", '-?>']:
                            conf_out[j, i, 0] = conf_matrix[i, j, 0]

        return {
            'val_matrix': val_out,
            'p_matrix': p_out,
            'conf_matrix': conf_out
        }

    def run_sliding_window_of(self, method, method_args, window_step,
                              window_length, conf_lev=0.9):
        """Runs chosen method on sliding windows."""
        valid_methods = [
            'run_pc_stable', 'run_mci', 'get_lagged_dependencies',
            'run_fullci', 'run_bivci', 'run_pcmci', 'run_pcalg',
            'run_lpcmci', 'run_jpcmciplus', 'run_pcmciplus'
        ]

        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        if self.dataframe.reference_points_is_none is False:
            raise ValueError("Reference points not accepted in sliding windows analysis.")

        T = self.dataframe.largest_time_step

        if self.cond_ind_test.recycle_residuals:
            raise ValueError("cond_ind_test.recycle_residuals must be False.")

        if self.verbosity > 0:
            print(f"\n##\n## Running sliding window analysis of {method}\n##\n"
                  f"\nwindow_step = {window_step}\nwindow_length = {window_length}\n")

        original_reference_points = deepcopy(self.dataframe.reference_points)
        window_start_points = np.arange(0, T - window_length, window_step)
        n_windows = len(window_start_points)

        if len(window_start_points) == 0:
            raise ValueError("Empty list of windows!")

        window_results = {}
        for iw, w in enumerate(window_start_points):
            if self.verbosity > 0:
                print(f"\n# Window start {w} ({iw+1}/{len(window_start_points)})\n")
            
            time_window = np.arange(w, min(w + window_length, T), 1)
            self.dataframe.reference_points = time_window
            window_res = deepcopy(getattr(self, method)(**method_args))

            for key in window_res:
                res_item = window_res[key]
                if iw == 0:
                    if isinstance(res_item, np.ndarray):
                        window_results[key] = np.empty(
                            (n_windows,) + res_item.shape, dtype=res_item.dtype)
                    else:
                        window_results[key] = {}
                window_results[key][iw] = res_item

        self.dataframe.reference_points = original_reference_points
        summary_results = self.return_summary_results(window_results, conf_lev)

        return {'summary_results': summary_results, 'window_results': window_results}

    def run_bootstrap_of(self, method, method_args, boot_samples=100,
                         boot_blocklength=1, conf_lev=0.9,
                         aggregation="majority", seed=None):
        """Runs chosen method on bootstrap samples."""
        valid_methods = [
            'run_pc_stable', 'run_mci', 'get_lagged_dependencies',
            'run_fullci', 'run_bivci', 'run_pcmci', 'run_pcalg',
            'run_pcalg_non_timeseries_data', 'run_pcmciplus',
            'run_lpcmci', 'run_jpcmciplus'
        ]
        
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        T = self.dataframe.largest_time_step
        seed_sequence = np.random.SeedSequence(seed)

        if 'tau_max' not in method_args:
            raise ValueError("tau_max must be explicitly set in method_args.")
        tau_max = method_args['tau_max']

        if self.cond_ind_test.recycle_residuals:
            raise ValueError("cond_ind_test.recycle_residuals must be False.")

        if self.verbosity > 0:
            print(f"\n##\n## Running Bootstrap of {method}\n##\n"
                  f"\nboot_samples = {boot_samples}\nboot_blocklength = {boot_blocklength}\n")

        self.dataframe.bootstrap = {'boot_blocklength': boot_blocklength}
        child_seeds = seed_sequence.spawn(boot_samples)

        # Parallel bootstrap
        aggregated_results = Parallel(n_jobs=-1)(
            delayed(self.parallelized_bootstraps)(method, method_args, boot_seed=child_seeds[b])
            for b in range(boot_samples)
        )

        boot_results = {}
        for b in range(boot_samples):
            boot_res = aggregated_results[b]
            for key in boot_res:
                res_item = boot_res[key]
                if isinstance(res_item, np.ndarray):
                    if b == 0:
                        boot_results[key] = np.empty(
                            (boot_samples,) + res_item.shape, dtype=res_item.dtype)
                    boot_results[key][b] = res_item
                else:
                    if b == 0:
                        boot_results[key] = {}
                    boot_results[key][b] = res_item

        summary_results = self.return_summary_results(boot_results, conf_lev, aggregation)
        self.dataframe.bootstrap = None

        return {'summary_results': summary_results, 'boot_results': boot_results}

    def parallelized_bootstraps(self, method, method_args, boot_seed):
        """Single bootstrap iteration for parallel execution."""
        boot_random_state = np.random.default_rng(boot_seed)
        self.dataframe.bootstrap['random_state'] = boot_random_state
        return getattr(self, method)(**method_args)

    @staticmethod
    def return_summary_results(results, conf_lev=0.9, aggregation="majority"):
        """Return summary results for causal graphs.
        
        Optimized version using Numba kernels for frequency counting.
        """
        valid_aggregations = {"majority", "no_edge_majority"}
        if aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation: {aggregation}")

        summary_results = {}

        if 'graph' in results:
            n_results, N, _, tau_max_plusone = results['graph'].shape
            
            # Encode string graphs to integers for Numba processing
            graphs_encoded = np.zeros((n_results, N, N, tau_max_plusone), dtype=np.int64)
            for b in range(n_results):
                graphs_encoded[b] = encode_graph(results['graph'][b])
            
            # Count frequencies using Numba
            counts = _count_link_frequencies(
                graphs_encoded, n_results, N, tau_max_plusone, N_LINK_TYPES
            )
            
            summary_results['most_frequent_links'] = np.zeros(
                (N, N, tau_max_plusone), dtype=results['graph'][0].dtype
            )
            summary_results['link_frequency'] = np.zeros(
                (N, N, tau_max_plusone), dtype='float'
            )
            
            preferred_order = ["", "x-x", "o-o"]
            
            for (i, j) in itertools.product(range(N), range(N)):
                for abstau in range(tau_max_plusone):
                    link_counts = counts[i, j, abstau, :]
                    max_count = link_counts.max()
                    max_indices = np.where(link_counts == max_count)[0]
                    
                    if aggregation == "majority":
                        if len(max_indices) == 1:
                            choice = INT_TO_LINK_TYPE[max_indices[0]]
                        else:
                            # Tie-breaking
                            candidates = [INT_TO_LINK_TYPE[idx] for idx in max_indices]
                            ordered = [l for l in preferred_order if l in candidates]
                            choice = ordered[0] if ordered else "x-x"
                        
                        summary_results['most_frequent_links'][i, j, abstau] = choice
                        summary_results['link_frequency'][i, j, abstau] = max_count / n_results
                    
                    elif aggregation == "no_edge_majority":
                        freq_no_edge = int(counts[i, j, abstau, LINK_TYPE_TO_INT['']])
                        freq_adjacency = n_results - freq_no_edge
                        
                        if freq_adjacency > freq_no_edge:
                            # Adjacency wins - find most frequent among edges
                            edge_counts = link_counts.copy()
                            edge_counts[LINK_TYPE_TO_INT['']] = 0
                            max_edge_count = edge_counts.max()
                            max_edge_indices = np.where(edge_counts == max_edge_count)[0]
                            
                            if len(max_edge_indices) == 1:
                                choice = INT_TO_LINK_TYPE[max_edge_indices[0]]
                            else:
                                candidates = [INT_TO_LINK_TYPE[idx] for idx in max_edge_indices]
                                ordered = [l for l in preferred_order if l in candidates]
                                choice = ordered[0] if ordered else "x-x"
                            
                            summary_results['most_frequent_links'][i, j, abstau] = choice
                            summary_results['link_frequency'][i, j, abstau] = max_edge_count / n_results
                        else:
                            summary_results['most_frequent_links'][i, j, abstau] = ""
                            summary_results['link_frequency'][i, j, abstau] = freq_no_edge / n_results

        # Confidence intervals using Numba
        c_int = 1. - (1. - conf_lev) / 2.
        val_matrix_stack = results['val_matrix']
        
        summary_results['val_matrix_mean'] = np.mean(val_matrix_stack, axis=0)
        
        q_low = (1. - c_int)
        q_high = c_int
        low, high = _compute_percentiles(val_matrix_stack, q_low, q_high)
        summary_results['val_matrix_interval'] = np.stack([low, high], axis=3)

        return summary_results

    @staticmethod
    def graph_to_dict(graph):
        """Convert graph array to dictionary of links."""
        N = graph.shape[0]
        links = {j: {} for j in range(N)}
        for (i, j, tau) in zip(*np.where(graph != '')):
            links[j][(i, -tau)] = graph[i, j, tau]
        return links

    def _dict_to_graph(self, links, tau_max=None):
        """Convert dictionary of links to graph array."""
        N = len(links)
        max_lag = 0
        
        for j in range(N):
            for link in links[j]:
                var, lag = link
                if isinstance(links[j], dict):
                    link_type = links[j][link]
                    if link_type != "":
                        max_lag = max(max_lag, abs(lag))
                else:
                    max_lag = max(max_lag, abs(lag))

        if tau_max is None:
            tau_max = max_lag
        elif tau_max < max_lag:
            raise ValueError("maxlag(links) > tau_max")

        graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
        graph[:] = ""
        
        for j in range(N):
            for link in links[j]:
                i, tau = link
                link_type = links[j][link] if isinstance(links[j], dict) else '-->'
                graph[i, j, abs(tau)] = link_type

        return graph

    @staticmethod
    def get_graph_from_dict(links, tau_max=None):
        """Convert dictionary of links to graph array format."""
        def _get_minmax_lag(links):
            N = len(links)
            min_lag = np.inf
            max_lag = 0
            for j in range(N):
                for link_props in links[j]:
                    if len(link_props) > 2:
                        var, lag = link_props[0]
                        coeff = link_props[1]
                        if coeff != 0.:
                            min_lag = min(min_lag, abs(lag))
                            max_lag = max(max_lag, abs(lag))
                    else:
                        var, lag = link_props
                        min_lag = min(min_lag, abs(lag))
                        max_lag = max(max_lag, abs(lag))
            return min_lag, max_lag

        N = len(links)
        min_lag, max_lag = _get_minmax_lag(links)

        if tau_max is None:
            tau_max = max_lag
        elif max_lag > tau_max:
            raise ValueError(f"tau_max smaller than maximum lag = {max_lag}")

        graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
        
        for j in links.keys():
            for link_props in links[j]:
                if len(link_props) > 2:
                    var, lag = link_props[0]
                    coeff = link_props[1]
                    if coeff != 0.:
                        graph[var, j, abs(lag)] = "-->"
                        if lag == 0:
                            graph[j, var, 0] = "<--"
                else:
                    var, lag = link_props
                    graph[var, j, abs(lag)] = "-->"
                    if lag == 0:
                        graph[j, var, 0] = "<--"

        return graph

    @staticmethod
    def build_link_assumptions(link_assumptions_absent_link_means_no_knowledge,
                               n_component_time_series, tau_max, tau_min=0):
        """Build link assumptions dictionary."""
        out = {
            j: {
                (i, -tau_i): ("o?>" if tau_i > 0 else "o?o")
                for i in range(n_component_time_series)
                for tau_i in range(tau_min, tau_max + 1)
                if (tau_i > 0 or i != j)
            }
            for j in range(n_component_time_series)
        }

        for j, links_j in link_assumptions_absent_link_means_no_knowledge.items():
            for (i, lag_i), link_ij in links_j.items():
                if link_ij == "":
                    del out[j][(i, lag_i)]
                else:
                    out[j][(i, lag_i)] = link_ij
        
        return out
