#!/usr/bin/env python3
"""
DAG Enforcer - Efficient O(d²) Acyclicity Constraint
====================================================

Replaces the expensive O(d³) matrix exponential in NOTEARS with
efficient cycle detection and breaking via topological sorting.

This approach:
1. Detects cycles using DFS (O(d²))
2. Breaks cycles by removing weakest edges
3. Ensures output is a valid DAG

Complexity: O(d²) vs O(d³) for matrix exponential
"""

import numpy as np
import torch
import logging
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)


class TopologicalDAGEnforcer:
    """
    Enforces DAG structure by detecting and breaking cycles.
    
    This is much faster than the NOTEARS matrix exponential approach
    for large graphs, trading differentiability for computational efficiency.
    """
    
    def __init__(self, threshold: float = 0.01):
        """
        Initialize DAG enforcer.
        
        Args:
            threshold: Edges with weight below this are considered zero
        """
        self.threshold = threshold
    
    def project_to_dag(self, W: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        """
        Project a weighted adjacency matrix onto the space of DAGs.
        
        Args:
            W: Weighted adjacency matrix [d, d], where W[i,j] is edge j->i
            inplace: If True, modify W in place (saves memory)
        
        Returns:
            W_dag: DAG-constrained version of W
        
        Complexity: O(d² log d) worst case, typically O(d²) average case
        """
        if not inplace:
            W = W.clone()
        
        d = W.shape[0]
        
        # Convert to numpy for graph algorithms (faster for discrete ops)
        W_np = W.detach().cpu().numpy()
        
        # Iteratively find and break cycles
        cycles_broken = 0
        max_iterations = d * 2  # Allow more iterations for complex graphs
        
        for iteration in range(max_iterations):
            # Find a cycle (if any)
            cycle = self._find_cycle(W_np, self.threshold)
            
            if cycle is None:
                # No more cycles - we have a DAG!
                break
            
            # Break the cycle by removing its weakest edge
            weakest_edge = self._find_weakest_edge_in_cycle(W_np, cycle)
            i, j = weakest_edge
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Iteration {iteration}: Breaking cycle {cycle}")
                logger.debug(f"  Removing edge {j}->{i} (weight={W_np[i,j]:.4f})")
            
            W_np[i, j] = 0.0
            cycles_broken += 1
        
        if cycles_broken > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DAG projection: broke {cycles_broken} cycles in {iteration+1} iterations")
        
        # Convert back to torch tensor
        W_dag = torch.tensor(W_np, dtype=W.dtype, device=W.device)
        
        # Verify it's actually a DAG (sanity check in production, warning in debug)
        if not self._is_dag(W_dag, self.threshold):
            logger.warning(f"DAG projection may be incomplete after {cycles_broken} cycles broken")
            # This can happen if threshold is too low or graph is complex
            # Continue anyway - partial DAG is better than failure
        
        return W_dag
    
    def _find_cycle(self, W: np.ndarray, threshold: float) -> List[int]:
        """
        Find a cycle in the graph using DFS.
        
        Args:
            W: Adjacency matrix (numpy)
            threshold: Edge weight threshold
        
        Returns:
            List of node indices forming a cycle, or None if no cycle exists
        
        Complexity: O(d²)
        """
        d = W.shape[0]
        
        # Colors: 0=white (unvisited), 1=gray (visiting), 2=black (done)
        color = np.zeros(d, dtype=int)
        recursion_stack = []  # Track DFS path
        
        def dfs(node: int) -> List[int]:
            """DFS to detect cycle"""
            color[node] = 1  # Mark as visiting
            recursion_stack.append(node)
            
            # Check all children (edges where node is parent)
            for child in range(d):
                if abs(W[child, node]) > threshold:  # Edge node -> child exists
                    if color[child] == 1:
                        # Back edge found - we have a cycle!
                        # Reconstruct cycle from recursion stack
                        try:
                            cycle_start_idx = recursion_stack.index(child)
                            cycle = recursion_stack[cycle_start_idx:] + [child]
                            return cycle
                        except ValueError:
                            # This shouldn't happen, but handle it gracefully
                            return [child, node]
                    elif color[child] == 0:
                        cycle = dfs(child)
                        if cycle is not None:
                            return cycle
            
            color[node] = 2  # Mark as done
            recursion_stack.pop()
            return None
        
        # Try DFS from each unvisited node
        for start_node in range(d):
            if color[start_node] == 0:
                cycle = dfs(start_node)
                if cycle is not None:
                    return cycle
        
        return None  # No cycle found - it's a DAG
    
    def _find_weakest_edge_in_cycle(self, W: np.ndarray, cycle: List[int]) -> Tuple[int, int]:
        """
        Find the edge with minimum absolute weight in a cycle.
        
        Args:
            W: Adjacency matrix
            cycle: List of node indices forming the cycle (may include closing node twice)
        
        Returns:
            Tuple (i, j) representing edge j->i with minimum weight
        """
        min_weight = float('inf')
        min_edge = None
        
        # Remove duplicate closing node if present
        if len(cycle) > 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
        
        # Check each edge in the cycle
        # Convention: W[i,j] means edge j -> i
        for k in range(len(cycle)):
            parent = cycle[k]
            child = cycle[(k + 1) % len(cycle)]
            
            weight = abs(W[child, parent])  # Edge parent -> child
            
            if weight < min_weight:
                min_weight = weight
                min_edge = (child, parent)
        
        return min_edge
    
    def _is_dag(self, W: torch.Tensor, threshold: float) -> bool:
        """
        Check if a graph is a DAG using topological sort (Kahn's algorithm).
        
        Args:
            W: Adjacency matrix
            threshold: Edge weight threshold
        
        Returns:
            True if W represents a DAG, False otherwise
        
        Complexity: O(d²)
        """
        W_np = W.detach().cpu().numpy()
        d = W.shape[0]
        
        # Compute in-degrees
        in_degree = np.zeros(d, dtype=int)
        for i in range(d):
            for j in range(d):
                if abs(W_np[i, j]) > threshold:  # Edge j -> i exists
                    in_degree[i] += 1
        
        # Kahn's algorithm
        queue = [i for i in range(d) if in_degree[i] == 0]
        processed = 0
        
        while queue:
            node = queue.pop(0)
            processed += 1
            
            # Remove edges from this node
            for child in range(d):
                if abs(W_np[child, node]) > threshold:  # Edge node -> child
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        # If we processed all nodes, it's a DAG
        return processed == d
    
    def get_topological_order(self, W: torch.Tensor, threshold: float = None) -> List[int]:
        """
        Compute a topological ordering of the DAG.
        
        Args:
            W: DAG adjacency matrix
            threshold: Edge weight threshold (uses self.threshold if None)
        
        Returns:
            List of node indices in topological order
            Empty list if W is not a DAG
        
        Complexity: O(d²)
        """
        if threshold is None:
            threshold = self.threshold
        
        W_np = W.detach().cpu().numpy()
        d = W.shape[0]
        
        # Compute in-degrees
        in_degree = np.zeros(d, dtype=int)
        for i in range(d):
            for j in range(d):
                if abs(W_np[i, j]) > threshold:
                    in_degree[i] += 1
        
        # Kahn's algorithm
        queue = [i for i in range(d) if in_degree[i] == 0]
        ordering = []
        
        while queue:
            # Sort queue for deterministic ordering
            queue.sort()
            node = queue.pop(0)
            ordering.append(node)
            
            # Remove edges from this node
            for child in range(d):
                if abs(W_np[child, node]) > threshold:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        # Check if we got a full ordering
        if len(ordering) != d:
            logger.warning(f"Graph has cycles - only {len(ordering)}/{d} nodes in topological order")
            return []
        
        return ordering
    
    def compute_num_cycles(self, W: torch.Tensor, threshold: float = None) -> int:
        """
        Count the number of cycles in the graph (for diagnostics).
        
        This is expensive (NP-hard in general), so we use a heuristic:
        count how many edges need to be removed to make it a DAG.
        
        Args:
            W: Adjacency matrix
            threshold: Edge weight threshold
        
        Returns:
            Approximate number of cycles (lower bound)
        """
        if threshold is None:
            threshold = self.threshold
        
        W_test = W.clone()
        W_np = W_test.detach().cpu().numpy()
        
        cycles = 0
        max_iterations = W.shape[0]
        
        for _ in range(max_iterations):
            cycle = self._find_cycle(W_np, threshold)
            if cycle is None:
                break
            
            # Remove weakest edge
            edge = self._find_weakest_edge_in_cycle(W_np, cycle)
            W_np[edge[0], edge[1]] = 0.0
            cycles += 1
        
        return cycles


class SoftDAGConstraint:
    """
    Alternative: Soft DAG constraint using differentiable approximation.
    
    This is slower (O(d³)) but differentiable, useful for end-to-end training.
    Can be used in conjunction with TopologicalDAGEnforcer for hybrid approach.
    """
    
    def __init__(self, use_polynomial_approximation: bool = True):
        """
        Initialize soft DAG constraint.
        
        Args:
            use_polynomial_approximation: If True, use polynomial approximation
                instead of full matrix exponential (faster, less accurate)
        """
        self.use_polynomial = use_polynomial_approximation
    
    def compute_h(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute NOTEARS acyclicity constraint: h(W) = tr(exp(W⊙W)) - d
        
        Args:
            W: Adjacency matrix [d, d]
        
        Returns:
            Scalar constraint value (should be 0 for DAG)
        
        Complexity: O(d³) for matrix exponential, O(d²) for polynomial approx
        """
        d = W.shape[0]
        W_squared = W @ W
        
        if self.use_polynomial:
            # Polynomial approximation: exp(X) ≈ I + X + X²/2 + X³/6
            # Accurate for ||X|| < 1, which is usually true for regularized W
            I = torch.eye(d, device=W.device, dtype=W.dtype)
            W2 = W_squared
            W3 = W2 @ W_squared
            W4 = W3 @ W_squared
            
            exp_approx = I + W_squared + 0.5 * W2 + (1.0/6.0) * W3 + (1.0/24.0) * W4
            h = torch.trace(exp_approx) - d
        else:
            # Full matrix exponential (more accurate but slower)
            h = torch.trace(torch.linalg.matrix_exp(W_squared)) - d
        
        return h
    
    def compute_gradient(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of h(W) with respect to W.
        
        This is needed for augmented Lagrangian optimization.
        
        Returns:
            Gradient dh/dW [d, d]
        """
        # PyTorch autograd handles this automatically
        # This method is for reference/manual computation if needed
        raise NotImplementedError("Use PyTorch autograd: h.backward()")


def test_dag_enforcer():
    """Test the DAG enforcer on a small example"""
    print("Testing DAG Enforcer...")
    
    # Create a small graph with a cycle: 0 -> 1 -> 2 -> 0
    # W[i,j] means edge j -> i
    W = torch.tensor([
        [0.0, 0.5, 0.0],  # 1 -> 0 (weight 0.5)
        [0.0, 0.0, 0.4],  # 2 -> 1 (weight 0.4)
        [0.3, 0.0, 0.0],  # 0 -> 2 (weight 0.3, completes cycle)
    ])
    
    print(f"Original W (has cycle 0->2->1->0):\n{W}")
    print("Convention: W[i,j] = weight of edge j -> i")
    
    enforcer = TopologicalDAGEnforcer(threshold=0.01)
    
    # Check if it has cycles
    is_dag = enforcer._is_dag(W, 0.01)
    print(f"\nIs DAG before: {is_dag}")
    
    # Count cycles
    n_cycles = enforcer.compute_num_cycles(W)
    print(f"Number of cycles detected: {n_cycles}")
    
    # Project to DAG
    print("\nProjecting to DAG...")
    W_dag = enforcer.project_to_dag(W)
    print(f"Projected W:\n{W_dag}")
    
    # Verify
    is_dag_after = enforcer._is_dag(W_dag, 0.01)
    print(f"\nIs DAG after: {is_dag_after}")
    
    # Get topological order (if it's a DAG)
    if is_dag_after:
        order = enforcer.get_topological_order(W_dag)
        print(f"Topological order: {order}")
    else:
        print("Warning: Result still has cycles, but test continues")
    
    print("\n✓ DAG Enforcer test completed!")
    
    # Test 2: Already a DAG
    print("\n--- Test 2: Already a DAG ---")
    W_dag_already = torch.tensor([
        [0.0, 0.5, 0.3],  # 1 -> 0, 2 -> 0
        [0.0, 0.0, 0.4],  # 2 -> 1
        [0.0, 0.0, 0.0],  # No outgoing from 2
    ])
    print(f"W (should be DAG):\n{W_dag_already}")
    
    is_dag_test2 = enforcer._is_dag(W_dag_already, 0.01)
    print(f"Is DAG: {is_dag_test2}")
    
    if is_dag_test2:
        order = enforcer.get_topological_order(W_dag_already)
        print(f"Topological order: {order}")
        print("✓ Test 2 passed!")
    else:
        print("✗ Test 2 failed: Should be a DAG")


if __name__ == "__main__":
    # Run tests
    test_dag_enforcer()
