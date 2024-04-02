from typing import Tuple

from pwarp.core import ops_torch
from pwarp.core import dtype_torch
import torch

__all__ = (
    'StepOne',
    'StepTwo',
)


class StepOne_torch(object):
    @staticmethod
    def compute_g_matrix(
            vertices: torch.tensor,
            edges: torch.tensor,
            faces: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        The paper requires to compute expression (G.T)^{-1} @ G.T = X.
        The problem might be solved by solving equation (G.T @ G) @ X = G.T, hence
        we can simply use torch.linalg.lstsq(G.T @ G, G.T, ...).

        :param vertices: torch.tensor;
        :param edges: torch.tensor;
        :param faces: torch.tensor;
        :return: Tuple[torch.tensor, torch.tensor];

        ::

            gi represents indices of edges that contains result
            for expression (G.T)^{-1} @ G.T in g_product
        """
        g_product = torch.zeros((edges.shape[0], 2, 8), dtype=dtype_torch.FLOAT)
        gi = torch.zeros((edges.shape[0], 4), dtype=dtype_torch.FLOAT)

        if edges.dtype not in [dtype_torch.INDEX]:
            raise ValueError('Invalid dtype of edge indices. Requires torch.long')

        # Compute G_k matrix for each `k`.
        for k, edge in enumerate(edges):
            i_vert, j_vert = vertices[edge]
            i_index, j_index = edge

            l_index, r_index = ops_torch.find_ijlr_vertices(edge, faces)
            l_vert = vertices[l_index]

            # For 3 neighbour points (when at the graph edge).
            if torch.isnan(torch.tensor(r_index)).any():
                # g = torch.tensor([[i_vert[0], i_vert[1], 1, 0],
                #               [i_vert[1], -i_vert[0], 0, 1],
                #               [j_vert[0], j_vert[1], 1, 0],
                #               [j_vert[1], -j_vert[0], 0, 1],
                #               [l_vert[0], l_vert[1], 1, 0],
                #               [l_vert[1], -l_vert[0], 0, 1]],
                #              dtype=dtype_torch.FLOAT)
                mask = torch.tensor([
                    [1.0,  1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0, 1.0],
                ]).repeat(3, 1)
                g = torch.cat((
                    i_vert, torch.flip(i_vert, dims=[0]),
                    j_vert, torch.flip(j_vert, dims=[0]),
                    l_vert, torch.flip(l_vert, dims=[0]),
                ), dim=0).reshape(6, 2)
                g = torch.cat((g, torch.ones((6, 2))), dim=1) * mask
                _slice = 6
            # For 4 neighbour points (when not at the graph edge).
            else:
                r_vert = vertices[r_index]
                # g = torch.tensor([[i_vert[0], i_vert[1], 1, 0],
                #               [i_vert[1], -i_vert[0], 0, 1],
                #               [j_vert[0], j_vert[1], 1, 0],
                #               [j_vert[1], -j_vert[0], 0, 1],
                #               [l_vert[0], l_vert[1], 1, 0],
                #               [l_vert[1], -l_vert[0], 0, 1],
                #               [r_vert[0], r_vert[1], 1, 0],
                #               [r_vert[1], -r_vert[0], 0, 1]],
                #              dtype=dtype_torch.FLOAT)
                mask = torch.tensor([
                    [1.0,  1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0, 1.0],
                ]).repeat(4, 1)
                g = torch.cat((
                    i_vert, torch.flip(i_vert, dims=[0]),
                    j_vert, torch.flip(j_vert, dims=[0]),
                    l_vert, torch.flip(l_vert, dims=[0]),
                    r_vert, torch.flip(r_vert, dims=[0]),
                ), dim=0).reshape(8, 2)
                g = torch.cat((g, torch.ones((8, 2))), dim=1) * mask
                _slice = 8

            # G[k,:,:]
            gi[k, :] = torch.tensor([i_index, j_index, l_index, torch.nan if torch.isnan(torch.tensor(r_index)).any() else r_index])
            g = g.to(dtype_torch.FLOAT64)
            x_matrix_pad = torch.linalg.lstsq(g.T @ g, g.T, rcond=None, driver='gels').solution
            g_product[k, :, :_slice] = x_matrix_pad[0:2]

        return gi, g_product

    @staticmethod
    def compute_h_matrix(
            edges: torch.tensor,
            g_product: torch.tensor,
            gi: torch.tensor,
            vertices: torch.tensor
    ) -> torch.tensor:
        """
        Transformed term (v′_j − v′_i) − T_{ij} (v_j − v_i) from paper requires
        computation of matrix H. To be able compute matrix H, we need matrix G
        from other method.

        :param edges: torch.tensor; requires dtype torch.long
        :param g_product: torch.tensor;
        :param gi: torch.tensor;
        :param vertices: torch.tensor;
        :return: torch.tensor;
        """
        mask = torch.tensor([[1.0,  1.0], [1.0, -1.0]])
        h_matrix = torch.zeros((edges.shape[0] * 2, 8), dtype=dtype_torch.FLOAT)
        for k, edge in enumerate(edges):
            # ...where e is an edge vector..
            ek = torch.sub(*vertices[torch.flip(edge, dims=[0])])
            # ek_matrix = torch.tensor([[ek[0], ek[1]], [ek[1], -ek[0]]], dtype=dtype_torch.FLOAT)
            ek_matrix = torch.cat((ek, torch.flip(ek, dims=[0])), dim=0).reshape(2, 2) * mask

            # Ful llength of ones/zero matrix (will be sliced in case on the contour of graph).
            _oz = torch.tensor([[-1, 0, 1, 0, 0, 0, 0, 0],
                            [0, -1, 0, 1, 0, 0, 0, 0]],
                           dtype=dtype_torch.FLOAT)
            if torch.isnan(gi[k, 3]).any():
                _slice = 6
            else:
                _slice = 8

            g = g_product[k, :, :_slice]
            oz = _oz[:, :_slice]
            h_calc = oz - (ek_matrix @ g)
            h_matrix[k * 2, :_slice] = h_calc[0]
            h_matrix[k * 2 + 1, :_slice] = h_calc[1]

        return h_matrix

    @staticmethod
    def compute_A_matrix(
        edges: torch.tensor,
        vertices: torch.tensor,
        gi: torch.tensor,
        h_matrix: torch.tensor,
        c_indices: torch.tensor,
        weight: dtype_torch.FLOAT = torch.tensor(1000., dtype=dtype_torch.FLOAT),
    ):
        a1_matrix = torch.zeros((edges.shape[0] * 2 + c_indices.shape[0] * 2, vertices.shape[0] * 2), dtype=dtype_torch.FLOAT)

        # Fill values in prepared matrices/vectors
        for k, g_indices in enumerate(gi):
            for i, point_index in enumerate(g_indices):
                if not torch.isnan(point_index).any():
                    point_index = int(point_index)
                    a1_matrix[k * 2, point_index * 2] = h_matrix[k * 2, i * 2]
                    a1_matrix[k * 2 + 1, point_index * 2] = h_matrix[k * 2 + 1, i * 2]
                    a1_matrix[k * 2, point_index * 2 + 1] = h_matrix[k * 2, i * 2 + 1]
                    a1_matrix[k * 2 + 1, point_index * 2 + 1] = h_matrix[k * 2 + 1, i * 2 + 1]
        for c_enum_index, c_vertex_index in enumerate(c_indices):
            # Set weights for given position of control point.
            a1_matrix[edges.shape[0] * 2 + c_enum_index * 2, c_vertex_index * 2] = weight
            a1_matrix[edges.shape[0] * 2 + c_enum_index * 2 + 1, c_vertex_index * 2 + 1] = weight
        return a1_matrix

    @staticmethod
    def compute_v_prime(
            edges: torch.tensor,
            vertices: torch.tensor,
            gi: torch.tensor,
            h_matrix: torch.tensor,
            c_indices: torch.tensor,
            c_vertices: torch.tensor,
            a1_matrix: torch.tensor,
            weight: dtype_torch.FLOAT = torch.tensor(1000., dtype=dtype_torch.FLOAT),
            device: torch.device = torch.device('cpu')
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        [√] TODO:
            - make this method cacheable on A and b matrices.

        The cookbook from paper requires to compute expression `A1 @ v′ = b1`.
        Regards to the paper, we will compute v'.

        For more information see page 24 of paper on::

            https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf

        Warning: The position of h_{kuv} in matrix A of paper is based on position of given points (in k-th edge
        surounding points) in original vertices. It just demonstrates that 01, 23, 45, 67 indices of H row form a pair,
        but you have to put them on valid position or be aware of ordering of results in v' vector.

        :return: Tuple[torch.tensor, torch.tensor, torch.tensor]; (v_prime, A matrix, b vector)
        """
        # Prepare defaults.
        bs = c_vertices.shape[0]
        # a1_matrix = torch.zeros((edges.shape[0] * 2 + c_indices.shape[0] * 2, vertices.shape[0] * 2), dtype=dtype_torch.FLOAT).to(device)
        b1_vector = torch.zeros((bs, edges.shape[0] * 2 + c_indices.shape[0] * 2, 1), dtype=dtype_torch.FLOAT).to(device)
        v_prime = torch.zeros((bs, vertices.shape[0], 2), dtype=dtype_torch.FLOAT).to(device)

        # Fill values in prepared matrices/vectors
        for k, g_indices in enumerate(gi):
            for i, point_index in enumerate(g_indices):
                if not torch.isnan(point_index).any():
                    point_index = int(point_index)
                    # In the h_matrix we have stored values for index k (edge index) in following form:
                    # for k = 0, two lines of h_matrix are going one after another like
                    # [k00, k10, k20, ..., k70] forlowed by [k01, k11, k21, ..., k71], hence we have to access values
                    # via (k * 2), and (k * 2 + 1).

                    # Variable point_index represent index from original vertices set of vertex from
                    # 4 neighbours of k-th edge.
                    # Index i represents an index of point from 4 (3 in case of contour) neighbours in k-th set.

                    # The row in the A matrix is defiend by index k. Since we have stored for given k index two rows
                    # of H matrix, we have to work with indexing of (k * 2) and (k * 2 + 1).

                    # The column of H matrix is accessible via index i, since H row is 0 - 7 indices long. Than
                    # (i * 2) and (i * 2 + 1) will access h valus for given point in given h row.

                    a1_matrix[k * 2, point_index * 2] = h_matrix[k * 2, i * 2]
                    a1_matrix[k * 2 + 1, point_index * 2] = h_matrix[k * 2 + 1, i * 2]
                    a1_matrix[k * 2, point_index * 2 + 1] = h_matrix[k * 2, i * 2 + 1]
                    a1_matrix[k * 2 + 1, point_index * 2 + 1] = h_matrix[k * 2 + 1, i * 2 + 1]

        for c_enum_index, c_vertex_index in enumerate(c_indices):
            # Set weights for given position of control point.
            # Do the same for values of b_vector
            b1_vector[:, edges.shape[0] * 2 + c_enum_index * 2] = weight * c_vertices[:, c_enum_index, 0].reshape(bs, 1)
            b1_vector[:, edges.shape[0] * 2 + c_enum_index * 2 + 1] = weight * c_vertices[:, c_enum_index, 1].reshape(bs, 1)

        a1_matrix = a1_matrix.to(dtype_torch.FLOAT64)
        b1_vector = b1_vector.to(dtype_torch.FLOAT64)
        v = torch.linalg.lstsq((a1_matrix.T @ a1_matrix).unsqueeze(0), a1_matrix.T @ b1_vector, rcond=None, driver='gels').solution
        v_prime[:, :, 0] = v[:, 0::2, 0]
        v_prime[:, :, 1] = v[:, 1::2, 0]

        return v_prime, a1_matrix, b1_vector


class StepTwo_torch(object):
    @staticmethod
    def compute_t_matrix(
            edges: torch.tensor,
            g_product: torch.tensor,
            gi: torch.tensor,
            v_prime: torch.tensor,
            device: torch.device = torch.device('cpu')
    ) -> torch.tensor:
        """
        From paper:

        The second step takes the rotation information from the result of the first step
        (i.e., computing the explicit values of T′k and normalizing them to remove the
        scaling factor), rotates the original edge vectors ek by the amount T′k, and
        then solves Equation (1) using the original rotated edge vectors. That is, we
        compute the rotation of each edge by using the result of the first step,
        and then normalize it.
        
        :param edges: torch.tensor; 
        :param g_product: torch.tensor; 
        :param gi: torch.tensor;
        :param v_prime: torch.tensor; transformed point in sense of rotation from step one
        :return: torch.tensor;
        """
        bs = v_prime.shape[0]
        t_matrix = torch.zeros((bs, edges.shape[0], 2, 2), dtype=dtype_torch.FLOAT).to(device)
        mask = torch.tensor([[1.0,  1.0], [-1.0, 1.0]]).unsqueeze(0).to(device)
        # We compute T′k for each edge.
        for k, edge in enumerate(edges):
            if torch.isnan(gi[k, 3]).any():
                _slice = 6
                # v = torch.tensor([
                #     [v_prime[int(gi[k, 0]), 0]],
                #     [v_prime[int(gi[k, 0]), 1]],
                #     [v_prime[int(gi[k, 1]), 0]],
                #     [v_prime[int(gi[k, 1]), 1]],
                #     [v_prime[int(gi[k, 2]), 0]],
                #     [v_prime[int(gi[k, 2]), 1]]
                # ], dtype=dtype_torch.FLOAT)
                v = torch.cat((
                    v_prime[:, int(gi[k, 0])],
                    v_prime[:, int(gi[k, 1])],
                    v_prime[:, int(gi[k, 2])],
                ), dim=1).reshape(bs, 6, 1)
            else:
                _slice = 8
                # v = torch.tensor([
                #     [v_prime[int(gi[k, 0]), 0]],
                #     [v_prime[int(gi[k, 0]), 1]],
                #     [v_prime[int(gi[k, 1]), 0]],
                #     [v_prime[int(gi[k, 1]), 1]],
                #     [v_prime[int(gi[k, 2]), 0]],
                #     [v_prime[int(gi[k, 2]), 1]],
                #     [v_prime[int(gi[k, 3]), 0]],
                #     [v_prime[int(gi[k, 3]), 1]]],
                #     dtype=dtype_torch.FLOAT
                # )
                v = torch.cat((
                    v_prime[:, int(gi[k, 0])],
                    v_prime[:, int(gi[k, 1])],
                    v_prime[:, int(gi[k, 2])],
                    v_prime[:, int(gi[k, 3])],
                ), dim=1).reshape(bs, 8, 1)
            # We compute the rotation of each edge by using the result of the first step,
            g = g_product[k, :, :_slice]
            t = g @ v
            # rot = torch.tensor([[t[0], t[1]], [-t[1], t[0]]], dtype=dtype_torch.FLOAT)
            rot = torch.cat((t, torch.flip(t, dims=[1])), dim=1).reshape(-1, 2, 2) * mask
            # and then normalize it.
            # t_normalized = (torch.tensor(1, dtype=dtype_torch.FLOAT) / torch.sqrt(torch.pow(t[0], 2) + torch.pow(t[1], 2))) * rot
            t_normalized = (1.0 / torch.sqrt(torch.pow(t, 2).sum(dim=1))).unsqueeze(1) * rot
            # Store result.
            t_matrix[:, k, :, :] = t_normalized
        return t_matrix

    @staticmethod
    def compute_v_2prime(
            edges: torch.tensor,
            vertices: torch.tensor,
            t_matrix: torch.tensor,
            c_indices: torch.tensor,
            c_vertices: torch.tensor,
            weight: dtype_torch.FLOAT = torch.tensor(1000., dtype=dtype_torch.FLOAT),
            device: torch.device = torch.device('cpu')
    ) -> torch.tensor:
        """

        :param edges: torch.tensor;
        :param vertices: torch.tensor;
        :param t_matrix: torch.tensor;
        :param c_indices: torch.tensor;
        :param c_vertices: torch.tensor;
        :param weight: torch.float; dtype_torch.FLOAT
        :return: torch.tensor;
        """
        # Prepare blueprints.
        bs = c_vertices.shape[0]
        a2_matrix = torch.zeros((edges.shape[0] + c_indices.shape[0], vertices.shape[0]), dtype=dtype_torch.FLOAT).to(device)
        b2_vector = torch.zeros((bs, edges.shape[0] + c_indices.shape[0], 2), dtype=dtype_torch.FLOAT).to(device)

        # Update values from precomputed components.
        # Matrix A2 is identical for both x- and y- components.
        for k, edge in enumerate(edges):
            # The values are set due to optimization equation from paper, where
            # arg min {sum_{i,j}||(v_j'' - v_i'')  ... ||} what gives 1 to position
            # of edge 0 and -1 to position of edge 1 and finally we will obtain (v_j'' - v_i'').
            a2_matrix[k, int(edge[0])] = -1.0
            a2_matrix[k, int(edge[1])] = 1.0

            e = vertices[int(edge[1])] - vertices[int(edge[0])]
            t_e = t_matrix[:, k, :, :] @ e.unsqueeze(-1)
            b2_vector[:, k, :] = t_e.squeeze(-1)

        for c_index, c in enumerate(c_indices):
            a2_matrix[edges.shape[0] + c_index, c] = weight
            b2_vector[:, edges.shape[0] + c_index, :] = weight * c_vertices[:, c_index, :]

        a2_matrix = a2_matrix.to(dtype_torch.FLOAT64)
        b2_vector = b2_vector.to(dtype_torch.FLOAT64)
        return torch.linalg.lstsq((a2_matrix.T @ a2_matrix).unsqueeze(0), a2_matrix.T @ b2_vector, rcond=None, driver='gels').solution
