use ndarray::ArrayView2;

/// Path-halving find: iterative, no stack allocation.
/// Makes every other node on the path point to its grandparent.
#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize];
        if p == x {
            return x;
        }
        // Path-halving: point x to its grandparent
        let gp = parent[p as usize];
        parent[x as usize] = gp;
        x = gp;
    }
}

/// Find connected components of a triangle mesh.
///
/// Uses Union-Find (DSU) with path-halving. The only extra allocation is a
/// single `Vec<u32>` of length `n_vertices` for the parent array — no
/// adjacency list is built.
///
/// Arguments
/// ---------
/// - `faces`:       (N, 3) array of triangular faces given as vertex indices.
/// - `n_vertices`:  Total number of vertices.
///
/// Returns
/// -------
/// A `Vec<u32>` of length `n_vertices` where each entry contains the
/// root-vertex index of the component the vertex belongs to.
pub fn mesh_connected_components(faces: ArrayView2<u32>, n_vertices: usize) -> Vec<u32> {
    // Each vertex is its own parent initially — the only allocation.
    let mut parent: Vec<u32> = (0..n_vertices as u32).collect();

    // Walk every face and union the three vertices.
    for face in faces.rows() {
        let a = face[0];
        let b = face[1];
        let c = face[2];

        // Union a–b
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            // Attach larger root to smaller root so root IDs are consistent.
            if ra < rb {
                parent[rb as usize] = ra;
            } else {
                parent[ra as usize] = rb;
            }
        }

        // Union a–c  (re-find ra since it may have changed)
        let ra2 = find(&mut parent, a);
        let rc = find(&mut parent, c);
        if ra2 != rc {
            if ra2 < rc {
                parent[rc as usize] = ra2;
            } else {
                parent[ra2 as usize] = rc;
            }
        }
    }

    // Final compression: make every vertex point directly to its root.
    for i in 0..n_vertices {
        parent[i] = find(&mut parent, i as u32);
    }

    parent
}
