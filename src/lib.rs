extern crate pyo3;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod frontier;


#[pyfunction]
pub fn calc_frontier_combination(
    number_of_vertices: usize,
    edge_list: Vec<(usize, usize)>,
    srcs: Vec<usize>,
    dsts: Vec<usize>,
    n_samples: usize,
    fixed_edges: Option<Vec<(usize, usize)>>) -> PyResult<Vec<Vec<usize>>> {
    Ok(frontier::calc_frontier_combination(number_of_vertices, edge_list, srcs, dsts, n_samples, fixed_edges))
}

#[pymodule]
fn frontier(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(calc_frontier_combination))?;
    Ok(())
}
