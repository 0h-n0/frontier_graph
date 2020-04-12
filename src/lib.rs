extern crate pyo3;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod frontier;

#[pyfunction]
pub fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
struct FrontierInterface {
    num: usize,
}

#[pymethods]
impl FrontierInterface {
    #[new]
    fn new(num: usize) -> Self {
        FrontierInterface { num }
    }
    pub fn method(&self) -> PyResult<Vec<usize>> {
        Ok(vec![1, 2, 3, 4])
    }
}
#[pyfunction]
pub fn calc_frontier_combination(
    number_of_vertices: usize,
    edge_list: Vec<(usize, usize)>,
    srcs: Vec<usize>,
    dsts: Vec<usize>,
    n_samples: usize) -> PyResult<Vec<Vec<usize>>> {
    Ok(frontier::calc_frontier_combination(number_of_vertices, edge_list, srcs, dsts, n_samples))
}

#[pymodule]
fn frontier(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;
    m.add_class::<FrontierInterface>()?;
    m.add_wrapped(wrap_pyfunction!(calc_frontier_combination))?;
    Ok(())
}
