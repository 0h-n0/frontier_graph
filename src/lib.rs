extern crate pyo3;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
pub fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn frontier(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let a = sum_as_string(1, 2);
        assert_eq!("3".to_string(), a.unwrap());
    }
}
