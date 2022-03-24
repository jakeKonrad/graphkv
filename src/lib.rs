// glzip is a graph compression library for graph learning systems
// Copyright (C) 2022 Jacob Konrad
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::{
    error::Error,
    convert::TryInto
};

use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError,
};

use numpy::{
    PyReadonlyArray1,
    PyReadonlyArray2,
    NotContiguousError,
    ndarray::Axis,
};

use glzip as core;

fn raise<E: Error>(err: E) -> PyErr
{
    PyRuntimeError::new_err(err.to_string())
}

#[pyclass(module="glzip")]
struct CSR 
{
    csr: core::CSR,
}

#[pymethods]
impl CSR
{
    #[new]
    #[args(
        "*",
        edge_index = "None",
        indptr = "None",
        indices = "None",
    )]
    fn new<'py>(
        edge_index: Option<PyReadonlyArray2<'py, i64>>,
        indptr: Option<PyReadonlyArray1<'py, i64>>,
        indices: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<Self>
    {
        if let Some(edge_index) = edge_index {
            let arr = edge_index.as_array();
            let src = arr.index_axis(Axis(0), 0).to_slice().ok_or(NotContiguousError)?;
            let dst = arr.index_axis(Axis(0), 1).to_slice().ok_or(NotContiguousError)?;
            let csr = core::CSR::try_from_edge_index(src, dst).map_err(raise)?;
            Ok(Self { csr })
        }
        else if let (Some(indptr), Some(indices)) = (indptr, indices) {
            let indptr = indptr.as_slice()?;
            let indices = indices.as_slice()?;
            let csr = core::CSR::try_from_csr(indptr, indices).map_err(raise)?;
            Ok(Self { csr })
        }
        else {
            Err(PyRuntimeError::new_err("provide either edge_index or indptr and indices"))
        }
    }

    fn __str__(&self) -> String
    {
        format!("glzip.CSR(order={}, size={}, nbytes={})", self.csr.order(), self.csr.size(), self.csr.nbytes())
    }

    #[getter]
    fn nbytes(&self) -> usize
    {
        self.csr.nbytes()
    }

    #[getter]
    fn order(&self) -> usize
    {
        self.csr.order()
    }

    #[getter]
    fn size(&self) -> usize
    {
        self.csr.size()
    }

    fn optimize<'py>(
        &self,
        train_idx: PyReadonlyArray1<'py, i64>,
        sizes: Vec<usize>
    ) -> PyResult<(Self, Vec<u32>)>
    {
        let mut train_idx_bitmap: Vec<bool> = std::iter::repeat(false).take(self.order()).collect();
        for &i in train_idx.as_slice()? {
            let i: usize = i.try_into()?;
            train_idx_bitmap[i] = true;
        }

        let (csr, eid) = self.csr.optimize(&train_idx_bitmap[..], &sizes[..]);

        Ok((Self { csr }, eid))
    }
}   

#[pymodule]
fn glzip(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CSR>()?;
    Ok(())
}
