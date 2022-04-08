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

use std::convert::TryInto;

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use numpy::{
    ndarray::{self, Axis, Array2, Array1}, 
    NotContiguousError, 
    PyArray1,
    PyArray2,
    PyReadonlyArray1, 
    PyReadonlyArray2
};

use glzip as core;

fn core_from_csr(indptr: &[i64], indices: &[i64]) -> PyResult<core::CSR> {
    core::CSR::try_from_edges_with_capacity(
        indices.len(),
        indptr
            .windows(2)
            .enumerate()
            .flat_map(|(v, window)| {
                indices[window[0] as usize..window[1] as usize].iter().map(move |&u| {
                    let v = v.try_into()?;
                    let u = u.try_into()?;
                    Ok(core::Edge(v, u))
                })
            }),
    )
}

fn core_from_edge_index(src: &[i64], dst: &[i64]) -> PyResult<core::CSR> {
    core::CSR::try_from_edges_with_capacity(
        std::cmp::min(src.len(), dst.len()),
        src.iter().copied().zip(dst.iter().copied()).map(|(u, v)| {
            let u = u.try_into()?;
            let v = v.try_into()?;
            Ok(core::Edge(u, v))
        }),
    )
}

#[pyclass(module = "glzip")]
struct _CSR {
    csr: core::CSR,
}

#[pymethods]
impl _CSR {
    #[new]
    #[args("*", edge_index = "None", indptr = "None", indices = "None")]
    fn new<'py>(
        edge_index: Option<PyReadonlyArray2<'py, i64>>,
        indptr: Option<PyReadonlyArray1<'py, i64>>,
        indices: Option<PyReadonlyArray1<'py, i64>>,
    ) -> PyResult<Self> {
        if let Some(edge_index) = edge_index {
            let arr = edge_index.as_array();
            let src = arr
                .index_axis(Axis(0), 0)
                .to_slice()
                .ok_or(NotContiguousError)?;
            let dst = arr
                .index_axis(Axis(0), 1)
                .to_slice()
                .ok_or(NotContiguousError)?;
            let csr = core_from_edge_index(src, dst)?;
            Ok(Self { csr })
        } else if let (Some(indptr), Some(indices)) = (indptr, indices) {
            let indptr = indptr.as_slice()?;
            let indices = indices.as_slice()?;
            let csr = core_from_csr(indptr, indices)?;
            Ok(Self { csr })
        } else {
            Err(PyRuntimeError::new_err(
                "provide either edge_index or indptr and indices",
            ))
        }
    }

    fn __str__(&self) -> String {
        format!(
            "glzip.CSR(order={}, size={}, nbytes={})",
            self.csr.order(),
            self.csr.size(),
            self.csr.nbytes()
        )
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

        let (csr, eid) = core::reorder::by_access_probabilites(&self.csr, &train_idx_bitmap[..], &sizes[..]);

        Ok((Self { csr }, eid))
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.csr.nbytes()
    }

    #[getter]
    fn order(&self) -> usize {
        self.csr.order()
    }

    #[getter]
    fn size(&self) -> usize {
        self.csr.size()
    }
}

type Adj = (Py<PyArray2<i64>>, Py<PyArray1<i64>>, Py<PyArray1<i64>>);

fn from_core_adj<'py>(py: Python<'py>, core_adj: core::graph_sage_sampler::Adj) -> PyResult<Adj> 
{
    let edge_array: Array2<u32> = ndarray::stack![Axis(0), Array1::from_vec(core_adj.src), Array1::from_vec(core_adj.dst)];
    let edge_index: &PyArray2<i64> = PyArray2::from_array(py, &edge_array).cast(false)?;
    let e_id = PyArray1::from_vec(py, vec![]);
    let size = numpy::pyarray![py, core_adj.size.0, core_adj.size.1].cast(false)?;
    Ok((edge_index.to_owned(), e_id.to_owned(), size.to_owned()))
}

#[pyclass]
struct _GraphSageSampler
{
    csr: Py<_CSR>,
    sizes: Vec<usize>,
}

#[pymethods]
impl _GraphSageSampler
{
    #[new]
    fn new(csr: Py<_CSR>, sizes: Vec<usize>) -> PyResult<Self>
    {
        Ok(Self { csr, sizes })
    }

    fn sample<'py>(&self, py: Python<'py>, input_nodes: PyReadonlyArray1<'py, i64>) -> PyResult<(Py<PyArray1<i64>>, usize, Vec<Adj>)>
    {
        let csr = &self.csr.as_ref(py).try_borrow()?.csr;
        let sampler = core::graph_sage_sampler::GraphSageSampler::new(csr, &self.sizes[..]);
        let casted = input_nodes.cast(false)?.readonly();
        let (nodes, batch_size, adjs) = sampler.sample(casted.as_slice()?);
        let mut new_adjs = Vec::with_capacity(adjs.len());
        for adj in adjs {
            new_adjs.push(from_core_adj(py, adj)?);
        }

        let nodes = PyArray1::from_vec(py, nodes).cast(false)?;

        Ok((nodes.to_owned(), batch_size, new_adjs))
    }
}

#[pymodule]
fn glzip(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<_CSR>()?;
    m.add_class::<_GraphSageSampler>()?;
    Ok(())
}
