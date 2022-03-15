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
    convert::Infallible
};

use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError,
};

use glzip_core as core;

fn handle_core_err(err: Box<dyn Error + Send + Sync>) -> PyErr
{
    PyRuntimeError::new_err(err.to_string())
}

#[pyclass(module="glzip")]
struct CSR 
{
    csr: core::csr::CSR,
}

#[pymethods]
impl CSR
{
    #[new]
    #[args(
        "*",
        edgelist="None",
        filename="None",
        edges_per_chunk = "None",
        bytes_per_chunk = "None",
        num_threads = "None"
    )]
    fn new(
        edgelist: Option<Vec<(u32, u32)>>,
        filename: Option<String>,
        edges_per_chunk: Option<usize>,
        bytes_per_chunk: Option<usize>,
        num_threads: Option<usize>,
    ) -> PyResult<Self>
    {
        if edgelist.is_none() && filename.is_none() {
            return Ok(Self { csr: core::csr::CSR::new() });
        }

        let mut builder = core::csr::CSRBuilder::new();
        builder = match bytes_per_chunk {
            Some(b) => {
                match builder.bytes_per_chunk(b) {
                    Ok(builder) => builder,
                    Err(builder) => {
                        match edges_per_chunk {
                            Some(e) => builder.edges_per_chunk(e),
                            None => builder,
                        }
                    }
                }
            }
            None => { 
                match edges_per_chunk {
                    Some(e) => builder.edges_per_chunk(e),
                    None => builder,
                }
            }
        };
        builder = match num_threads {
            Some(n) => builder.num_threads(n),
            None => builder,
        };

        Ok(Self { 
            csr: match edgelist {
                Some(e) => {
                    builder.build::<Infallible, _>(e.iter().map(|&(u, v)| Ok([u, v]))).map_err(handle_core_err)?
                }
                None => {
                    match filename {
                        Some(f) => {
                            let iter = core::io::load(f).map_err(handle_core_err)?;
                            builder.build(iter).map_err(handle_core_err)?
                        }
                        None => {
                            unreachable!()
                        }
                    }
                }
            }
        })
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
}   

#[pymodule]
fn glzip(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CSR>()?;
    Ok(())
}
