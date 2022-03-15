# glzip is a graph compression library for graph learning systems
# Copyright (C) 2022 Jacob Konrad
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .glzip import CSR

if __name__ == '__main__':

    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Given a path to an edgelist, compute the compression ratio of the CSR.')
    parser.add_argument('path', metavar='PATH', type=str, help='the path to the edgelist, can be *.csv.gz, *.csv or *.npy')

    args = parser.parse_args()

    csr_ = CSR(filename=args.path)
    
    pre_bytes = ((csr_.order + 1) * 8) + (csr_.size * 4) + 24 

    print(str(csr_))
    print("Compression Ratio: {:.2}".format(csr_.nbytes / pre_bytes))
