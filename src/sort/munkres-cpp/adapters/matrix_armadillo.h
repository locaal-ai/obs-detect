/*
 *   Copyright (c) 2016 Gluttton <gluttton@ukr.net>
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#if !defined(_MUNKRES_ADAPTERS_ARMADILLO_MATRIX_H_)
#define _MUNKRES_ADAPTERS_ARMADILLO_MATRIX_H_

#include "munkres-cpp/matrix_base.h"
#include <armadillo>

namespace munkres_cpp {

template<class T> class matrix_armadillo : public matrix_base<T>, public arma::Mat<T> {
public:
	using matrix_base<T>::begin;
	using matrix_base<T>::end;

	matrix_armadillo(size_t rows, size_t columns) : arma::Mat<T>::Mat(rows, columns)
	{
		std::fill(matrix_base<T>::begin(), matrix_base<T>::end(), T(0));
	}

	matrix_armadillo(const arma::Mat<T> &other) : arma::Mat<T>::Mat(other) {}

	matrix_armadillo<T> &operator=(const arma::Mat<T> &other)
	{
		arma::Mat<T>::operator=(other);
		return *this;
	}

	const T &operator()(size_t row, size_t column) const override
	{
		return arma::Mat<T>::operator()(row, column);
	};

	T &operator()(size_t row, size_t column) override
	{
		return arma::Mat<T>::operator()(row, column);
	}

	size_t columns() const override { return arma::Mat<T>::n_cols; }

	size_t rows() const override { return arma::Mat<T>::n_rows; }

	void resize(size_t rows, size_t columns, T value = T(0)) override
	{
		if (rows == arma::Mat<T>::n_rows && columns == arma::Mat<T>::n_cols) {
			return;
		}

		const auto rows_old = arma::Mat<T>::n_rows;
		const auto columns_old = arma::Mat<T>::n_cols;

		arma::Mat<T>::resize(rows, columns);

		if (rows_old < rows) {
			for (size_t i = rows_old; i < rows; ++i) {
				for (size_t j = 0; j < columns; ++j) {
					this->operator()(i, j) = value;
				}
			}
		}
		if (columns_old < columns) {
			for (size_t i = columns_old; i < columns; ++i) {
				for (size_t j = 0; j < rows; ++j) {
					this->operator()(j, i) = value;
				}
			}
		}
	}

	bool is_zero(size_t row, size_t column) const
	{
		return matrix_base<T>::is_zero(row, column);
	}
};

} // namespace munkres_cpp

#endif /* !defined(_MUNKRES_ADAPTERS_ARMADILLO_MATRIX_H_) */
